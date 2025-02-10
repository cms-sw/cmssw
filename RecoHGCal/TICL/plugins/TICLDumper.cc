// Authors:  Philipp Zehetner, Wahid Redjeb, Aurora Perego, Felice Pantaleo

#include "TTree.h"
#include "TFile.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <variant>

#include <memory>  // unique_ptr
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/HGCalReco/interface/Common.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

// TFileService
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

using TracksterToTracksterMap =
    ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore, std::vector<ticl::Trackster>, std::vector<ticl::Trackster>>;
// Helper class for geometry, magnetic field, etc
class DetectorTools {
public:
  DetectorTools(const HGCalDDDConstants& hgcons,
                const CaloGeometry& geom,
                const MagneticField& bfieldH,
                const Propagator& propH)
      : hgcons(hgcons), rhtools(), bfield(bfieldH), propagator(propH) {
    rhtools.setGeometry(geom);

    // build disks at HGCal front & EM-Had interface for track propagation
    float zVal = hgcons.waferZ(1, true);
    std::pair<float, float> rMinMax = hgcons.rangeR(zVal, true);

    float zVal_interface = rhtools.getPositionLayer(rhtools.lastLayerEE()).z();
    std::pair<float, float> rMinMax_interface = hgcons.rangeR(zVal_interface, true);

    for (int iSide = 0; iSide < 2; ++iSide) {
      float zSide = (iSide == 0) ? (-1. * zVal) : zVal;
      firstDisk_[iSide] = std::make_unique<GeomDet>(
          Disk::build(Disk::PositionType(0, 0, zSide),
                      Disk::RotationType(),
                      SimpleDiskBounds(rMinMax.first, rMinMax.second, zSide - 0.5, zSide + 0.5))
              .get());

      zSide = (iSide == 0) ? (-1. * zVal_interface) : zVal_interface;
      interfaceDisk_[iSide] = std::make_unique<GeomDet>(
          Disk::build(Disk::PositionType(0, 0, zSide),
                      Disk::RotationType(),
                      SimpleDiskBounds(rMinMax_interface.first, rMinMax_interface.second, zSide - 0.5, zSide + 0.5))
              .get());
    }
  }

  const HGCalDDDConstants& hgcons;
  std::unique_ptr<GeomDet> firstDisk_[2];
  std::unique_ptr<GeomDet> interfaceDisk_[2];
  hgcal::RecHitTools rhtools;
  const MagneticField& bfield;
  const Propagator& propagator;
};

// Helper class that dumps a single trackster collection (either tracksters or simTracksters)
class TracksterDumperHelper {
public:
  enum class TracksterType {
    Trackster,       ///< Regular trackster (from RECO)
    SimTracksterCP,  ///< SimTrackster from CaloParticle
    SimTracksterSC   ///< SimTrackster from SimCluster
  };

  static TracksterType tracksterTypeFromString(std::string str) {
    if (str == "Trackster")
      return TracksterType::Trackster;
    if (str == "SimTracksterCP")
      return TracksterType::SimTracksterCP;
    if (str == "SimTracksterSC")
      return TracksterType::SimTracksterSC;
    throw std::runtime_error("TICLDumper : TracksterDumperHelper : Invalid trackster type " + str);
  }

  /** tracksterType : dtermines additional information that will be saved (calo truth information, track information) */
  TracksterDumperHelper(TracksterType tracksterType = TracksterType::Trackster) : tracksterType_(tracksterType) {}

  /**
   * To be called once after tree creation. eventId_ should be a pointer to the EventID.
   * *Do not copy/move or resize vector holding object after calling this function*
  */
  void initTree(TTree* trackster_tree_, edm::EventID* eventId_) {
    trackster_tree_->Branch("event", eventId_);
    trackster_tree_->Branch("NTracksters", &nTracksters);
    trackster_tree_->Branch("NClusters", &nClusters);
    trackster_tree_->Branch("time", &trackster_time);
    trackster_tree_->Branch("timeError", &trackster_timeError);
    trackster_tree_->Branch("regressed_energy", &trackster_regressed_energy);
    trackster_tree_->Branch("raw_energy", &trackster_raw_energy);
    trackster_tree_->Branch("raw_em_energy", &trackster_raw_em_energy);
    trackster_tree_->Branch("raw_pt", &trackster_raw_pt);
    trackster_tree_->Branch("raw_em_pt", &trackster_raw_em_pt);
    trackster_tree_->Branch("barycenter_x", &trackster_barycenter_x);
    trackster_tree_->Branch("barycenter_y", &trackster_barycenter_y);
    trackster_tree_->Branch("barycenter_z", &trackster_barycenter_z);
    trackster_tree_->Branch("barycenter_eta", &trackster_barycenter_eta);
    trackster_tree_->Branch("barycenter_phi", &trackster_barycenter_phi);
    trackster_tree_->Branch("EV1", &trackster_EV1);
    trackster_tree_->Branch("EV2", &trackster_EV2);
    trackster_tree_->Branch("EV3", &trackster_EV3);
    trackster_tree_->Branch("eVector0_x", &trackster_eVector0_x);
    trackster_tree_->Branch("eVector0_y", &trackster_eVector0_y);
    trackster_tree_->Branch("eVector0_z", &trackster_eVector0_z);
    trackster_tree_->Branch("sigmaPCA1", &trackster_sigmaPCA1);
    trackster_tree_->Branch("sigmaPCA2", &trackster_sigmaPCA2);
    trackster_tree_->Branch("sigmaPCA3", &trackster_sigmaPCA3);
    if (tracksterType_ != TracksterType::Trackster) {
      trackster_tree_->Branch("regressed_pt", &simtrackster_regressed_pt);
      trackster_tree_->Branch("pdgID", &simtrackster_pdgID);
      trackster_tree_->Branch("trackIdx", &simtrackster_trackIdx);
      trackster_tree_->Branch("trackTime", &simtrackster_trackTime);
      trackster_tree_->Branch("timeBoundary", &simtrackster_timeBoundary);
      trackster_tree_->Branch("boundaryX", &simtrackster_boundaryX);
      trackster_tree_->Branch("boundaryY", &simtrackster_boundaryY);
      trackster_tree_->Branch("boundaryZ", &simtrackster_boundaryZ);
      trackster_tree_->Branch("boundaryEta", &simtrackster_boundaryEta);
      trackster_tree_->Branch("boundaryPhi", &simtrackster_boundaryPhi);
      trackster_tree_->Branch("boundaryPx", &simtrackster_boundaryPx);
      trackster_tree_->Branch("boundaryPy", &simtrackster_boundaryPy);
      trackster_tree_->Branch("boundaryPz", &simtrackster_boundaryPz);
      trackster_tree_->Branch("track_boundaryX", &simtrackster_track_boundaryX);
      trackster_tree_->Branch("track_boundaryY", &simtrackster_track_boundaryY);
      trackster_tree_->Branch("track_boundaryZ", &simtrackster_track_boundaryZ);
      trackster_tree_->Branch("track_boundaryEta", &simtrackster_track_boundaryEta);
      trackster_tree_->Branch("track_boundaryPhi", &simtrackster_track_boundaryPhi);
      trackster_tree_->Branch("track_boundaryPx", &simtrackster_track_boundaryPx);
      trackster_tree_->Branch("track_boundaryPy", &simtrackster_track_boundaryPy);
      trackster_tree_->Branch("track_boundaryPz", &simtrackster_track_boundaryPz);
    }
    trackster_tree_->Branch("id_probabilities", &trackster_id_probabilities);
    trackster_tree_->Branch("vertices_indexes", &trackster_vertices_indexes);
    trackster_tree_->Branch("vertices_x", &trackster_vertices_x);
    trackster_tree_->Branch("vertices_y", &trackster_vertices_y);
    trackster_tree_->Branch("vertices_z", &trackster_vertices_z);
    trackster_tree_->Branch("vertices_time", &trackster_vertices_time);
    trackster_tree_->Branch("vertices_timeErr", &trackster_vertices_timeErr);
    trackster_tree_->Branch("vertices_energy", &trackster_vertices_energy);
    trackster_tree_->Branch("vertices_correctedEnergy", &trackster_vertices_correctedEnergy);
    trackster_tree_->Branch("vertices_correctedEnergyUncertainty", &trackster_vertices_correctedEnergyUncertainty);
    trackster_tree_->Branch("vertices_multiplicity", &trackster_vertices_multiplicity);
  }

  void clearVariables() {
    nTracksters = 0;
    nClusters = 0;
    trackster_time.clear();
    trackster_timeError.clear();
    trackster_regressed_energy.clear();
    trackster_raw_energy.clear();
    trackster_raw_em_energy.clear();
    trackster_raw_pt.clear();
    trackster_raw_em_pt.clear();
    trackster_barycenter_x.clear();
    trackster_barycenter_y.clear();
    trackster_barycenter_z.clear();
    trackster_EV1.clear();
    trackster_EV2.clear();
    trackster_EV3.clear();
    trackster_eVector0_x.clear();
    trackster_eVector0_y.clear();
    trackster_eVector0_z.clear();
    trackster_sigmaPCA1.clear();
    trackster_sigmaPCA2.clear();
    trackster_sigmaPCA3.clear();

    simtrackster_regressed_pt.clear();
    simtrackster_pdgID.clear();
    simtrackster_trackIdx.clear();
    simtrackster_trackTime.clear();
    simtrackster_timeBoundary.clear();
    simtrackster_boundaryX.clear();
    simtrackster_boundaryY.clear();
    simtrackster_boundaryZ.clear();
    simtrackster_boundaryEta.clear();
    simtrackster_boundaryPhi.clear();
    simtrackster_boundaryPx.clear();
    simtrackster_boundaryPy.clear();
    simtrackster_boundaryPz.clear();
    simtrackster_track_boundaryX.clear();
    simtrackster_track_boundaryY.clear();
    simtrackster_track_boundaryZ.clear();
    simtrackster_track_boundaryEta.clear();
    simtrackster_track_boundaryPhi.clear();
    simtrackster_track_boundaryPx.clear();
    simtrackster_track_boundaryPy.clear();
    simtrackster_track_boundaryPz.clear();

    trackster_barycenter_eta.clear();
    trackster_barycenter_phi.clear();
    trackster_id_probabilities.clear();
    trackster_vertices_indexes.clear();
    trackster_vertices_x.clear();
    trackster_vertices_y.clear();
    trackster_vertices_z.clear();
    trackster_vertices_time.clear();
    trackster_vertices_timeErr.clear();
    trackster_vertices_energy.clear();
    trackster_vertices_correctedEnergy.clear();
    trackster_vertices_correctedEnergyUncertainty.clear();
    trackster_vertices_multiplicity.clear();
  }

  void fillFromEvent(std::vector<ticl::Trackster> const& tracksters,
                     std::vector<reco::CaloCluster> const& clusters,
                     edm::ValueMap<std::pair<float, float>> const& layerClustersTimes,
                     DetectorTools const& detectorTools,
                     edm::Handle<std::vector<SimCluster>> simClusters_h,
                     edm::Handle<std::vector<CaloParticle>> caloparticles_h,
                     std::vector<reco::Track> const& tracks) {
    nTracksters = tracksters.size();
    nClusters = clusters.size();
    for (auto trackster_iterator = tracksters.begin(); trackster_iterator != tracksters.end(); ++trackster_iterator) {
      //per-trackster analysis
      trackster_time.push_back(trackster_iterator->time());
      trackster_timeError.push_back(trackster_iterator->timeError());
      trackster_regressed_energy.push_back(trackster_iterator->regressed_energy());
      trackster_raw_energy.push_back(trackster_iterator->raw_energy());
      trackster_raw_em_energy.push_back(trackster_iterator->raw_em_energy());
      trackster_raw_pt.push_back(trackster_iterator->raw_pt());
      trackster_raw_em_pt.push_back(trackster_iterator->raw_em_pt());
      trackster_barycenter_x.push_back(trackster_iterator->barycenter().x());
      trackster_barycenter_y.push_back(trackster_iterator->barycenter().y());
      trackster_barycenter_z.push_back(trackster_iterator->barycenter().z());
      trackster_barycenter_eta.push_back(trackster_iterator->barycenter().eta());
      trackster_barycenter_phi.push_back(trackster_iterator->barycenter().phi());
      trackster_EV1.push_back(trackster_iterator->eigenvalues()[0]);
      trackster_EV2.push_back(trackster_iterator->eigenvalues()[1]);
      trackster_EV3.push_back(trackster_iterator->eigenvalues()[2]);
      trackster_eVector0_x.push_back((trackster_iterator->eigenvectors()[0]).x());
      trackster_eVector0_y.push_back((trackster_iterator->eigenvectors()[0]).y());
      trackster_eVector0_z.push_back((trackster_iterator->eigenvectors()[0]).z());
      trackster_sigmaPCA1.push_back(trackster_iterator->sigmasPCA()[0]);
      trackster_sigmaPCA2.push_back(trackster_iterator->sigmasPCA()[1]);
      trackster_sigmaPCA3.push_back(trackster_iterator->sigmasPCA()[2]);

      if (tracksterType_ != TracksterType::Trackster) {  // is simtrackster
        auto const& simclusters = *simClusters_h;
        auto const& caloparticles = *caloparticles_h;

        simtrackster_timeBoundary.push_back(trackster_iterator->boundaryTime());

        if (tracksterType_ == TracksterType::SimTracksterCP)
          simtrackster_pdgID.push_back(caloparticles[trackster_iterator->seedIndex()].pdgId());
        else if (tracksterType_ == TracksterType::SimTracksterSC)
          simtrackster_pdgID.push_back(simclusters[trackster_iterator->seedIndex()].pdgId());

        using CaloObjectVariant = std::variant<CaloParticle, SimCluster>;
        CaloObjectVariant caloObj;
        if (trackster_iterator->seedID() == caloparticles_h.id()) {
          caloObj = caloparticles[trackster_iterator->seedIndex()];
        } else {
          caloObj = simclusters[trackster_iterator->seedIndex()];
        }

        auto const& simTrack = std::visit([](auto&& obj) { return obj.g4Tracks()[0]; }, caloObj);
        auto const& caloPt = std::visit([](auto&& obj) { return obj.pt(); }, caloObj);
        simtrackster_regressed_pt.push_back(caloPt);
        if (simTrack.crossedBoundary()) {
          simtrackster_boundaryX.push_back(simTrack.getPositionAtBoundary().x());
          simtrackster_boundaryY.push_back(simTrack.getPositionAtBoundary().y());
          simtrackster_boundaryZ.push_back(simTrack.getPositionAtBoundary().z());
          simtrackster_boundaryEta.push_back(simTrack.getPositionAtBoundary().eta());
          simtrackster_boundaryPhi.push_back(simTrack.getPositionAtBoundary().phi());
          simtrackster_boundaryPx.push_back(simTrack.getMomentumAtBoundary().x());
          simtrackster_boundaryPy.push_back(simTrack.getMomentumAtBoundary().y());
          simtrackster_boundaryPz.push_back(simTrack.getMomentumAtBoundary().z());
        } else {
          simtrackster_boundaryX.push_back(-999);
          simtrackster_boundaryY.push_back(-999);
          simtrackster_boundaryZ.push_back(-999);
          simtrackster_boundaryEta.push_back(-999);
          simtrackster_boundaryPhi.push_back(-999);
          simtrackster_boundaryPx.push_back(-999);
          simtrackster_boundaryPy.push_back(-999);
          simtrackster_boundaryPz.push_back(-999);
        }

        auto const trackIdx = trackster_iterator->trackIdx();
        simtrackster_trackIdx.push_back(trackIdx);
        if (trackIdx != -1) {
          const auto& track = tracks[trackIdx];

          int iSide = int(track.eta() > 0);

          const auto& fts = trajectoryStateTransform::outerFreeState((track), &detectorTools.bfield);
          // to the HGCal front
          const auto& tsos = detectorTools.propagator.propagate(fts, detectorTools.firstDisk_[iSide]->surface());
          if (tsos.isValid()) {
            const auto& globalPos = tsos.globalPosition();
            const auto& globalMom = tsos.globalMomentum();
            simtrackster_track_boundaryX.push_back(globalPos.x());
            simtrackster_track_boundaryY.push_back(globalPos.y());
            simtrackster_track_boundaryZ.push_back(globalPos.z());
            simtrackster_track_boundaryEta.push_back(globalPos.eta());
            simtrackster_track_boundaryPhi.push_back(globalPos.phi());
            simtrackster_track_boundaryPx.push_back(globalMom.x());
            simtrackster_track_boundaryPy.push_back(globalMom.y());
            simtrackster_track_boundaryPz.push_back(globalMom.z());
            simtrackster_trackTime.push_back(track.t0());
          } else {
            simtrackster_track_boundaryX.push_back(-999);
            simtrackster_track_boundaryY.push_back(-999);
            simtrackster_track_boundaryZ.push_back(-999);
            simtrackster_track_boundaryEta.push_back(-999);
            simtrackster_track_boundaryPhi.push_back(-999);
            simtrackster_track_boundaryPx.push_back(-999);
            simtrackster_track_boundaryPy.push_back(-999);
            simtrackster_track_boundaryPz.push_back(-999);
          }
        } else {
          simtrackster_track_boundaryX.push_back(-999);
          simtrackster_track_boundaryY.push_back(-999);
          simtrackster_track_boundaryZ.push_back(-999);
          simtrackster_track_boundaryEta.push_back(-999);
          simtrackster_track_boundaryPhi.push_back(-999);
          simtrackster_track_boundaryPx.push_back(-999);
          simtrackster_track_boundaryPy.push_back(-999);
          simtrackster_track_boundaryPz.push_back(-999);
        }
      }

      std::vector<float> id_probs;
      for (size_t i = 0; i < 8; i++)
        id_probs.push_back(trackster_iterator->id_probabilities(i));
      trackster_id_probabilities.push_back(id_probs);

      // Clusters
      std::vector<uint32_t> vertices_indexes;
      std::vector<float> vertices_x;
      std::vector<float> vertices_y;
      std::vector<float> vertices_z;
      std::vector<float> vertices_time;
      std::vector<float> vertices_timeErr;
      std::vector<float> vertices_energy;
      std::vector<float> vertices_correctedEnergy;
      std::vector<float> vertices_correctedEnergyUncertainty;
      for (auto idx : trackster_iterator->vertices()) {
        vertices_indexes.push_back(idx);
        const auto& associated_cluster = clusters[idx];
        vertices_x.push_back(associated_cluster.x());
        vertices_y.push_back(associated_cluster.y());
        vertices_z.push_back(associated_cluster.z());
        vertices_energy.push_back(associated_cluster.energy());
        vertices_correctedEnergy.push_back(associated_cluster.correctedEnergy());
        vertices_correctedEnergyUncertainty.push_back(associated_cluster.correctedEnergyUncertainty());
        vertices_time.push_back(layerClustersTimes.get(idx).first);
        vertices_timeErr.push_back(layerClustersTimes.get(idx).second);
      }
      trackster_vertices_indexes.push_back(vertices_indexes);
      trackster_vertices_x.push_back(vertices_x);
      trackster_vertices_y.push_back(vertices_y);
      trackster_vertices_z.push_back(vertices_z);
      trackster_vertices_time.push_back(vertices_time);
      trackster_vertices_timeErr.push_back(vertices_timeErr);
      trackster_vertices_energy.push_back(vertices_energy);
      trackster_vertices_correctedEnergy.push_back(vertices_correctedEnergy);
      trackster_vertices_correctedEnergyUncertainty.push_back(vertices_correctedEnergyUncertainty);

      // Multiplicity
      std::vector<float> vertices_multiplicity;
      for (auto multiplicity : trackster_iterator->vertex_multiplicity()) {
        vertices_multiplicity.push_back(multiplicity);
      }
      trackster_vertices_multiplicity.push_back(vertices_multiplicity);
    }
  }

private:
  TracksterType tracksterType_;

  unsigned int nTracksters;
  unsigned int nClusters;
  std::vector<float> trackster_time;
  std::vector<float> trackster_timeError;
  std::vector<float> trackster_regressed_energy;
  std::vector<float> trackster_raw_energy;
  std::vector<float> trackster_raw_em_energy;
  std::vector<float> trackster_raw_pt;
  std::vector<float> trackster_raw_em_pt;
  std::vector<float> trackster_barycenter_x;
  std::vector<float> trackster_barycenter_y;
  std::vector<float> trackster_barycenter_z;
  std::vector<float> trackster_EV1;
  std::vector<float> trackster_EV2;
  std::vector<float> trackster_EV3;
  std::vector<float> trackster_eVector0_x;
  std::vector<float> trackster_eVector0_y;
  std::vector<float> trackster_eVector0_z;
  std::vector<float> trackster_sigmaPCA1;
  std::vector<float> trackster_sigmaPCA2;
  std::vector<float> trackster_sigmaPCA3;
  std::vector<float> trackster_barycenter_eta;
  std::vector<float> trackster_barycenter_phi;

  // for simtrackster
  std::vector<float> simtrackster_regressed_pt;
  std::vector<int> simtrackster_pdgID;
  std::vector<int> simtrackster_trackIdx;
  std::vector<float> simtrackster_trackTime;
  std::vector<float> simtrackster_timeBoundary;
  std::vector<float> simtrackster_boundaryX;
  std::vector<float> simtrackster_boundaryY;
  std::vector<float> simtrackster_boundaryZ;
  std::vector<float> simtrackster_boundaryEta;
  std::vector<float> simtrackster_boundaryPhi;
  std::vector<float> simtrackster_boundaryPx;
  std::vector<float> simtrackster_boundaryPy;
  std::vector<float> simtrackster_boundaryPz;
  std::vector<float> simtrackster_track_boundaryX;
  std::vector<float> simtrackster_track_boundaryY;
  std::vector<float> simtrackster_track_boundaryZ;
  std::vector<float> simtrackster_track_boundaryEta;
  std::vector<float> simtrackster_track_boundaryPhi;
  std::vector<float> simtrackster_track_boundaryPx;
  std::vector<float> simtrackster_track_boundaryPy;
  std::vector<float> simtrackster_track_boundaryPz;

  std::vector<std::vector<float>> trackster_id_probabilities;
  std::vector<std::vector<uint32_t>> trackster_vertices_indexes;
  std::vector<std::vector<float>> trackster_vertices_x;
  std::vector<std::vector<float>> trackster_vertices_y;
  std::vector<std::vector<float>> trackster_vertices_z;
  std::vector<std::vector<float>> trackster_vertices_time;
  std::vector<std::vector<float>> trackster_vertices_timeErr;
  std::vector<std::vector<float>> trackster_vertices_energy;
  std::vector<std::vector<float>> trackster_vertices_correctedEnergy;
  std::vector<std::vector<float>> trackster_vertices_correctedEnergyUncertainty;
  std::vector<std::vector<float>> trackster_vertices_multiplicity;
};

// Helper class to dump a TracksterToSimTrackster association map (dumps recoToSim and simToReco at the same time)
class TracksterToSimTracksterAssociationHelper {
public:
  /**
   * To be called once after tree creation. Output branches will be named prefix_recoToSim/simToReco_suffix_score/sharedE/
   * branchPrefix : for example tsCLUE3D. branchSuffix : usually one of SC or CP.
   * *Do not copy/move or resize vector holding object after calling this function*
  */
  void initTree(TTree* tree, std::string branchPrefix, std::string branchSuffix) {
    tree->Branch((branchPrefix + "_recoToSim_" + branchSuffix).c_str(), &recoToSim);
    tree->Branch((branchPrefix + "_recoToSim_" + branchSuffix + "_score").c_str(), &recoToSim_score);
    tree->Branch((branchPrefix + "_recoToSim_" + branchSuffix + "_sharedE").c_str(), &recoToSim_sharedE);
    tree->Branch((branchPrefix + "_simToReco_" + branchSuffix).c_str(), &simToReco);
    tree->Branch((branchPrefix + "_simToReco_" + branchSuffix + "_score").c_str(), &simToReco_score);
    tree->Branch((branchPrefix + "_simToReco_" + branchSuffix + "_sharedE").c_str(), &simToReco_sharedE);
  }

  void clearVariables() {
    recoToSim.clear();
    recoToSim_score.clear();
    recoToSim_sharedE.clear();
    simToReco.clear();
    simToReco_score.clear();
    simToReco_sharedE.clear();
  }

  void fillFromEvent(TracksterToTracksterMap const& recoToSimMap, TracksterToTracksterMap const& simToRecoMap) {
    // Reco -> Sim
    const auto numberOfTracksters = recoToSimMap.getMap().size();
    recoToSim.resize(numberOfTracksters);
    recoToSim_score.resize(numberOfTracksters);
    recoToSim_sharedE.resize(numberOfTracksters);

    for (size_t i = 0; i < numberOfTracksters; ++i) {
      for (const auto& simTracksterElement : recoToSimMap[i]) {
        recoToSim[i].push_back(simTracksterElement.index());
        recoToSim_sharedE[i].push_back(simTracksterElement.sharedEnergy());
        recoToSim_score[i].push_back(simTracksterElement.score());
      }
    }

    // Sim -> Reco
    const auto numberOfSimTracksters = simToRecoMap.getMap().size();
    simToReco.resize(numberOfSimTracksters);
    simToReco_score.resize(numberOfSimTracksters);
    simToReco_sharedE.resize(numberOfSimTracksters);

    for (size_t i = 0; i < numberOfSimTracksters; ++i) {
      for (const auto& recoTracksterElement : simToRecoMap[i]) {
        simToReco[i].push_back(recoTracksterElement.index());
        simToReco_sharedE[i].push_back(recoTracksterElement.sharedEnergy());
        simToReco_score[i].push_back(recoTracksterElement.score());
      }
    }
  }

private:
  std::vector<std::vector<uint32_t>> recoToSim;
  std::vector<std::vector<float>> recoToSim_score;
  std::vector<std::vector<float>> recoToSim_sharedE;
  std::vector<std::vector<uint32_t>> simToReco;
  std::vector<std::vector<float>> simToReco_score;
  std::vector<std::vector<float>> simToReco_sharedE;
};

class TICLDumper : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit TICLDumper(const edm::ParameterSet&);
  ~TICLDumper() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  typedef ticl::Vector Vector;
  typedef std::vector<double> Vec;

private:
  void beginJob() override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {};
  void endJob() override;

  // Define Tokens
  const std::vector<edm::ParameterSet>
      tracksters_parameterSets_;  ///< A parameter set for each trackster collection to dump (giving tree name, etc)
  std::vector<edm::EDGetTokenT<std::vector<ticl::Trackster>>>
      tracksters_token_;  ///< a token for each trackster collection to dump
  std::vector<TracksterDumperHelper>
      tracksters_dumperHelpers_;         ///< the dumper helpers for each trackster collection to dump
  std::vector<TTree*> tracksters_trees;  ///< TTree for each trackster collection to dump

  const edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksters_in_candidate_token_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> layer_clusters_token_;
  const edm::EDGetTokenT<std::vector<TICLCandidate>> ticl_candidates_token_;
  const edm::EDGetTokenT<std::vector<ticl::Trackster>>
      ticl_candidates_tracksters_token_;  ///< trackster collection used by TICLCandidate
  const edm::EDGetTokenT<std::vector<reco::Track>> tracks_token_;
  const edm::EDGetTokenT<std::vector<bool>> tracks_mask_token_;
  const edm::EDGetTokenT<edm::ValueMap<float>> tracks_time_token_;
  const edm::EDGetTokenT<edm::ValueMap<float>> tracks_time_quality_token_;
  const edm::EDGetTokenT<edm::ValueMap<float>> tracks_time_err_token_;
  const edm::EDGetTokenT<edm::ValueMap<float>> tracks_beta_token_;
  const edm::EDGetTokenT<edm::ValueMap<float>> tracks_time_mtd_token_;
  const edm::EDGetTokenT<edm::ValueMap<float>> tracks_time_mtd_err_token_;
  const edm::EDGetTokenT<edm::ValueMap<GlobalPoint>> tracks_pos_mtd_token_;
  const edm::EDGetTokenT<std::vector<double>> hgcaltracks_x_token_;
  const edm::EDGetTokenT<std::vector<double>> hgcaltracks_y_token_;
  const edm::EDGetTokenT<std::vector<double>> hgcaltracks_z_token_;
  const edm::EDGetTokenT<std::vector<double>> hgcaltracks_eta_token_;
  const edm::EDGetTokenT<std::vector<double>> hgcaltracks_phi_token_;
  const edm::EDGetTokenT<std::vector<double>> hgcaltracks_px_token_;
  const edm::EDGetTokenT<std::vector<double>> hgcaltracks_py_token_;
  const edm::EDGetTokenT<std::vector<double>> hgcaltracks_pz_token_;
  const edm::EDGetTokenT<std::vector<reco::Muon>> muons_token_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> clustersTime_token_;
  const edm::EDGetTokenT<std::vector<int>> tracksterSeeds_token_;
  edm::EDGetTokenT<std::vector<std::vector<unsigned int>>> superclustering_linkedResultTracksters_token;
  edm::EDGetTokenT<reco::SuperClusterCollection> recoSuperClusters_token;
  edm::EDGetTokenT<reco::CaloClusterCollection> recoSuperClusters_caloClusters_token;
  edm::EDGetTokenT<std::vector<ticl::Trackster>> recoSuperClusters_sourceTracksters_token;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometry_token_;
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> simTracksters_SC_token_;  // needed for simticlcandidate
  const edm::EDGetTokenT<std::vector<TICLCandidate>> simTICLCandidate_token_;

  // associators
  const std::vector<edm::ParameterSet>
      associations_parameterSets_;  ///< A parameter set for each associator collection to dump (with treeName, etc)
  std::vector<edm::EDGetTokenT<TracksterToTracksterMap>>
      associations_simToReco_token_;  ///< The tokens for each assocation
  std::vector<edm::EDGetTokenT<TracksterToTracksterMap>> associations_recoToSim_token_;
  std::vector<TracksterToSimTracksterAssociationHelper>
      associations_dumperHelpers_;  ///< the dumper helpers for each association map to dump

  TTree* associations_tree_;

  const edm::EDGetTokenT<std::vector<SimCluster>> simclusters_token_;
  const edm::EDGetTokenT<std::vector<CaloParticle>> caloparticles_token_;

  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometry_token_;
  const std::string detector_;
  const std::string propName_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bfield_token_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagator_token_;
  edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> hdc_token_;
  std::unique_ptr<DetectorTools> detectorTools_;
  bool saveLCs_;
  bool saveSuperclustering_;
  bool saveSuperclusteringDNNScore_;
  bool saveRecoSuperclusters_;
  bool saveTICLCandidate_;
  bool saveSimTICLCandidate_;
  bool saveTracks_;

  // Output tree
  TTree* tree_;

  void clearVariables();

  // Variables for branches
  edm::EventID eventId_;
  unsigned int nclusters_;

  std::vector<std::vector<unsigned int>>
      superclustering_linkedResultTracksters;  // Map of indices from superclusteredTracksters collection back into ticlTrackstersCLUE3DEM collection
  // reco::SuperCluster dump
  std::vector<double> recoSuperCluster_rawEnergy;
  std::vector<double> recoSuperCluster_energy;
  std::vector<double> recoSuperCluster_correctedEnergy;
  std::vector<double> recoSuperCluster_position_x;
  std::vector<double> recoSuperCluster_position_y;
  std::vector<double> recoSuperCluster_position_z;
  std::vector<double> recoSuperCluster_position_eta;
  std::vector<double> recoSuperCluster_position_phi;
  std::vector<uint32_t>
      recoSuperCluster_seedTs;  ///< Index to seed trackster (into the trackster collection used to make superclusters, given by config recoSuperClusters_sourceTracksterCollection)
  std::vector<std::vector<uint32_t>>
      recoSuperCluster_constituentTs;  ///< Indices to all tracksters inside the supercluster (same)

  std::vector<float> simTICLCandidate_raw_energy;
  std::vector<float> simTICLCandidate_regressed_energy;
  std::vector<std::vector<int>> simTICLCandidate_simTracksterCPIndex;
  std::vector<float> simTICLCandidate_boundaryX;
  std::vector<float> simTICLCandidate_boundaryY;
  std::vector<float> simTICLCandidate_boundaryZ;
  std::vector<float> simTICLCandidate_boundaryPx;
  std::vector<float> simTICLCandidate_boundaryPy;
  std::vector<float> simTICLCandidate_boundaryPz;
  std::vector<float> simTICLCandidate_caloParticleMass;
  std::vector<float> simTICLCandidate_time;
  std::vector<int> simTICLCandidate_pdgId;
  std::vector<int> simTICLCandidate_charge;
  std::vector<int> simTICLCandidate_track_in_candidate;

  // from TICLCandidate, product of linking
  size_t nCandidates;
  std::vector<int> candidate_charge;
  std::vector<int> candidate_pdgId;
  std::vector<float> candidate_energy;
  std::vector<float> candidate_raw_energy;
  std::vector<double> candidate_px;
  std::vector<double> candidate_py;
  std::vector<double> candidate_pz;
  std::vector<float> candidate_time;
  std::vector<float> candidate_time_err;
  std::vector<std::vector<float>> candidate_id_probabilities;
  std::vector<std::vector<uint32_t>> tracksters_in_candidate;
  std::vector<int> track_in_candidate;

  // Layer clusters
  std::vector<uint32_t> cluster_seedID;
  std::vector<float> cluster_energy;
  std::vector<float> cluster_correctedEnergy;
  std::vector<float> cluster_correctedEnergyUncertainty;
  std::vector<float> cluster_position_x;
  std::vector<float> cluster_position_y;
  std::vector<float> cluster_position_z;
  std::vector<float> cluster_position_eta;
  std::vector<float> cluster_position_phi;
  std::vector<unsigned int> cluster_layer_id;
  std::vector<int> cluster_type;
  std::vector<float> cluster_time;
  std::vector<float> cluster_timeErr;
  std::vector<uint32_t> cluster_number_of_hits;

  // Tracks
  std::vector<unsigned int> track_id;
  std::vector<float> track_hgcal_x;
  std::vector<float> track_hgcal_y;
  std::vector<float> track_hgcal_z;
  std::vector<float> track_hgcal_px;
  std::vector<float> track_hgcal_py;
  std::vector<float> track_hgcal_pz;
  std::vector<float> track_hgcal_eta;
  std::vector<float> track_hgcal_phi;
  std::vector<float> track_hgcal_pt;
  std::vector<float> track_pt;
  std::vector<int> track_quality;
  std::vector<int> track_missing_outer_hits;
  std::vector<int> track_missing_inner_hits;
  std::vector<int> track_charge;
  std::vector<double> track_time;
  std::vector<float> track_time_quality;
  std::vector<float> track_time_err;
  std::vector<float> track_beta;
  std::vector<float> track_time_mtd;
  std::vector<float> track_time_mtd_err;
  std::vector<GlobalPoint> track_pos_mtd;
  std::vector<int> track_nhits;
  std::vector<int> track_isMuon;
  std::vector<int> track_isTrackerMuon;

  TTree* cluster_tree_;
  TTree* candidate_tree_;
  TTree* superclustering_tree_;
  TTree* tracks_tree_;
  TTree* simTICLCandidate_tree;
};

void TICLDumper::clearVariables() {
  // event info
  nclusters_ = 0;

  for (TracksterDumperHelper& tsDumper : tracksters_dumperHelpers_) {
    tsDumper.clearVariables();
  }

  superclustering_linkedResultTracksters.clear();

  recoSuperCluster_rawEnergy.clear();
  recoSuperCluster_energy.clear();
  recoSuperCluster_correctedEnergy.clear();
  recoSuperCluster_position_x.clear();
  recoSuperCluster_position_y.clear();
  recoSuperCluster_position_z.clear();
  recoSuperCluster_position_eta.clear();
  recoSuperCluster_position_phi.clear();
  recoSuperCluster_seedTs.clear();
  recoSuperCluster_constituentTs.clear();

  simTICLCandidate_raw_energy.clear();
  simTICLCandidate_regressed_energy.clear();
  simTICLCandidate_simTracksterCPIndex.clear();
  simTICLCandidate_boundaryX.clear();
  simTICLCandidate_boundaryY.clear();
  simTICLCandidate_boundaryZ.clear();
  simTICLCandidate_boundaryPx.clear();
  simTICLCandidate_boundaryPy.clear();
  simTICLCandidate_boundaryPz.clear();
  simTICLCandidate_time.clear();
  simTICLCandidate_caloParticleMass.clear();
  simTICLCandidate_pdgId.clear();
  simTICLCandidate_charge.clear();
  simTICLCandidate_track_in_candidate.clear();

  nCandidates = 0;
  candidate_charge.clear();
  candidate_pdgId.clear();
  candidate_energy.clear();
  candidate_raw_energy.clear();
  candidate_px.clear();
  candidate_py.clear();
  candidate_pz.clear();
  candidate_time.clear();
  candidate_time_err.clear();
  candidate_id_probabilities.clear();
  tracksters_in_candidate.clear();
  track_in_candidate.clear();

  for (auto& helper : associations_dumperHelpers_) {
    helper.clearVariables();
  }

  cluster_seedID.clear();
  cluster_energy.clear();
  cluster_correctedEnergy.clear();
  cluster_correctedEnergyUncertainty.clear();
  cluster_position_x.clear();
  cluster_position_y.clear();
  cluster_position_z.clear();
  cluster_position_eta.clear();
  cluster_position_phi.clear();
  cluster_layer_id.clear();
  cluster_type.clear();
  cluster_time.clear();
  cluster_timeErr.clear();
  cluster_number_of_hits.clear();

  track_id.clear();
  track_hgcal_x.clear();
  track_hgcal_y.clear();
  track_hgcal_z.clear();
  track_hgcal_eta.clear();
  track_hgcal_phi.clear();
  track_hgcal_px.clear();
  track_hgcal_py.clear();
  track_hgcal_pz.clear();
  track_hgcal_pt.clear();
  track_quality.clear();
  track_pt.clear();
  track_missing_outer_hits.clear();
  track_missing_inner_hits.clear();
  track_charge.clear();
  track_time.clear();
  track_time_quality.clear();
  track_time_err.clear();
  track_beta.clear();
  track_time_mtd.clear();
  track_time_mtd_err.clear();
  track_pos_mtd.clear();
  track_nhits.clear();
  track_isMuon.clear();
  track_isTrackerMuon.clear();
};

TICLDumper::TICLDumper(const edm::ParameterSet& ps)
    : tracksters_parameterSets_(ps.getParameter<std::vector<edm::ParameterSet>>("tracksterCollections")),
      tracksters_token_(),
      tracksters_in_candidate_token_(
          consumes<std::vector<ticl::Trackster>>(ps.getParameter<edm::InputTag>("trackstersInCand"))),
      layer_clusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layerClusters"))),
      ticl_candidates_token_(consumes<std::vector<TICLCandidate>>(ps.getParameter<edm::InputTag>("ticlcandidates"))),
      ticl_candidates_tracksters_token_(
          consumes<std::vector<ticl::Trackster>>(ps.getParameter<edm::InputTag>("ticlcandidates"))),
      tracks_token_(consumes<std::vector<reco::Track>>(ps.getParameter<edm::InputTag>("tracks"))),
      tracks_time_token_(consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksTime"))),
      tracks_time_quality_token_(consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksTimeQual"))),
      tracks_time_err_token_(consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksTimeErr"))),
      tracks_beta_token_(consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksBeta"))),
      tracks_time_mtd_token_(consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksTimeMtd"))),
      tracks_time_mtd_err_token_(consumes<edm::ValueMap<float>>(ps.getParameter<edm::InputTag>("tracksTimeMtdErr"))),
      tracks_pos_mtd_token_(consumes<edm::ValueMap<GlobalPoint>>(ps.getParameter<edm::InputTag>("tracksPosMtd"))),
      muons_token_(consumes<std::vector<reco::Muon>>(ps.getParameter<edm::InputTag>("muons"))),
      clustersTime_token_(
          consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("layer_clustersTime"))),
      superclustering_linkedResultTracksters_token(
          consumes<std::vector<std::vector<unsigned int>>>(ps.getParameter<edm::InputTag>("superclustering"))),
      recoSuperClusters_token(
          consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("recoSuperClusters"))),
      recoSuperClusters_caloClusters_token(
          consumes<reco::CaloClusterCollection>(ps.getParameter<edm::InputTag>("recoSuperClusters"))),
      recoSuperClusters_sourceTracksters_token(consumes<std::vector<ticl::Trackster>>(
          ps.getParameter<edm::InputTag>("recoSuperClusters_sourceTracksterCollection"))),
      caloGeometry_token_(esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>()),
      simTracksters_SC_token_(
          consumes<std::vector<ticl::Trackster>>(ps.getParameter<edm::InputTag>("simtrackstersSC"))),
      simTICLCandidate_token_(
          consumes<std::vector<TICLCandidate>>(ps.getParameter<edm::InputTag>("simTICLCandidates"))),
      associations_parameterSets_(ps.getParameter<std::vector<edm::ParameterSet>>("associators")),
      // The DumperHelpers should not be moved after construction (needed by TTree branch pointers), so construct them all here
      associations_dumperHelpers_(associations_parameterSets_.size()),
      simclusters_token_(consumes(ps.getParameter<edm::InputTag>("simclusters"))),
      caloparticles_token_(consumes(ps.getParameter<edm::InputTag>("caloparticles"))),
      geometry_token_(esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>()),
      detector_(ps.getParameter<std::string>("detector")),
      propName_(ps.getParameter<std::string>("propagator")),
      bfield_token_(esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>()),
      propagator_token_(
          esConsumes<Propagator, TrackingComponentsRecord, edm::Transition::BeginRun>(edm::ESInputTag("", propName_))),
      saveLCs_(ps.getParameter<bool>("saveLCs")),
      saveSuperclustering_(ps.getParameter<bool>("saveSuperclustering")),
      //saveSuperclusteringDNNScore_(ps.getParameter<bool>("saveSuperclusteringDNNScore")),
      saveRecoSuperclusters_(ps.getParameter<bool>("saveRecoSuperclusters")),
      saveTICLCandidate_(ps.getParameter<bool>("saveSimTICLCandidate")),
      saveSimTICLCandidate_(ps.getParameter<bool>("saveSimTICLCandidate")),
      saveTracks_(ps.getParameter<bool>("saveTracks")) {
  if (saveSuperclustering_) {
    superclustering_linkedResultTracksters_token =
        consumes<std::vector<std::vector<unsigned int>>>(ps.getParameter<edm::InputTag>("superclustering"));
    recoSuperClusters_token =
        consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("recoSuperClusters"));
    recoSuperClusters_caloClusters_token =
        consumes<reco::CaloClusterCollection>(ps.getParameter<edm::InputTag>("recoSuperClusters"));
    recoSuperClusters_sourceTracksters_token = consumes<std::vector<ticl::Trackster>>(
        ps.getParameter<edm::InputTag>("recoSuperClusters_sourceTracksterCollection"));
  }
  std::string detectorName_ = (detector_ == "HFNose") ? "HGCalHFNoseSensitive" : "HGCalEESensitive";
  hdc_token_ =
      esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag("", detectorName_));

  for (edm::ParameterSet const& tracksterPset : tracksters_parameterSets_) {
    tracksters_token_.push_back(
        consumes<std::vector<ticl::Trackster>>(tracksterPset.getParameter<edm::InputTag>("inputTag")));
    tracksters_dumperHelpers_.emplace_back(
        TracksterDumperHelper::tracksterTypeFromString(tracksterPset.getParameter<std::string>("tracksterType")));
  }

  for (edm::ParameterSet const& associationPset : associations_parameterSets_) {
    associations_recoToSim_token_.push_back(consumes<TracksterToTracksterMap>(
        edm::InputTag(associationPset.getParameter<edm::InputTag>("associatorRecoToSimInputTag"))));
    associations_simToReco_token_.push_back(consumes<TracksterToTracksterMap>(
        edm::InputTag(associationPset.getParameter<edm::InputTag>("associatorSimToRecoInputTag"))));
  }
};

TICLDumper::~TICLDumper() { clearVariables(); };

void TICLDumper::beginRun(edm::Run const&, edm::EventSetup const& es) {
  detectorTools_ = std::make_unique<DetectorTools>(es.getData(hdc_token_),
                                                   es.getData(caloGeometry_token_),
                                                   es.getData(bfield_token_),
                                                   es.getData(propagator_token_));
}

// Define tree and branches
void TICLDumper::beginJob() {
  edm::Service<TFileService> fs;

  // Trackster trees
  for (unsigned int i = 0; i < tracksters_parameterSets_.size(); i++) {
    edm::ParameterSet const& tracksterPset = tracksters_parameterSets_[i];
    TTree* tree =
        fs->make<TTree>(tracksterPset.getParameter<std::string>("treeName").c_str(),
                        ("Tracksters : " + tracksterPset.getParameter<std::string>("treeName") +
                         " (InputTag : " + tracksterPset.getParameter<edm::InputTag>("inputTag").encode() + ")")
                            .c_str());
    tracksters_trees.push_back(tree);
    tracksters_dumperHelpers_[i].initTree(tree, &eventId_);
  }
  if (saveLCs_) {
    cluster_tree_ = fs->make<TTree>("clusters", "TICL tracksters");
    cluster_tree_->Branch("event", &eventId_);
    cluster_tree_->Branch("seedID", &cluster_seedID);
    cluster_tree_->Branch("energy", &cluster_energy);
    cluster_tree_->Branch("correctedEnergy", &cluster_correctedEnergy);
    cluster_tree_->Branch("correctedEnergyUncertainty", &cluster_correctedEnergyUncertainty);
    cluster_tree_->Branch("position_x", &cluster_position_x);
    cluster_tree_->Branch("position_y", &cluster_position_y);
    cluster_tree_->Branch("position_z", &cluster_position_z);
    cluster_tree_->Branch("position_eta", &cluster_position_eta);
    cluster_tree_->Branch("position_phi", &cluster_position_phi);
    cluster_tree_->Branch("cluster_layer_id", &cluster_layer_id);
    cluster_tree_->Branch("cluster_type", &cluster_type);
    cluster_tree_->Branch("cluster_time", &cluster_time);
    cluster_tree_->Branch("cluster_timeErr", &cluster_timeErr);
    cluster_tree_->Branch("cluster_number_of_hits", &cluster_number_of_hits);
  }
  if (saveTICLCandidate_) {
    candidate_tree_ = fs->make<TTree>("candidates", "TICL candidates");
    candidate_tree_->Branch("event", &eventId_);
    candidate_tree_->Branch("NCandidates", &nCandidates);
    candidate_tree_->Branch("candidate_charge", &candidate_charge);
    candidate_tree_->Branch("candidate_pdgId", &candidate_pdgId);
    candidate_tree_->Branch("candidate_id_probabilities", &candidate_id_probabilities);
    candidate_tree_->Branch("candidate_time", &candidate_time);
    candidate_tree_->Branch("candidate_timeErr", &candidate_time_err);
    candidate_tree_->Branch("candidate_energy", &candidate_energy);
    candidate_tree_->Branch("candidate_raw_energy", &candidate_raw_energy);
    candidate_tree_->Branch("candidate_px", &candidate_px);
    candidate_tree_->Branch("candidate_py", &candidate_py);
    candidate_tree_->Branch("candidate_pz", &candidate_pz);
    candidate_tree_->Branch("track_in_candidate", &track_in_candidate);
    candidate_tree_->Branch("tracksters_in_candidate", &tracksters_in_candidate);
  }
  if (saveSuperclustering_ || saveRecoSuperclusters_) {
    superclustering_tree_ = fs->make<TTree>("superclustering", "Superclustering in HGCAL CE-E");
    superclustering_tree_->Branch("event", &eventId_);
  }
  if (saveSuperclustering_) {
    superclustering_tree_->Branch("linkedResultTracksters", &superclustering_linkedResultTracksters);
  }
  if (saveRecoSuperclusters_) {
    superclustering_tree_->Branch("recoSuperCluster_rawEnergy", &recoSuperCluster_rawEnergy);
    superclustering_tree_->Branch("recoSuperCluster_energy", &recoSuperCluster_energy);
    superclustering_tree_->Branch("recoSuperCluster_correctedEnergy", &recoSuperCluster_correctedEnergy);
    superclustering_tree_->Branch("recoSuperCluster_position_x", &recoSuperCluster_position_x);
    superclustering_tree_->Branch("recoSuperCluster_position_y", &recoSuperCluster_position_y);
    superclustering_tree_->Branch("recoSuperCluster_position_z", &recoSuperCluster_position_z);
    superclustering_tree_->Branch("recoSuperCluster_position_eta", &recoSuperCluster_position_eta);
    superclustering_tree_->Branch("recoSuperCluster_position_phi", &recoSuperCluster_position_phi);
    superclustering_tree_->Branch("recoSuperCluster_seedTs", &recoSuperCluster_seedTs);
    superclustering_tree_->Branch("recoSuperCluster_constituentTs", &recoSuperCluster_constituentTs);
  }

  if (!associations_parameterSets_.empty()) {
    associations_tree_ = fs->make<TTree>("associations", "Associations");
    associations_tree_->Branch("event", &eventId_);
  }
  for (unsigned int i = 0; i < associations_parameterSets_.size(); i++) {
    associations_dumperHelpers_[i].initTree(associations_tree_,
                                            associations_parameterSets_[i].getParameter<std::string>("branchName"),
                                            associations_parameterSets_[i].getParameter<std::string>("suffix"));
  }

  if (saveTracks_) {
    tracks_tree_ = fs->make<TTree>("tracks", "Tracks");
    tracks_tree_->Branch("event", &eventId_);
    tracks_tree_->Branch("track_id", &track_id);
    tracks_tree_->Branch("track_hgcal_x", &track_hgcal_x);
    tracks_tree_->Branch("track_hgcal_y", &track_hgcal_y);
    tracks_tree_->Branch("track_hgcal_z", &track_hgcal_z);
    tracks_tree_->Branch("track_hgcal_eta", &track_hgcal_eta);
    tracks_tree_->Branch("track_hgcal_phi", &track_hgcal_phi);
    tracks_tree_->Branch("track_hgcal_pt", &track_hgcal_pt);
    tracks_tree_->Branch("track_pt", &track_pt);
    tracks_tree_->Branch("track_missing_outer_hits", &track_missing_outer_hits);
    tracks_tree_->Branch("track_missing_inner_hits", &track_missing_inner_hits);
    tracks_tree_->Branch("track_quality", &track_quality);
    tracks_tree_->Branch("track_charge", &track_charge);
    tracks_tree_->Branch("track_time", &track_time);
    tracks_tree_->Branch("track_time_quality", &track_time_quality);
    tracks_tree_->Branch("track_time_err", &track_time_err);
    tracks_tree_->Branch("track_beta", &track_beta);
    tracks_tree_->Branch("track_time_mtd", &track_time_mtd);
    tracks_tree_->Branch("track_time_mtd_err", &track_time_mtd_err);
    tracks_tree_->Branch("track_pos_mtd", &track_pos_mtd);
    tracks_tree_->Branch("track_nhits", &track_nhits);
    tracks_tree_->Branch("track_isMuon", &track_isMuon);
    tracks_tree_->Branch("track_isTrackerMuon", &track_isTrackerMuon);
  }

  if (saveSimTICLCandidate_) {
    simTICLCandidate_tree = fs->make<TTree>("simTICLCandidate", "Sim TICL Candidate");
    simTICLCandidate_tree->Branch("event", &eventId_);
    simTICLCandidate_tree->Branch("simTICLCandidate_raw_energy", &simTICLCandidate_raw_energy);
    simTICLCandidate_tree->Branch("simTICLCandidate_regressed_energy", &simTICLCandidate_regressed_energy);
    simTICLCandidate_tree->Branch("simTICLCandidate_simTracksterCPIndex", &simTICLCandidate_simTracksterCPIndex);
    simTICLCandidate_tree->Branch("simTICLCandidate_boundaryX", &simTICLCandidate_boundaryX);
    simTICLCandidate_tree->Branch("simTICLCandidate_boundaryY", &simTICLCandidate_boundaryY);
    simTICLCandidate_tree->Branch("simTICLCandidate_boundaryZ", &simTICLCandidate_boundaryZ);
    simTICLCandidate_tree->Branch("simTICLCandidate_boundaryPx", &simTICLCandidate_boundaryPx);
    simTICLCandidate_tree->Branch("simTICLCandidate_boundaryPy", &simTICLCandidate_boundaryPy);
    simTICLCandidate_tree->Branch("simTICLCandidate_boundaryPz", &simTICLCandidate_boundaryPz);
    simTICLCandidate_tree->Branch("simTICLCandidate_time", &simTICLCandidate_time);
    simTICLCandidate_tree->Branch("simTICLCandidate_caloParticleMass", &simTICLCandidate_caloParticleMass);
    simTICLCandidate_tree->Branch("simTICLCandidate_pdgId", &simTICLCandidate_pdgId);
    simTICLCandidate_tree->Branch("simTICLCandidate_charge", &simTICLCandidate_charge);
    simTICLCandidate_tree->Branch("simTICLCandidate_track_in_candidate", &simTICLCandidate_track_in_candidate);
  }
}

void TICLDumper::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  eventId_ = event.id();
  clearVariables();

  edm::Handle<std::vector<ticl::Trackster>> tracksters_in_candidate_handle;
  event.getByToken(tracksters_in_candidate_token_, tracksters_in_candidate_handle);

  //get all the layer clusters
  edm::Handle<std::vector<reco::CaloCluster>> layer_clusters_h;
  event.getByToken(layer_clusters_token_, layer_clusters_h);
  const auto& clusters = *layer_clusters_h;

  edm::Handle<edm::ValueMap<std::pair<float, float>>> clustersTime_h;
  event.getByToken(clustersTime_token_, clustersTime_h);
  const auto& layerClustersTimes = *clustersTime_h;

  //TICL Candidate
  edm::Handle<std::vector<TICLCandidate>> candidates_h;
  event.getByToken(ticl_candidates_token_, candidates_h);
  const auto& ticlcandidates = *candidates_h;
  edm::Handle<std::vector<ticl::Trackster>> ticlcandidates_tracksters_h =
      event.getHandle(ticl_candidates_tracksters_token_);

  //Track
  edm::Handle<std::vector<reco::Track>> tracks_h;
  event.getByToken(tracks_token_, tracks_h);
  const auto& tracks = *tracks_h;

  edm::Handle<edm::ValueMap<float>> trackTime_h;
  event.getByToken(tracks_time_token_, trackTime_h);
  const auto& trackTime = *trackTime_h;

  edm::Handle<edm::ValueMap<float>> trackTimeErr_h;
  event.getByToken(tracks_time_err_token_, trackTimeErr_h);
  const auto& trackTimeErr = *trackTimeErr_h;

  edm::Handle<edm::ValueMap<float>> trackBeta_h;
  event.getByToken(tracks_beta_token_, trackBeta_h);
  const auto& trackBeta = *trackBeta_h;

  edm::Handle<edm::ValueMap<float>> trackTimeQual_h;
  event.getByToken(tracks_time_quality_token_, trackTimeQual_h);
  const auto& trackTimeQual = *trackTimeQual_h;

  edm::Handle<edm::ValueMap<float>> trackTimeMtd_h;
  event.getByToken(tracks_time_mtd_token_, trackTimeMtd_h);
  const auto& trackTimeMtd = *trackTimeMtd_h;

  edm::Handle<edm::ValueMap<float>> trackTimeMtdErr_h;
  event.getByToken(tracks_time_mtd_err_token_, trackTimeMtdErr_h);
  const auto& trackTimeMtdErr = *trackTimeMtdErr_h;

  edm::Handle<edm::ValueMap<GlobalPoint>> trackPosMtd_h;
  event.getByToken(tracks_pos_mtd_token_, trackPosMtd_h);
  const auto& trackPosMtd = *trackPosMtd_h;

  // superclustering
  if (saveSuperclustering_)  // To support running with Mustache
    superclustering_linkedResultTracksters = event.get(superclustering_linkedResultTracksters_token);

  // muons
  edm::Handle<std::vector<reco::Muon>> muons_h;
  event.getByToken(muons_token_, muons_h);
  auto& muons = *muons_h;

  // recoSuperClusters
  if (saveRecoSuperclusters_) {
    reco::SuperClusterCollection const& recoSuperClusters = event.get(recoSuperClusters_token);
    // reco::CaloClusterCollection const& recoCaloClusters = event.get(recoSuperClusters_caloClusters_token);
    std::vector<ticl::Trackster> const& recoSuperClusters_sourceTracksters =
        event.get(recoSuperClusters_sourceTracksters_token);

    // Map for fast lookup of hit to trackster index in recoSuperClusters_sourceTracksters
    std::unordered_map<DetId, unsigned> hitToTracksterMap;

    for (unsigned ts_id = 0; ts_id < recoSuperClusters_sourceTracksters.size(); ts_id++) {
      for (unsigned int lc_index : recoSuperClusters_sourceTracksters[ts_id].vertices()) {
        for (auto [detId, fraction] : clusters[lc_index].hitsAndFractions()) {
          bool insertionSucceeded = hitToTracksterMap.emplace(detId, ts_id).second;
          assert(insertionSucceeded && "TICLDumper found tracksters sharing rechits");
        }
      }
    }

    for (auto const& recoSc : recoSuperClusters) {
      recoSuperCluster_rawEnergy.push_back(recoSc.rawEnergy());
      recoSuperCluster_energy.push_back(recoSc.energy());
      recoSuperCluster_correctedEnergy.push_back(recoSc.correctedEnergy());
      recoSuperCluster_position_x.push_back(recoSc.position().x());
      recoSuperCluster_position_y.push_back(recoSc.position().y());
      recoSuperCluster_position_z.push_back(recoSc.position().z());
      recoSuperCluster_position_eta.push_back(recoSc.position().eta());
      recoSuperCluster_position_phi.push_back(recoSc.position().phi());

      // Finding the trackster that was used to create the CaloCluster, using the DetId of a hit (we assume there is no sharing of rechits between tracksters)

      // Seed trackster of the supercluster : Using the DetId of the seed rechit of the seed CaloCluster
      recoSuperCluster_seedTs.push_back(hitToTracksterMap.at(recoSc.seed()->seed()));
      recoSuperCluster_constituentTs.emplace_back();
      for (edm::Ptr<reco::CaloCluster> const& caloClusterPtr : recoSc.clusters()) {
        // Using the DetId of the seed rechit of the CaloCluster
        recoSuperCluster_constituentTs.back().push_back(hitToTracksterMap.at(caloClusterPtr->seed()));
      }
    }
  }

  edm::Handle<std::vector<TICLCandidate>> simTICLCandidates_h;
  event.getByToken(simTICLCandidate_token_, simTICLCandidates_h);
  const auto& simTICLCandidates = *simTICLCandidates_h;

  edm::Handle<std::vector<CaloParticle>> caloparticles_h;
  event.getByToken(caloparticles_token_, caloparticles_h);

  auto simclusters_h = event.getHandle(simclusters_token_);

  nclusters_ = clusters.size();

  // Save all the trackster collections
  for (unsigned int i = 0; i < tracksters_dumperHelpers_.size(); i++) {
    edm::Handle<std::vector<ticl::Trackster>> tracksters_handle;
    std::vector<ticl::Trackster> const& tracksters = event.get<std::vector<ticl::Trackster>>(tracksters_token_[i]);
    tracksters_dumperHelpers_[i].fillFromEvent(
        tracksters, clusters, layerClustersTimes, *detectorTools_, simclusters_h, caloparticles_h, tracks);
    tracksters_trees[i]->Fill();
  }

  const auto& simTrackstersSC_h = event.getHandle(simTracksters_SC_token_);
  simTICLCandidate_track_in_candidate.resize(simTICLCandidates.size(), -1);
  for (size_t i = 0; i < simTICLCandidates.size(); ++i) {
    auto const& cand = simTICLCandidates[i];

    simTICLCandidate_raw_energy.push_back(cand.rawEnergy());
    simTICLCandidate_regressed_energy.push_back(cand.p4().energy());
    simTICLCandidate_pdgId.push_back(cand.pdgId());
    simTICLCandidate_charge.push_back(cand.charge());
    simTICLCandidate_time.push_back(cand.time());
    std::vector<int> tmpIdxVec;
    for (auto const& simTS : cand.tracksters()) {
      auto trackster_idx = simTS.get() - (edm::Ptr<ticl::Trackster>(simTrackstersSC_h, 0)).get();
      tmpIdxVec.push_back(trackster_idx);
    }
    simTICLCandidate_simTracksterCPIndex.push_back(tmpIdxVec);
    tmpIdxVec.clear();
    auto const& trackPtr = cand.trackPtr();
    if (!trackPtr.isNull()) {
      auto const& track = *trackPtr;
      int iSide = int(track.eta() > 0);
      int tk_idx = trackPtr.get() - (edm::Ptr<reco::Track>(tracks_h, 0)).get();
      simTICLCandidate_track_in_candidate[i] = tk_idx;

      const auto& fts = trajectoryStateTransform::outerFreeState((track), &detectorTools_->bfield);
      // to the HGCal front
      const auto& tsos = detectorTools_->propagator.propagate(fts, detectorTools_->firstDisk_[iSide]->surface());
      if (tsos.isValid()) {
        const auto& globalPos = tsos.globalPosition();
        const auto& globalMom = tsos.globalMomentum();
        simTICLCandidate_boundaryX.push_back(globalPos.x());
        simTICLCandidate_boundaryY.push_back(globalPos.y());
        simTICLCandidate_boundaryZ.push_back(globalPos.z());
        simTICLCandidate_boundaryPx.push_back(globalMom.x());
        simTICLCandidate_boundaryPy.push_back(globalMom.y());
        simTICLCandidate_boundaryPz.push_back(globalMom.z());
      } else {
        simTICLCandidate_boundaryX.push_back(-999);
        simTICLCandidate_boundaryY.push_back(-999);
        simTICLCandidate_boundaryZ.push_back(-999);
        simTICLCandidate_boundaryPx.push_back(-999);
        simTICLCandidate_boundaryPy.push_back(-999);
        simTICLCandidate_boundaryPz.push_back(-999);
      }
    } else {
      simTICLCandidate_boundaryX.push_back(-999);
      simTICLCandidate_boundaryY.push_back(-999);
      simTICLCandidate_boundaryZ.push_back(-999);
      simTICLCandidate_boundaryPx.push_back(-999);
      simTICLCandidate_boundaryPy.push_back(-999);
      simTICLCandidate_boundaryPz.push_back(-999);
    }
  }

  int c_id = 0;

  for (auto cluster_iterator = clusters.begin(); cluster_iterator != clusters.end(); ++cluster_iterator) {
    auto lc_seed = cluster_iterator->seed();
    cluster_seedID.push_back(lc_seed);
    cluster_energy.push_back(cluster_iterator->energy());
    cluster_correctedEnergy.push_back(cluster_iterator->correctedEnergy());
    cluster_correctedEnergyUncertainty.push_back(cluster_iterator->correctedEnergyUncertainty());
    cluster_position_x.push_back(cluster_iterator->x());
    cluster_position_y.push_back(cluster_iterator->y());
    cluster_position_z.push_back(cluster_iterator->z());
    cluster_position_eta.push_back(cluster_iterator->eta());
    cluster_position_phi.push_back(cluster_iterator->phi());
    auto haf = cluster_iterator->hitsAndFractions();
    auto layerId = detectorTools_->rhtools.getLayerWithOffset(haf[0].first);
    cluster_layer_id.push_back(layerId);
    uint32_t number_of_hits = cluster_iterator->hitsAndFractions().size();
    cluster_number_of_hits.push_back(number_of_hits);
    cluster_type.push_back(detectorTools_->rhtools.getCellType(lc_seed));
    cluster_timeErr.push_back(layerClustersTimes.get(c_id).second);
    cluster_time.push_back(layerClustersTimes.get(c_id).first);
    c_id += 1;
  }

  tracksters_in_candidate.resize(ticlcandidates.size());
  track_in_candidate.resize(ticlcandidates.size(), -1);
  nCandidates = ticlcandidates.size();
  for (int i = 0; i < static_cast<int>(ticlcandidates.size()); ++i) {
    const auto& candidate = ticlcandidates[i];
    candidate_charge.push_back(candidate.charge());
    candidate_pdgId.push_back(candidate.pdgId());
    candidate_energy.push_back(candidate.energy());
    candidate_raw_energy.push_back(candidate.rawEnergy());
    candidate_px.push_back(candidate.px());
    candidate_py.push_back(candidate.py());
    candidate_pz.push_back(candidate.pz());
    candidate_time.push_back(candidate.time());
    candidate_time_err.push_back(candidate.timeError());
    std::vector<float> id_probs;
    for (int j = 0; j < 8; j++) {
      ticl::Trackster::ParticleType type = static_cast<ticl::Trackster::ParticleType>(j);
      id_probs.push_back(candidate.id_probability(type));
    }
    candidate_id_probabilities.push_back(id_probs);

    auto trackster_ptrs = candidate.tracksters();
    auto track_ptr = candidate.trackPtr();
    for (const auto& ts_ptr : trackster_ptrs) {
      auto ts_idx = ts_ptr.get() - (edm::Ptr<ticl::Trackster>(tracksters_in_candidate_handle, 0)).get();
      tracksters_in_candidate[i].push_back(ts_idx);
    }
    if (track_ptr.isNull())
      continue;
    int tk_idx = track_ptr.get() - (edm::Ptr<reco::Track>(tracks_h, 0)).get();
    track_in_candidate[i] = tk_idx;
  }

  // trackster to simTrackster associations
  for (unsigned int i = 0; i < associations_dumperHelpers_.size(); i++) {
    associations_dumperHelpers_[i].fillFromEvent(event.get(associations_recoToSim_token_[i]),
                                                 event.get(associations_simToReco_token_[i]));
  }
  if (!associations_dumperHelpers_.empty())
    associations_tree_->Fill();

  //Tracks
  for (size_t i = 0; i < tracks.size(); i++) {
    const auto& track = tracks[i];
    reco::TrackRef trackref = reco::TrackRef(tracks_h, i);
    int iSide = int(track.eta() > 0);
    const auto& fts = trajectoryStateTransform::outerFreeState((track), &detectorTools_->bfield);
    // to the HGCal front
    const auto& tsos = detectorTools_->propagator.propagate(fts, detectorTools_->firstDisk_[iSide]->surface());
    if (tsos.isValid()) {
      const auto& globalPos = tsos.globalPosition();
      const auto& globalMom = tsos.globalMomentum();
      track_id.push_back(i);
      track_hgcal_x.push_back(globalPos.x());
      track_hgcal_y.push_back(globalPos.y());
      track_hgcal_z.push_back(globalPos.z());
      track_hgcal_eta.push_back(globalPos.eta());
      track_hgcal_phi.push_back(globalPos.phi());
      track_hgcal_px.push_back(globalMom.x());
      track_hgcal_py.push_back(globalMom.y());
      track_hgcal_pz.push_back(globalMom.z());
      track_hgcal_pt.push_back(globalMom.perp());
      track_pt.push_back(track.pt());
      track_quality.push_back(track.quality(reco::TrackBase::highPurity));
      track_missing_outer_hits.push_back(track.missingOuterHits());
      track_missing_inner_hits.push_back(track.missingInnerHits());
      track_charge.push_back(track.charge());
      track_time.push_back(trackTime[trackref]);
      track_time_quality.push_back(trackTimeQual[trackref]);
      track_time_err.push_back(trackTimeErr[trackref]);
      track_beta.push_back(trackBeta[trackref]);
      track_time_mtd.push_back(trackTimeMtd[trackref]);
      track_time_mtd_err.push_back(trackTimeMtdErr[trackref]);
      track_pos_mtd.push_back(trackPosMtd[trackref]);
      track_nhits.push_back(tracks[i].recHitsSize());
      int muId = PFMuonAlgo::muAssocToTrack(trackref, *muons_h);
      if (muId != -1) {
        const reco::MuonRef muonref = reco::MuonRef(muons_h, muId);
        track_isMuon.push_back(PFMuonAlgo::isMuon(muonref));
        track_isTrackerMuon.push_back(muons[muId].isTrackerMuon());
      } else {
        track_isMuon.push_back(-1);
        track_isTrackerMuon.push_back(-1);
      }
    }
  }

  if (saveLCs_)
    cluster_tree_->Fill();
  if (saveTICLCandidate_)
    candidate_tree_->Fill();
  if (saveSuperclustering_ || saveRecoSuperclusters_)
    superclustering_tree_->Fill();
  if (saveTracks_)
    tracks_tree_->Fill();
  if (saveSimTICLCandidate_)
    simTICLCandidate_tree->Fill();
}

void TICLDumper::endJob() {}

void TICLDumper::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // Settings for dumping trackster collections
  edm::ParameterSetDescription tracksterDescValidator;
  tracksterDescValidator.add<std::string>("treeName")
      ->setComment("Name of the output tree for the trackster collection");
  tracksterDescValidator.add<edm::InputTag>("inputTag")->setComment("Input tag for the trackster collection to write");
  tracksterDescValidator.ifValue(
      edm::ParameterDescription<std::string>(
          "tracksterType",
          "Trackster",
          true,
          edm::Comment("Type of trackster. Trackster=regular trackster (from RECO). SimTracksterCP=Simtrackster "
                       "from CaloParticle. SimTracksterSC=Simtrackster from SimCluster")),
      edm::allowedValues<std::string>("Trackster", "SimTracksterCP", "SimTracksterSC"));
  desc.addVPSet("tracksterCollections", tracksterDescValidator)->setComment("Trackster collections to dump");

  desc.add<edm::InputTag>("trackstersInCand", edm::InputTag("ticlTrackstersCLUE3DHigh"));

  desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("layer_clustersTime", edm::InputTag("hgcalMergeLayerClusters", "timeLayerCluster"));
  desc.add<edm::InputTag>("ticlcandidates", edm::InputTag("ticlTrackstersMerge"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("tracksTime", edm::InputTag("tofPID:t0"));
  desc.add<edm::InputTag>("tracksTimeQual", edm::InputTag("mtdTrackQualityMVA:mtdQualMVA"));
  desc.add<edm::InputTag>("tracksTimeErr", edm::InputTag("tofPID:sigmat0"));
  desc.add<edm::InputTag>("tracksBeta", edm::InputTag("trackExtenderWithMTD:generalTrackBeta"));
  desc.add<edm::InputTag>("tracksTimeMtd", edm::InputTag("trackExtenderWithMTD:generalTracktmtd"));
  desc.add<edm::InputTag>("tracksTimeMtdErr", edm::InputTag("trackExtenderWithMTD:generalTracksigmatmtd"));
  desc.add<edm::InputTag>("tracksPosMtd", edm::InputTag("trackExtenderWithMTD:generalTrackmtdpos"));
  desc.add<edm::InputTag>("muons", edm::InputTag("muons1stStep"));
  desc.add<edm::InputTag>("superclustering", edm::InputTag("ticlTracksterLinksSuperclusteringDNN"));
  desc.add<edm::InputTag>("recoSuperClusters", edm::InputTag("particleFlowSuperClusterHGCal"))
      ->setComment(
          "egamma supercluster collection (either from PFECALSuperClusterProducer for Mustache, or from "
          "TICL->Egamma converter in case of TICL DNN superclusters)");
  desc.add<edm::InputTag>("recoSuperClusters_sourceTracksterCollection", edm::InputTag("ticlTrackstersMerge"))
      ->setComment(
          "Trackster collection used to produce the reco::SuperCluster, used to provide a mapping back to the "
          "tracksters used in superclusters");

  desc.add<edm::InputTag>("simtrackstersSC", edm::InputTag("ticlSimTracksters"))
      ->setComment("SimTrackster from CaloParticle collection to use for simTICLcandidates");
  desc.add<edm::InputTag>("simTICLCandidates", edm::InputTag("ticlSimTracksters"));

  // Settings for dumping trackster associators (recoToSim & simToReco)
  edm::ParameterSetDescription associatorDescValidator;
  associatorDescValidator.add<std::string>("branchName")->setComment("Name of the output branches in the tree");
  associatorDescValidator.add<std::string>("suffix")->setComment("Should be CP or SC (for the output branch name)");
  associatorDescValidator.add<edm::InputTag>("associatorRecoToSimInputTag")
      ->setComment("Input tag for the RecoToSim associator to dump");
  associatorDescValidator.add<edm::InputTag>("associatorSimToRecoInputTag")
      ->setComment("Input tag for the SimToReco associator to dump");
  desc.addVPSet("associators", associatorDescValidator)->setComment("Tracksters to SimTracksters associators to dump");

  desc.add<edm::InputTag>("simclusters", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("caloparticles", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<std::string>("detector", "HGCAL");
  desc.add<std::string>("propagator", "PropagatorWithMaterial");

  desc.add<bool>("saveLCs", true);
  desc.add<bool>("saveTICLCandidate", true);
  desc.add<bool>("saveSimTICLCandidate", true);
  desc.add<bool>("saveTracks", true);
  desc.add<bool>("saveSuperclustering", true);
  desc.add<bool>("saveRecoSuperclusters", true)
      ->setComment("Save superclustering Egamma collections (as reco::SuperCluster)");
  descriptions.add("ticlDumper", desc);
}

DEFINE_FWK_MODULE(TICLDumper);
