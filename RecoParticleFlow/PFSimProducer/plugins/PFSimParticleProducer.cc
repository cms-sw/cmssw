/**\class PFSimParticleProducer 
\brief Producer for PFRecTracks and PFSimParticles

\todo Remove the PFRecTrack part, which is now handled by PFTracking
\author Colin Bernet
\date   April 2007
*/

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "DataFormats/GeometrySurface/interface/TkRotation.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFSimParticle.h"
#include "DataFormats/ParticleFlowReco/interface/PFSimParticleFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"
#include "RecoParticleFlow/PFProducer/interface/PFBlockAlgo.h"
#include "RecoParticleFlow/PFTracking/interface/PFGeometry.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include <memory>
#include <set>
#include <sstream>
#include <string>

class PFSimParticleProducer : public edm::stream::EDProducer<> {
public:
  explicit PFSimParticleProducer(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

  typedef edm::Handle<reco::PFRecTrackCollection> TrackHandle;
  void getSimIDs(const TrackHandle& trackh, std::vector<unsigned>& recTrackSimID);

private:
  /// module label for retrieving input simtrack and simvertex
  edm::InputTag inputTagSim_;
  edm::EDGetTokenT<std::vector<SimTrack> > tokenSim_;
  edm::EDGetTokenT<std::vector<SimVertex> > tokenSimVertices_;

  //MC Truth Matching
  //modif-beg
  bool mctruthMatchingInfo_;
  edm::InputTag inputTagFastSimProducer_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tokenFastSimProducer_;
  //modif-end

  edm::InputTag inputTagRecTracks_;
  edm::EDGetTokenT<reco::PFRecTrackCollection> tokenRecTracks_;
  edm::InputTag inputTagEcalRecHitsEB_;
  edm::EDGetTokenT<EcalRecHitCollection> tokenEcalRecHitsEB_;
  edm::InputTag inputTagEcalRecHitsEE_;
  edm::EDGetTokenT<EcalRecHitCollection> tokenEcalRecHitsEE_;

  // parameters for retrieving true particles information --

  edm::ParameterSet particleFilter_;
  std::unique_ptr<FSimEvent> mySimEvent;

  // flags for the various tasks ---------------------------

  /// process particles on/off
  bool processParticles_;

  /// verbose ?
  bool verbose_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFSimParticleProducer);

void PFSimParticleProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // particleFlowSimParticle
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("fastSimProducer", edm::InputTag("fastSimProducer", "EcalHitsEB"));
  desc.addUntracked<bool>("MCTruthMatchingInfo", false);
  desc.add<edm::InputTag>("RecTracks", edm::InputTag("trackerDrivenElectronSeeds"));
  desc.add<std::string>("Fitter", "KFFittingSmoother");
  desc.add<edm::InputTag>("ecalRecHitsEE", edm::InputTag("caloRecHits", "EcalRecHitsEE"));
  desc.add<edm::InputTag>("ecalRecHitsEB", edm::InputTag("caloRecHits", "EcalRecHitsEB"));
  desc.addUntracked<bool>("process_RecTracks", false);
  {
    edm::ParameterSetDescription psd0;
    psd0.setUnknown();
    desc.add<edm::ParameterSetDescription>("ParticleFilter", psd0);
  }
  desc.add<std::string>("TTRHBuilder", "WithTrackAngle");
  desc.addUntracked<bool>("process_Particles", true);
  desc.add<std::string>("Propagator", "PropagatorWithMaterial");
  desc.add<edm::InputTag>("sim", edm::InputTag("g4SimHits"));
  desc.addUntracked<bool>("verbose", false);
  descriptions.add("particleFlowSimParticle", desc);
}

using namespace std;
using namespace edm;

PFSimParticleProducer::PFSimParticleProducer(const edm::ParameterSet& iConfig) {
  processParticles_ = iConfig.getUntrackedParameter<bool>("process_Particles", true);

  inputTagSim_ = iConfig.getParameter<InputTag>("sim");
  tokenSim_ = consumes<std::vector<SimTrack> >(inputTagSim_);
  tokenSimVertices_ = consumes<std::vector<SimVertex> >(inputTagSim_);

  //retrieving collections for MC Truth Matching

  //modif-beg
  inputTagFastSimProducer_ = iConfig.getUntrackedParameter<InputTag>("fastSimProducer");
  tokenFastSimProducer_ = consumes<edm::PCaloHitContainer>(inputTagFastSimProducer_);
  mctruthMatchingInfo_ = iConfig.getUntrackedParameter<bool>("MCTruthMatchingInfo", false);
  //modif-end

  inputTagRecTracks_ = iConfig.getParameter<InputTag>("RecTracks");
  tokenRecTracks_ = consumes<reco::PFRecTrackCollection>(inputTagRecTracks_);

  inputTagEcalRecHitsEB_ = iConfig.getParameter<InputTag>("ecalRecHitsEB");
  tokenEcalRecHitsEB_ = consumes<EcalRecHitCollection>(inputTagEcalRecHitsEB_);
  inputTagEcalRecHitsEE_ = iConfig.getParameter<InputTag>("ecalRecHitsEE");
  tokenEcalRecHitsEE_ = consumes<EcalRecHitCollection>(inputTagEcalRecHitsEE_);

  verbose_ = iConfig.getUntrackedParameter<bool>("verbose", false);

  // register products
  produces<reco::PFSimParticleCollection>();

  particleFilter_ = iConfig.getParameter<ParameterSet>("ParticleFilter");

  mySimEvent = std::make_unique<FSimEvent>(particleFilter_);
}

void PFSimParticleProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  // init Particle data table (from Pythia)
  edm::ESHandle<HepPDT::ParticleDataTable> pdt;
  //  edm::ESHandle < DefaultConfig::ParticleDataTable > pdt;
  iSetup.getData(pdt);
  mySimEvent->initializePdt(&(*pdt));

  LogDebug("PFSimParticleProducer") << "START event: " << iEvent.id().event() << " in run " << iEvent.id().run()
                                    << endl;

  //MC Truth Matching only with Famos and UnFoldedMode option to true!!

  //vector to store the trackIDs of rectracks corresponding
  //to the simulated particle.
  std::vector<unsigned> recTrackSimID;

  //In order to know which simparticule contribute to
  //a given Ecal RecHit energy, we need to access
  //the PCAloHit from FastSim.

  typedef std::pair<double, unsigned> hitSimID;
  std::vector<std::list<hitSimID> > caloHitsEBID(62000);
  std::vector<double> caloHitsEBTotE(62000, 0.0);

  if (mctruthMatchingInfo_) {
    //getting the PCAloHit
    auto pcalohits = iEvent.getHandle(tokenFastSimProducer_);

    if (!pcalohits) {
      ostringstream err;
      err << "could not find pcaloHit "
          << "fastSimProducer:EcalHitsEB";
      LogError("PFSimParticleProducer") << err.str() << endl;

      throw cms::Exception("MissingProduct", err.str());
    } else {
      assert(pcalohits.isValid());

      edm::PCaloHitContainer::const_iterator it = pcalohits.product()->begin();
      edm::PCaloHitContainer::const_iterator itend = pcalohits.product()->end();

      //loop on the PCaloHit from FastSim Calorimetry
      for (; it != itend; ++it) {
        EBDetId detid(it->id());

        if (it->energy() > 0.0) {
          std::pair<double, unsigned> phitsimid = make_pair(it->energy(), it->geantTrackId());
          caloHitsEBID[detid.hashedIndex()].push_back(phitsimid);
          caloHitsEBTotE[detid.hashedIndex()] += it->energy();  //summing pcalhit energy
        }                                                       //energy > 0

      }  //loop PcaloHits
    }    //pcalohit handle access

    //Retrieving the PFRecTrack collection for
    //Monte Carlo Truth Matching tool
    Handle<reco::PFRecTrackCollection> recTracks;
    try {
      LogDebug("PFSimParticleProducer") << "getting PFRecTracks" << endl;
      iEvent.getByToken(tokenRecTracks_, recTracks);

    } catch (cms::Exception& err) {
      LogError("PFSimParticleProducer") << err << " cannot get collection "
                                        << "particleFlowBlock"
                                        << ":"
                                        << "" << endl;
    }  //pfrectrack handle access

    //getting the simID corresponding to
    //each of the PFRecTracks
    getSimIDs(recTracks, recTrackSimID);

  }  //mctruthMatchingInfo_ //modif

  // deal with true particles
  if (processParticles_) {
    auto pOutputPFSimParticleCollection = std::make_unique<reco::PFSimParticleCollection>();

    Handle<vector<SimTrack> > simTracks;
    bool found = iEvent.getByToken(tokenSim_, simTracks);
    if (!found) {
      ostringstream err;
      err << "cannot find sim tracks: " << inputTagSim_;
      LogError("PFSimParticleProducer") << err.str() << endl;

      throw cms::Exception("MissingProduct", err.str());
    }

    Handle<vector<SimVertex> > simVertices;
    found = iEvent.getByToken(tokenSimVertices_, simVertices);
    if (!found) {
      LogError("PFSimParticleProducer") << "cannot find sim vertices: " << inputTagSim_ << endl;
      return;
    }

    mySimEvent->fill(*simTracks, *simVertices);

    if (verbose_)
      mySimEvent->print();

    for (unsigned i = 0; i < mySimEvent->nTracks(); i++) {
      const FSimTrack& fst = mySimEvent->track(i);

      int motherId = -1;
      if (!fst.noMother())  // a mother exist
        motherId = fst.mother().id();

      //This is finding out the simID corresponding
      //to the recTrack

      //GETTING THE TRACK ID
      unsigned recTrackID = 99999;
      vector<unsigned> recHitContrib;    //modif
      vector<double> recHitContribFrac;  //modif

      if (mctruthMatchingInfo_) {  //modif

        for (unsigned lo = 0; lo < recTrackSimID.size(); lo++) {
          if (i == recTrackSimID[lo]) {
            recTrackID = lo;
          }  //match track
        }    //loop rectrack

        // get the ecalBarrel rechits for MC truth matching tool
        edm::Handle<EcalRecHitCollection> rhcHandle;
        bool found = iEvent.getByToken(tokenEcalRecHitsEB_, rhcHandle);
        if (!found) {
          ostringstream err;
          err << "could not find rechits " << inputTagEcalRecHitsEB_;
          LogError("PFSimParticleProducer") << err.str() << endl;

          throw cms::Exception("MissingProduct", err.str());
        } else {
          assert(rhcHandle.isValid());

          EBRecHitCollection::const_iterator it_rh = rhcHandle.product()->begin();
          EBRecHitCollection::const_iterator itend_rh = rhcHandle.product()->end();

          for (; it_rh != itend_rh; ++it_rh) {
            unsigned rhit_hi = EBDetId(it_rh->id()).hashedIndex();
            EBDetId detid(it_rh->id());

            auto it_phit = caloHitsEBID[rhit_hi].begin();
            auto itend_phit = caloHitsEBID[rhit_hi].end();
            for (; it_phit != itend_phit; ++it_phit) {
              if (i == it_phit->second) {
                //Alex (08/10/08) TO BE REMOVED, eliminating
                //duplicated rechits
                bool alreadyin = false;
                for (unsigned ihit = 0; ihit < recHitContrib.size(); ++ihit)
                  if (detid.rawId() == recHitContrib[ihit])
                    alreadyin = true;

                if (!alreadyin) {
                  double pcalofraction = 0.0;
                  if (caloHitsEBTotE[rhit_hi] != 0.0)
                    pcalofraction = (it_phit->first / caloHitsEBTotE[rhit_hi]) * 100.0;

                  //store info
                  recHitContrib.push_back(it_rh->id());
                  recHitContribFrac.push_back(pcalofraction);
                }  //selected rechits
              }    //matching
            }      //loop pcalohit

          }  //loop rechits

        }  //getting the rechits

      }  //mctruthMatchingInfo_ //modif

      reco::PFSimParticle particle(
          fst.charge(), fst.type(), fst.id(), motherId, fst.daughters(), recTrackID, recHitContrib, recHitContribFrac);

      const FSimVertex& originVtx = fst.vertex();

      math::XYZPoint posOrig(originVtx.position().x(), originVtx.position().y(), originVtx.position().z());

      math::XYZTLorentzVector momOrig(
          fst.momentum().px(), fst.momentum().py(), fst.momentum().pz(), fst.momentum().e());
      reco::PFTrajectoryPoint pointOrig(-1, reco::PFTrajectoryPoint::ClosestApproach, posOrig, momOrig);

      // point 0 is origin vertex
      particle.addPoint(pointOrig);

      if (!fst.noEndVertex()) {
        const FSimVertex& endVtx = fst.endVertex();

        math::XYZPoint posEnd(endVtx.position().x(), endVtx.position().y(), endVtx.position().z());

        math::XYZTLorentzVector momEnd;

        reco::PFTrajectoryPoint pointEnd(-1, reco::PFTrajectoryPoint::BeamPipeOrEndVertex, posEnd, momEnd);

        particle.addPoint(pointEnd);
      } else {  // add a dummy point
        reco::PFTrajectoryPoint dummy;
        particle.addPoint(dummy);
      }

      if (fst.onLayer1()) {  // PS layer1
        const RawParticle& rp = fst.layer1Entrance();

        math::XYZPoint posLayer1(rp.x(), rp.y(), rp.z());
        math::XYZTLorentzVector momLayer1(rp.px(), rp.py(), rp.pz(), rp.e());
        reco::PFTrajectoryPoint layer1Pt(-1, reco::PFTrajectoryPoint::PS1, posLayer1, momLayer1);

        particle.addPoint(layer1Pt);

        // extrapolate to cluster depth
      } else {  // add a dummy point
        reco::PFTrajectoryPoint dummy;
        particle.addPoint(dummy);
      }

      if (fst.onLayer2()) {  // PS layer2
        const RawParticle& rp = fst.layer2Entrance();

        math::XYZPoint posLayer2(rp.x(), rp.y(), rp.z());
        math::XYZTLorentzVector momLayer2(rp.px(), rp.py(), rp.pz(), rp.e());
        reco::PFTrajectoryPoint layer2Pt(-1, reco::PFTrajectoryPoint::PS2, posLayer2, momLayer2);

        particle.addPoint(layer2Pt);

        // extrapolate to cluster depth
      } else {  // add a dummy point
        reco::PFTrajectoryPoint dummy;
        particle.addPoint(dummy);
      }

      if (fst.onEcal()) {
        const RawParticle& rp = fst.ecalEntrance();

        math::XYZPoint posECAL(rp.x(), rp.y(), rp.z());
        math::XYZTLorentzVector momECAL(rp.px(), rp.py(), rp.pz(), rp.e());
        reco::PFTrajectoryPoint ecalPt(-1, reco::PFTrajectoryPoint::ECALEntrance, posECAL, momECAL);

        particle.addPoint(ecalPt);

        // extrapolate to cluster depth
      } else {  // add a dummy point
        reco::PFTrajectoryPoint dummy;
        particle.addPoint(dummy);
      }

      // add a dummy point for ECAL Shower max
      reco::PFTrajectoryPoint dummy;
      particle.addPoint(dummy);

      if (fst.onHcal()) {
        const RawParticle& rpin = fst.hcalEntrance();

        math::XYZPoint posHCALin(rpin.x(), rpin.y(), rpin.z());
        math::XYZTLorentzVector momHCALin(rpin.px(), rpin.py(), rpin.pz(), rpin.e());
        reco::PFTrajectoryPoint hcalPtin(-1, reco::PFTrajectoryPoint::HCALEntrance, posHCALin, momHCALin);

        particle.addPoint(hcalPtin);

        const RawParticle& rpout = fst.hcalExit();

        math::XYZPoint posHCALout(rpout.x(), rpout.y(), rpout.z());
        math::XYZTLorentzVector momHCALout(rpout.px(), rpout.py(), rpout.pz(), rpout.e());
        reco::PFTrajectoryPoint hcalPtout(-1, reco::PFTrajectoryPoint::HCALExit, posHCALout, momHCALout);

        particle.addPoint(hcalPtout);

        const RawParticle& rpho = fst.hoEntrance();

        math::XYZPoint posHOEntrance(rpho.x(), rpho.y(), rpho.z());
        math::XYZTLorentzVector momHOEntrance(rpho.px(), rpho.py(), rpho.pz(), rpho.e());
        reco::PFTrajectoryPoint hoPtin(-1, reco::PFTrajectoryPoint::HOLayer, posHOEntrance, momHOEntrance);

        particle.addPoint(hoPtin);

      } else {  // add a dummy point
        reco::PFTrajectoryPoint dummy;
        particle.addPoint(dummy);
      }

      pOutputPFSimParticleCollection->push_back(particle);
    }

    iEvent.put(std::move(pOutputPFSimParticleCollection));
  }

  LogDebug("PFSimParticleProducer") << "STOP event: " << iEvent.id().event() << " in run " << iEvent.id().run() << endl;
}

void PFSimParticleProducer::getSimIDs(const TrackHandle& trackh, std::vector<unsigned>& recTrackSimID) {
  if (trackh.isValid()) {
    for (unsigned i = 0; i < trackh->size(); i++) {
      const reco::PFRecTrackRef ref(trackh, i);

      for (auto const& hit : ref->trackRef()->recHits()) {
        if (hit->isValid()) {
          auto rechit = dynamic_cast<const FastTrackerRecHit*>(hit);

          for (unsigned int st_index = 0; st_index < rechit->nSimTrackIds(); ++st_index) {
            recTrackSimID.push_back(rechit->simTrackId(st_index));
          }
          break;
        }
      }  //loop track rechit
    }    //loop recTracks
  }      //track handle valid
}
