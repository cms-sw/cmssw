// -*- C++ -*-
//
// Package:    Alignment/OfflineValidation
// Class:      PrimaryVertexValidation
//
/**\class PrimaryVertexValidation PrimaryVertexValidation.cc Alignment/OfflineValidation/plugins/PrimaryVertexValidation.cc

 Description: Validate alignment constants using unbiased vertex residuals

 Implementation:
 <Notes on implementation>
*/
//
// Original Author:  Marco Musich
//         Created:  Tue Mar 02 10:39:34 CDT 2010
//

// system include files
#include <memory>
#include <vector>
#include <regex>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <boost/range/adaptor/indexed.hpp>

// user include files
#include "Alignment/OfflineValidation/plugins/PrimaryVertexValidation.h"

// ROOT includes
#include "TH1F.h"
#include "TH2F.h"
#include "TF1.h"
#include "TVector3.h"
#include "TFile.h"
#include "TMath.h"
#include "TROOT.h"
#include "TChain.h"
#include "TNtuple.h"
#include "TMatrixD.h"
#include "TVectorD.h"

// CMSSW includes
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "RecoVertex/PrimaryVertexProducer/interface/DAClusterizerInZ_vect.h"
#include "RecoVertex/PrimaryVertexProducer/interface/DAClusterizerInZ.h"
#include "RecoVertex/PrimaryVertexProducer/interface/GapClusterizerInZ.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"

const int PrimaryVertexValidation::nMaxtracks_;

// Constructor
PrimaryVertexValidation::PrimaryVertexValidation(const edm::ParameterSet& iConfig)
    : magFieldToken_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
      trackingGeomToken_(esConsumes<GlobalTrackingGeometry, GlobalTrackingGeometryRecord>()),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      ttkToken_(esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"))),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>()),
      runInfoToken_(esConsumes<RunInfo, RunInfoRcd>()),
      compressionSettings_(iConfig.getUntrackedParameter<int>("compressionSettings", -1)),
      storeNtuple_(iConfig.getParameter<bool>("storeNtuple")),
      lightNtupleSwitch_(iConfig.getParameter<bool>("isLightNtuple")),
      useTracksFromRecoVtx_(iConfig.getParameter<bool>("useTracksFromRecoVtx")),
      vertexZMax_(iConfig.getUntrackedParameter<double>("vertexZMax", 99.)),
      intLumi_(iConfig.getUntrackedParameter<double>("intLumi", 0.)),
      askFirstLayerHit_(iConfig.getParameter<bool>("askFirstLayerHit")),
      doBPix_(iConfig.getUntrackedParameter<bool>("doBPix", true)),
      doFPix_(iConfig.getUntrackedParameter<bool>("doFPix", true)),
      ptOfProbe_(iConfig.getUntrackedParameter<double>("probePt", 0.)),
      pOfProbe_(iConfig.getUntrackedParameter<double>("probeP", 0.)),
      etaOfProbe_(iConfig.getUntrackedParameter<double>("probeEta", 2.4)),
      nHitsOfProbe_(iConfig.getUntrackedParameter<double>("probeNHits", 0.)),
      nBins_(iConfig.getUntrackedParameter<int>("numberOfBins", 24)),
      minPt_(iConfig.getUntrackedParameter<double>("minPt", 1.)),
      maxPt_(iConfig.getUntrackedParameter<double>("maxPt", 20.)),
      debug_(iConfig.getParameter<bool>("Debug")),
      runControl_(iConfig.getUntrackedParameter<bool>("runControl", false)),
      forceBeamSpotContraint_(iConfig.getUntrackedParameter<bool>("forceBeamSpot", false)) {
  // now do what ever initialization is needed
  // initialize phase space boundaries

  usesResource(TFileService::kSharedResource);

  std::vector<unsigned int> defaultRuns;
  defaultRuns.push_back(0);
  runControlNumbers_ = iConfig.getUntrackedParameter<std::vector<unsigned int>>("runControlNumber", defaultRuns);

  edm::InputTag TrackCollectionTag_ = iConfig.getParameter<edm::InputTag>("TrackCollectionTag");
  theTrackCollectionToken = consumes<reco::TrackCollection>(TrackCollectionTag_);

  edm::InputTag VertexCollectionTag_ = iConfig.getParameter<edm::InputTag>("VertexCollectionTag");
  theVertexCollectionToken = consumes<reco::VertexCollection>(VertexCollectionTag_);

  edm::InputTag BeamspotTag_ = iConfig.getParameter<edm::InputTag>("BeamSpotTag");
  theBeamspotToken = consumes<reco::BeamSpot>(BeamspotTag_);

  // select and configure the track filter
  theTrackFilter_ =
      std::make_unique<TrackFilterForPVFinding>(iConfig.getParameter<edm::ParameterSet>("TkFilterParameters"));
  // select and configure the track clusterizer
  std::string clusteringAlgorithm =
      iConfig.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<std::string>("algorithm");
  if (clusteringAlgorithm == "gap") {
    theTrackClusterizer_ =
        std::make_unique<GapClusterizerInZ>(iConfig.getParameter<edm::ParameterSet>("TkClusParameters")
                                                .getParameter<edm::ParameterSet>("TkGapClusParameters"));
  } else if (clusteringAlgorithm == "DA") {
    theTrackClusterizer_ =
        std::make_unique<DAClusterizerInZ>(iConfig.getParameter<edm::ParameterSet>("TkClusParameters")
                                               .getParameter<edm::ParameterSet>("TkDAClusParameters"));
    // provide the vectorized version of the clusterizer, if supported by the build
  } else if (clusteringAlgorithm == "DA_vect") {
    theTrackClusterizer_ =
        std::make_unique<DAClusterizerInZ_vect>(iConfig.getParameter<edm::ParameterSet>("TkClusParameters")
                                                    .getParameter<edm::ParameterSet>("TkDAClusParameters"));
  } else {
    throw VertexException("PrimaryVertexProducerAlgorithm: unknown clustering algorithm: " + clusteringAlgorithm);
  }

  theDetails_.histobins = 500;
  theDetails_.setMap(PVValHelper::dxy, PVValHelper::phi, -2000., 2000.);
  theDetails_.setMap(PVValHelper::dxy, PVValHelper::eta, -3000., 3000.);
  theDetails_.setMap(PVValHelper::dxy, PVValHelper::pT, -1000., 1000.);
  theDetails_.setMap(PVValHelper::dxy, PVValHelper::pTCentral, -1000., 1000.);
  theDetails_.setMap(PVValHelper::dxy, PVValHelper::ladder, -1000., 1000.);
  theDetails_.setMap(PVValHelper::dxy, PVValHelper::modZ, -1000., 1000.);

  for (int i = PVValHelper::phi; i < PVValHelper::END_OF_PLOTS; i++) {
    for (int j = PVValHelper::dx; j < PVValHelper::END_OF_TYPES; j++) {
      auto plot_index = static_cast<PVValHelper::plotVariable>(i);
      auto res_index = static_cast<PVValHelper::residualType>(j);

      if (debug_) {
        edm::LogInfo("PrimaryVertexValidation")
            << "==> " << std::get<0>(PVValHelper::getTypeString(res_index)) << " " << std::setw(10)
            << std::get<0>(PVValHelper::getVarString(plot_index)) << std::endl;
      }
      if (res_index != PVValHelper::d3D && res_index != PVValHelper::norm_d3D)
        theDetails_.setMap(res_index,
                           plot_index,
                           theDetails_.getLow(PVValHelper::dxy, plot_index),
                           theDetails_.getHigh(PVValHelper::dxy, plot_index));
      else
        theDetails_.setMap(res_index, plot_index, 0., theDetails_.getHigh(PVValHelper::dxy, plot_index));
    }
  }

  edm::LogVerbatim("PrimaryVertexValidation") << "######################################";
  for (const auto& it : theDetails_.range) {
    edm::LogVerbatim("PrimaryVertexValidation")
        << "|" << std::setw(10) << std::get<0>(PVValHelper::getTypeString(it.first.first)) << "|" << std::setw(10)
        << std::get<0>(PVValHelper::getVarString(it.first.second)) << "| (" << std::setw(5) << it.second.first << ";"
        << std::setw(5) << it.second.second << ") |" << std::endl;
  }

  theDetails_.trendbins[PVValHelper::phi] = PVValHelper::generateBins(nBins_ + 1, -180., 360.);
  theDetails_.trendbins[PVValHelper::eta] = PVValHelper::generateBins(nBins_ + 1, -etaOfProbe_, 2 * etaOfProbe_);

  if (debug_) {
    edm::LogVerbatim("PrimaryVertexValidation") << "etaBins: ";
    for (auto ieta : theDetails_.trendbins[PVValHelper::eta]) {
      edm::LogVerbatim("PrimaryVertexValidation") << ieta << " ";
    }
    edm::LogVerbatim("PrimaryVertexValidation") << "\n";

    edm::LogVerbatim("PrimaryVertexValidation") << "phiBins: ";
    for (auto iphi : theDetails_.trendbins[PVValHelper::phi]) {
      edm::LogVerbatim("PrimaryVertexValidation") << iphi << " ";
    }
    edm::LogVerbatim("PrimaryVertexValidation") << "\n";
  }

  // create the bins of the pT-binned distributions

  mypT_bins_ = PVValHelper::makeLogBins<float, nPtBins_>(minPt_, maxPt_);

  std::string toOutput = "";
  for (auto ptbin : mypT_bins_) {
    toOutput += " ";
    toOutput += std::to_string(ptbin);
    toOutput += ",";
  }

  edm::LogVerbatim("PrimaryVertexValidation") << "######################################\n";
  edm::LogVerbatim("PrimaryVertexValidation") << "The pT binning is: [" << toOutput << "] \n";
}

// Destructor
PrimaryVertexValidation::~PrimaryVertexValidation() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void PrimaryVertexValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace std;
  using namespace reco;
  using namespace IPTools;

  if (!isBFieldConsistentWithMode(iSetup)) {
    edm::LogWarning("PrimaryVertexValidation")
        << "*********************************************************************************\n"
        << "* The configuration (ptOfProbe > " << ptOfProbe_
        << "GeV) is not correctly set for current value of magnetic field \n"
        << "* Switching it to 0. !!! \n"
        << "*********************************************************************************" << std::endl;
    ptOfProbe_ = 0.;
  }

  if (nBins_ != 24 && debug_) {
    edm::LogInfo("PrimaryVertexValidation") << "Using: " << nBins_ << " bins plots";
  }

  bool passesRunControl = false;

  if (runControl_) {
    for (const auto& runControlNumber : runControlNumbers_) {
      if (iEvent.eventAuxiliary().run() == runControlNumber) {
        if (debug_) {
          edm::LogInfo("PrimaryVertexValidation")
              << " run number: " << iEvent.eventAuxiliary().run() << " keeping run:" << runControlNumber;
        }
        passesRunControl = true;
        break;
      }
    }
    if (!passesRunControl)
      return;
  }

  Nevt_++;

  //=======================================================
  // Initialize Root-tuple variables
  //=======================================================

  SetVarToZero();

  //=======================================================
  // Retrieve the Magnetic Field information
  //=======================================================

  edm::ESHandle<MagneticField> theMGField = iSetup.getHandle(magFieldToken_);

  //=======================================================
  // Retrieve the Tracking Geometry information
  //=======================================================

  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry = iSetup.getHandle(trackingGeomToken_);

  //=======================================================
  // Retrieve geometry information
  //=======================================================

  edm::LogInfo("read tracker geometry...");
  edm::ESHandle<TrackerGeometry> pDD = iSetup.getHandle(geomToken_);
  edm::LogInfo("tracker geometry read") << "There are: " << pDD->dets().size() << " detectors";

  // switch on the phase2
  if ((pDD->isThere(GeomDetEnumerators::P2PXB)) || (pDD->isThere(GeomDetEnumerators::P2PXEC))) {
    phase_ = PVValHelper::phase2;
    nLadders_ = 12;
    nModZ_ = 9;

    if (h_dxy_ladderOverlap_.size() != nLadders_) {
      PVValHelper::shrinkHistVectorToFit(h_dxy_ladderOverlap_, nLadders_);
      PVValHelper::shrinkHistVectorToFit(h_dxy_ladderNoOverlap_, nLadders_);
      PVValHelper::shrinkHistVectorToFit(h_dxy_ladder_, nLadders_);
      PVValHelper::shrinkHistVectorToFit(h_dz_ladder_, nLadders_);
      PVValHelper::shrinkHistVectorToFit(h_norm_dxy_ladder_, nLadders_);
      PVValHelper::shrinkHistVectorToFit(h_norm_dz_ladder_, nLadders_);

      if (debug_) {
        edm::LogInfo("PrimaryVertexValidation") << "checking size:" << h_dxy_ladder_.size() << std::endl;
      }
    }

    if (debug_) {
      edm::LogInfo("PrimaryVertexValidation")
          << " pixel phase2 setup, nLadders: " << nLadders_ << " nModules:" << nModZ_;
    }

  } else if ((pDD->isThere(GeomDetEnumerators::P1PXB)) || (pDD->isThere(GeomDetEnumerators::P1PXEC))) {
    // switch on the phase1
    phase_ = PVValHelper::phase1;
    nLadders_ = 12;
    nModZ_ = 8;

    if (h_dxy_ladderOverlap_.size() != nLadders_) {
      PVValHelper::shrinkHistVectorToFit(h_dxy_ladderOverlap_, nLadders_);
      PVValHelper::shrinkHistVectorToFit(h_dxy_ladderNoOverlap_, nLadders_);
      PVValHelper::shrinkHistVectorToFit(h_dxy_ladder_, nLadders_);
      PVValHelper::shrinkHistVectorToFit(h_dz_ladder_, nLadders_);
      PVValHelper::shrinkHistVectorToFit(h_norm_dxy_ladder_, nLadders_);
      PVValHelper::shrinkHistVectorToFit(h_norm_dz_ladder_, nLadders_);

      if (debug_) {
        edm::LogInfo("PrimaryVertexValidation") << "checking size:" << h_dxy_ladder_.size() << std::endl;
      }
    }

    if (h_dxy_modZ_.size() != nModZ_) {
      PVValHelper::shrinkHistVectorToFit(h_dxy_modZ_, nModZ_);
      PVValHelper::shrinkHistVectorToFit(h_dz_modZ_, nModZ_);
      PVValHelper::shrinkHistVectorToFit(h_norm_dxy_modZ_, nModZ_);
      PVValHelper::shrinkHistVectorToFit(h_norm_dxy_modZ_, nModZ_);

      if (debug_) {
        edm::LogInfo("PrimaryVertexValidation") << "checking size:" << h_dxy_modZ_.size() << std::endl;
      }
    }

    if (debug_) {
      edm::LogInfo("PrimaryVertexValidation")
          << " pixel phase1 setup, nLadders: " << nLadders_ << " nModules:" << nModZ_;
    }

  } else {
    phase_ = PVValHelper::phase0;
    nLadders_ = 20;
    nModZ_ = 8;

    if (h_dxy_modZ_.size() != nModZ_) {
      PVValHelper::shrinkHistVectorToFit(h_dxy_modZ_, nModZ_);
      PVValHelper::shrinkHistVectorToFit(h_dz_modZ_, nModZ_);
      PVValHelper::shrinkHistVectorToFit(h_norm_dxy_modZ_, nModZ_);
      PVValHelper::shrinkHistVectorToFit(h_norm_dxy_modZ_, nModZ_);

      if (debug_) {
        edm::LogInfo("PrimaryVertexValidation") << "checking size:" << h_dxy_modZ_.size() << std::endl;
      }
    }

    if (debug_) {
      edm::LogInfo("PrimaryVertexValidation")
          << " pixel phase0 setup, nLadders: " << nLadders_ << " nModules:" << nModZ_;
    }
  }

  switch (phase_) {
    case PVValHelper::phase0:
      etaOfProbe_ = std::min(etaOfProbe_, PVValHelper::max_eta_phase0);
      break;
    case PVValHelper::phase1:
      etaOfProbe_ = std::min(etaOfProbe_, PVValHelper::max_eta_phase1);
      break;
    case PVValHelper::phase2:
      etaOfProbe_ = std::min(etaOfProbe_, PVValHelper::max_eta_phase2);
      break;
    default:
      edm::LogWarning("LogicError") << "Unknown detector phase: " << phase_;
  }

  if (h_etaMax->GetEntries() == 0.) {
    h_etaMax->SetBinContent(1., etaOfProbe_);
    h_nbins->SetBinContent(1., nBins_);
    h_nLadders->SetBinContent(1., nLadders_);
    h_nModZ->SetBinContent(1., nModZ_);
    h_pTinfo->SetBinContent(1., mypT_bins_.size());
    h_pTinfo->SetBinContent(2., minPt_);
    h_pTinfo->SetBinContent(3., maxPt_);
  }

  //=======================================================
  // Retrieve the Transient Track Builder information
  //=======================================================

  edm::ESHandle<TransientTrackBuilder> theB_ = iSetup.getHandle(ttkToken_);
  double fBfield_ = ((*theB_).field()->inTesla(GlobalPoint(0., 0., 0.))).z();

  //=======================================================
  // Retrieve the Track information
  //=======================================================

  edm::Handle<TrackCollection> trackCollectionHandle;
  iEvent.getByToken(theTrackCollectionToken, trackCollectionHandle);
  if (!trackCollectionHandle.isValid())
    return;
  auto const& tracks = *trackCollectionHandle;

  //=======================================================
  // Retrieve tracker topology from geometry
  //=======================================================

  edm::ESHandle<TrackerTopology> tTopoHandle = iSetup.getHandle(topoToken_);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  //=======================================================
  // Retrieve offline vartex information (only for reco)
  //=======================================================

  //edm::Handle<VertexCollection> vertices;
  edm::Handle<std::vector<Vertex>> vertices;

  try {
    iEvent.getByToken(theVertexCollectionToken, vertices);
  } catch (cms::Exception& er) {
    LogTrace("PrimaryVertexValidation") << "caught std::exception " << er.what() << std::endl;
  }

  std::vector<Vertex> vsorted = *(vertices);
  // sort the vertices by number of tracks in descending order
  // use chi2 as tiebreaker
  std::sort(vsorted.begin(), vsorted.end(), PrimaryVertexValidation::vtxSort);

  // skip events with no PV, this should not happen

  if (vsorted.empty())
    return;

  // skip events failing vertex cut
  if (std::abs(vsorted[0].z()) > vertexZMax_)
    return;

  if (vsorted[0].isValid()) {
    xOfflineVertex_ = (vsorted)[0].x();
    yOfflineVertex_ = (vsorted)[0].y();
    zOfflineVertex_ = (vsorted)[0].z();

    xErrOfflineVertex_ = (vsorted)[0].xError();
    yErrOfflineVertex_ = (vsorted)[0].yError();
    zErrOfflineVertex_ = (vsorted)[0].zError();
  }

  h_xOfflineVertex->Fill(xOfflineVertex_);
  h_yOfflineVertex->Fill(yOfflineVertex_);
  h_zOfflineVertex->Fill(zOfflineVertex_);
  h_xErrOfflineVertex->Fill(xErrOfflineVertex_);
  h_yErrOfflineVertex->Fill(yErrOfflineVertex_);
  h_zErrOfflineVertex->Fill(zErrOfflineVertex_);

  unsigned int vertexCollectionSize = vsorted.size();
  int nvvertex = 0;

  for (unsigned int i = 0; i < vertexCollectionSize; i++) {
    const Vertex& vertex = vsorted.at(i);
    if (vertex.isValid())
      nvvertex++;
  }

  nOfflineVertices_ = nvvertex;
  h_nOfflineVertices->Fill(nvvertex);

  if (!vsorted.empty() && useTracksFromRecoVtx_) {
    double sumpt = 0;
    size_t ntracks = 0;
    double chi2ndf = 0.;
    double chi2prob = 0.;

    if (!vsorted.at(0).isFake()) {
      Vertex pv = vsorted.at(0);

      ntracks = pv.tracksSize();
      chi2ndf = pv.normalizedChi2();
      chi2prob = TMath::Prob(pv.chi2(), (int)pv.ndof());

      h_recoVtxNtracks_->Fill(ntracks);
      h_recoVtxChi2ndf_->Fill(chi2ndf);
      h_recoVtxChi2Prob_->Fill(chi2prob);

      for (Vertex::trackRef_iterator itrk = pv.tracks_begin(); itrk != pv.tracks_end(); ++itrk) {
        double pt = (**itrk).pt();
        sumpt += pt * pt;

        const math::XYZPoint myVertex(pv.position().x(), pv.position().y(), pv.position().z());

        double dxyRes = (**itrk).dxy(myVertex);
        double dzRes = (**itrk).dz(myVertex);

        double dxy_err = (**itrk).dxyError();
        double dz_err = (**itrk).dzError();

        float trackphi = ((**itrk).phi()) * (180 / M_PI);
        float tracketa = (**itrk).eta();

        for (int i = 0; i < nBins_; i++) {
          float phiF = theDetails_.trendbins[PVValHelper::phi][i];
          float phiL = theDetails_.trendbins[PVValHelper::phi][i + 1];

          float etaF = theDetails_.trendbins[PVValHelper::eta][i];
          float etaL = theDetails_.trendbins[PVValHelper::eta][i + 1];

          if (tracketa >= etaF && tracketa < etaL) {
            PVValHelper::fillByIndex(a_dxyEtaBiasResiduals, i, dxyRes * cmToum, "1");
            PVValHelper::fillByIndex(a_dzEtaBiasResiduals, i, dzRes * cmToum, "2");
            PVValHelper::fillByIndex(n_dxyEtaBiasResiduals, i, (dxyRes) / dxy_err, "3");
            PVValHelper::fillByIndex(n_dzEtaBiasResiduals, i, (dzRes) / dz_err, "4");
          }

          if (trackphi >= phiF && trackphi < phiL) {
            PVValHelper::fillByIndex(a_dxyPhiBiasResiduals, i, dxyRes * cmToum, "5");
            PVValHelper::fillByIndex(a_dzPhiBiasResiduals, i, dzRes * cmToum, "6");
            PVValHelper::fillByIndex(n_dxyPhiBiasResiduals, i, (dxyRes) / dxy_err, "7");
            PVValHelper::fillByIndex(n_dzPhiBiasResiduals, i, (dzRes) / dz_err, "8");

            for (int j = 0; j < nBins_; j++) {
              float etaJ = theDetails_.trendbins[PVValHelper::eta][j];
              float etaK = theDetails_.trendbins[PVValHelper::eta][j + 1];

              if (tracketa >= etaJ && tracketa < etaK) {
                a_dxyBiasResidualsMap[i][j]->Fill(dxyRes * cmToum);
                a_dzBiasResidualsMap[i][j]->Fill(dzRes * cmToum);

                n_dxyBiasResidualsMap[i][j]->Fill((dxyRes) / dxy_err);
                n_dzBiasResidualsMap[i][j]->Fill((dzRes) / dz_err);
              }
            }
          }
        }
      }

      h_recoVtxSumPt_->Fill(sumpt);
    }
  }

  //=======================================================
  // Retrieve Beamspot information
  //=======================================================

  BeamSpot beamSpot;
  edm::Handle<BeamSpot> beamSpotHandle;
  iEvent.getByToken(theBeamspotToken, beamSpotHandle);

  if (beamSpotHandle.isValid()) {
    beamSpot = *beamSpotHandle;
    BSx0_ = beamSpot.x0();
    BSy0_ = beamSpot.y0();
    BSz0_ = beamSpot.z0();
    Beamsigmaz_ = beamSpot.sigmaZ();
    Beamdxdz_ = beamSpot.dxdz();
    BeamWidthX_ = beamSpot.BeamWidthX();
    BeamWidthY_ = beamSpot.BeamWidthY();

    wxy2_ = TMath::Power(BeamWidthX_, 2) + TMath::Power(BeamWidthY_, 2);

  } else {
    edm::LogWarning("PrimaryVertexValidation") << "No BeamSpot found!";
  }

  h_BSx0->Fill(BSx0_);
  h_BSy0->Fill(BSy0_);
  h_BSz0->Fill(BSz0_);
  h_Beamsigmaz->Fill(Beamsigmaz_);
  h_BeamWidthX->Fill(BeamWidthX_);
  h_BeamWidthY->Fill(BeamWidthY_);

  if (debug_)
    edm::LogInfo("PrimaryVertexValidation") << "Beamspot x:" << BSx0_ << " y:" << BSy0_ << " z:" << BSz0_;

  //=======================================================
  // Starts here ananlysis
  //=======================================================

  RunNumber_ = iEvent.eventAuxiliary().run();
  h_runNumber->Fill(RunNumber_);

  if (!runNumbersTimesLog_.count(RunNumber_)) {
    auto times = getRunTime(iSetup);

    if (debug_) {
      const time_t start_time = times.first / 1000000;
      edm::LogInfo("PrimaryVertexValidation")
          << RunNumber_ << " has start time: " << times.first << " - " << times.second << std::endl;
      edm::LogInfo("PrimaryVertexValidation")
          << "human readable time: " << std::asctime(std::gmtime(&start_time)) << std::endl;
    }

    runNumbersTimesLog_[RunNumber_] = times;
  }

  if (h_runFromEvent->GetEntries() == 0) {
    h_runFromEvent->SetBinContent(1, RunNumber_);
  }

  LuminosityBlockNumber_ = iEvent.eventAuxiliary().luminosityBlock();
  EventNumber_ = iEvent.eventAuxiliary().id().event();

  if (debug_)
    edm::LogInfo("PrimaryVertexValidation") << " looping over " << trackCollectionHandle->size() << "tracks";

  h_nTracks->Fill(trackCollectionHandle->size());

  //======================================================
  // Interface RECO tracks to vertex reconstruction
  //======================================================

  std::vector<TransientTrack> t_tks;
  for (const auto& track : tracks) {
    TransientTrack tt = theB_->build(&(track));
    tt.setBeamSpot(beamSpot);
    t_tks.push_back(tt);
  }

  if (debug_) {
    edm::LogInfo("PrimaryVertexValidation") << "Found: " << t_tks.size() << " reconstructed tracks";
  }

  //======================================================
  // select the tracks
  //======================================================

  std::vector<TransientTrack> seltks = theTrackFilter_->select(t_tks);

  //======================================================
  // clusterize tracks in Z
  //======================================================

  vector<vector<TransientTrack>> clusters = theTrackClusterizer_->clusterize(seltks);

  if (debug_) {
    edm::LogInfo("PrimaryVertexValidation")
        << " looping over: " << clusters.size() << " clusters  from " << t_tks.size() << " selected tracks";
  }

  nClus_ = clusters.size();
  h_nClus->Fill(nClus_);

  //======================================================
  // Starts loop on clusters
  //======================================================
  for (const auto& iclus : clusters) {
    nTracksPerClus_ = 0;

    unsigned int i = 0;
    for (const auto& theTTrack : iclus) {
      i++;

      if (nTracks_ >= nMaxtracks_) {
        edm::LogError("PrimaryVertexValidation")
            << " Warning - Number of tracks: " << nTracks_ << " , greater than " << nMaxtracks_;
        continue;
      }

      const Track& theTrack = theTTrack.track();

      pt_[nTracks_] = theTrack.pt();
      p_[nTracks_] = theTrack.p();
      nhits_[nTracks_] = theTrack.numberOfValidHits();
      eta_[nTracks_] = theTrack.eta();
      theta_[nTracks_] = theTrack.theta();
      phi_[nTracks_] = theTrack.phi();
      chi2_[nTracks_] = theTrack.chi2();
      chi2ndof_[nTracks_] = theTrack.normalizedChi2();
      charge_[nTracks_] = theTrack.charge();
      qoverp_[nTracks_] = theTrack.qoverp();
      dz_[nTracks_] = theTrack.dz();
      dxy_[nTracks_] = theTrack.dxy();

      TrackBase::TrackQuality _trackQuality = TrackBase::qualityByName("highPurity");
      isHighPurity_[nTracks_] = theTrack.quality(_trackQuality);

      math::XYZPoint point(BSx0_, BSy0_, BSz0_);
      dxyBs_[nTracks_] = theTrack.dxy(point);
      dzBs_[nTracks_] = theTrack.dz(point);

      xPCA_[nTracks_] = theTrack.vertex().x();
      yPCA_[nTracks_] = theTrack.vertex().y();
      zPCA_[nTracks_] = theTrack.vertex().z();

      //=======================================================
      // Retrieve rechit information
      //=======================================================

      const reco::HitPattern& hits = theTrack.hitPattern();

      int nRecHit1D = 0;
      int nRecHit2D = 0;
      int nhitinTIB = hits.numberOfValidStripTIBHits();
      int nhitinTOB = hits.numberOfValidStripTOBHits();
      int nhitinTID = hits.numberOfValidStripTIDHits();
      int nhitinTEC = hits.numberOfValidStripTECHits();
      int nhitinBPIX = hits.numberOfValidPixelBarrelHits();
      int nhitinFPIX = hits.numberOfValidPixelEndcapHits();
      for (trackingRecHit_iterator iHit = theTTrack.recHitsBegin(); iHit != theTTrack.recHitsEnd(); ++iHit) {
        if ((*iHit)->isValid()) {
          if (this->isHit2D(**iHit, phase_)) {
            ++nRecHit2D;
          } else {
            ++nRecHit1D;
          }
        }
      }

      nhits1D_[nTracks_] = nRecHit1D;
      nhits2D_[nTracks_] = nRecHit2D;
      nhitsBPIX_[nTracks_] = nhitinBPIX;
      nhitsFPIX_[nTracks_] = nhitinFPIX;
      nhitsTIB_[nTracks_] = nhitinTIB;
      nhitsTID_[nTracks_] = nhitinTID;
      nhitsTOB_[nTracks_] = nhitinTOB;
      nhitsTEC_[nTracks_] = nhitinTEC;

      //=======================================================
      // Good tracks for vertexing selection
      //=======================================================

      bool pass = true;
      if (askFirstLayerHit_)
        pass = this->hasFirstLayerPixelHits(theTTrack);
      if (pass && (theTrack.pt() >= ptOfProbe_) && std::abs(theTrack.eta()) <= etaOfProbe_ &&
          (theTrack.numberOfValidHits()) >= nHitsOfProbe_ && (theTrack.p()) >= pOfProbe_) {
        isGoodTrack_[nTracks_] = 1;
      }

      //=======================================================
      // Fit unbiased vertex
      //=======================================================

      vector<TransientTrack> theFinalTracks;
      theFinalTracks.clear();

      for (const auto& tk : iclus) {
        pass = this->hasFirstLayerPixelHits(tk);
        if (pass) {
          if (tk == theTTrack)
            continue;
          else {
            theFinalTracks.push_back(tk);
          }
        }
      }

      if (theFinalTracks.size() > 1) {
        if (debug_)
          edm::LogInfo("PrimaryVertexValidation") << "Transient Track Collection size: " << theFinalTracks.size();
        try {
          //AdaptiveVertexFitter* theFitter = new AdaptiveVertexFitter;
          auto theFitter = std::unique_ptr<VertexFitter<5>>(new AdaptiveVertexFitter());
          TransientVertex theFittedVertex;

          if (forceBeamSpotContraint_) {
            theFittedVertex = theFitter->vertex(theFinalTracks, beamSpot);  // if you want the beam constraint
          } else {
            theFittedVertex = theFitter->vertex(theFinalTracks);
          }

          double totalTrackWeights = 0;
          if (theFittedVertex.isValid()) {
            if (theFittedVertex.hasTrackWeight()) {
              for (const auto& theFinalTrack : theFinalTracks) {
                sumOfWeightsUnbiasedVertex_[nTracks_] += theFittedVertex.trackWeight(theFinalTrack);
                totalTrackWeights += theFittedVertex.trackWeight(theFinalTrack);
                h_fitVtxTrackWeights_->Fill(theFittedVertex.trackWeight(theFinalTrack));
              }
            }

            h_fitVtxTrackAverageWeight_->Fill(totalTrackWeights / theFinalTracks.size());

            const math::XYZPoint theRecoVertex(xOfflineVertex_, yOfflineVertex_, zOfflineVertex_);
            const math::XYZPoint myVertex(
                theFittedVertex.position().x(), theFittedVertex.position().y(), theFittedVertex.position().z());

            const Vertex vertex = theFittedVertex;
            fillTrackHistos(hDA, "all", &theTTrack, vertex, beamSpot, fBfield_);

            hasRecVertex_[nTracks_] = 1;
            xUnbiasedVertex_[nTracks_] = theFittedVertex.position().x();
            yUnbiasedVertex_[nTracks_] = theFittedVertex.position().y();
            zUnbiasedVertex_[nTracks_] = theFittedVertex.position().z();

            chi2normUnbiasedVertex_[nTracks_] = theFittedVertex.normalisedChiSquared();
            chi2UnbiasedVertex_[nTracks_] = theFittedVertex.totalChiSquared();
            DOFUnbiasedVertex_[nTracks_] = theFittedVertex.degreesOfFreedom();
            chi2ProbUnbiasedVertex_[nTracks_] =
                TMath::Prob(theFittedVertex.totalChiSquared(), (int)theFittedVertex.degreesOfFreedom());
            tracksUsedForVertexing_[nTracks_] = theFinalTracks.size();

            h_fitVtxNtracks_->Fill(theFinalTracks.size());
            h_fitVtxChi2_->Fill(theFittedVertex.totalChiSquared());
            h_fitVtxNdof_->Fill(theFittedVertex.degreesOfFreedom());
            h_fitVtxChi2ndf_->Fill(theFittedVertex.normalisedChiSquared());
            h_fitVtxChi2Prob_->Fill(
                TMath::Prob(theFittedVertex.totalChiSquared(), (int)theFittedVertex.degreesOfFreedom()));

            // from my Vertex
            double dxyFromMyVertex = theTrack.dxy(myVertex);
            double dzFromMyVertex = theTrack.dz(myVertex);

            GlobalPoint vert(
                theFittedVertex.position().x(), theFittedVertex.position().y(), theFittedVertex.position().z());

            //FreeTrajectoryState theTrackNearVertex = theTTrack.trajectoryStateClosestToPoint(vert).theState();
            //double dz_err = sqrt(theFittedVertex.positionError().czz() + theTrackNearVertex.cartesianError().position().czz());
            //double dz_err = hypot(theTrack.dzError(),theFittedVertex.positionError().czz());

            double dz_err = sqrt(std::pow(theTrack.dzError(), 2) + theFittedVertex.positionError().czz());

            // PV2D
            std::pair<bool, Measurement1D> s_ip2dpv = signedTransverseImpactParameter(
                theTTrack, GlobalVector(theTrack.px(), theTrack.py(), theTrack.pz()), theFittedVertex);

            double s_ip2dpv_corr = s_ip2dpv.second.value();
            double s_ip2dpv_err = s_ip2dpv.second.error();

            // PV3D
            std::pair<bool, Measurement1D> s_ip3dpv = signedImpactParameter3D(
                theTTrack, GlobalVector(theTrack.px(), theTrack.py(), theTrack.pz()), theFittedVertex);

            double s_ip3dpv_corr = s_ip3dpv.second.value();
            double s_ip3dpv_err = s_ip3dpv.second.error();

            // PV3D absolute
            std::pair<bool, Measurement1D> ip3dpv = absoluteImpactParameter3D(theTTrack, theFittedVertex);
            double ip3d_corr = ip3dpv.second.value();
            double ip3d_err = ip3dpv.second.error();

            // with respect to any specified vertex, such as primary vertex
            TrajectoryStateClosestToPoint traj = (theTTrack).trajectoryStateClosestToPoint(vert);

            GlobalPoint refPoint = traj.position();
            GlobalPoint cPToVtx = traj.theState().position();

            float my_dx = refPoint.x() - myVertex.x();
            float my_dy = refPoint.y() - myVertex.y();

            float my_dx2 = cPToVtx.x() - myVertex.x();
            float my_dy2 = cPToVtx.y() - myVertex.y();

            float my_dxy = std::sqrt(my_dx * my_dx + my_dy * my_dy);

            double d0 = traj.perigeeParameters().transverseImpactParameter();
            //double d0_error = traj.perigeeError().transverseImpactParameterError();
            double z0 = traj.perigeeParameters().longitudinalImpactParameter();
            double z0_error = traj.perigeeError().longitudinalImpactParameterError();

            if (debug_) {
              edm::LogInfo("PrimaryVertexValidation")
                  << "my_dx:" << my_dx << " my_dy:" << my_dy << " my_dxy:" << my_dxy << " my_dx2:" << my_dx2
                  << " my_dy2:" << my_dy2 << " d0: " << d0 << " dxyFromVtx:" << dxyFromMyVertex << "\n"
                  << " ============================== "
                  << "\n"
                  << "diff1:" << std::abs(d0) - std::abs(my_dxy) << "\n"
                  << "diff2:" << std::abs(d0) - std::abs(dxyFromMyVertex) << "\n"
                  << "diff3:" << (my_dx - my_dx2) << " " << (my_dy - my_dy2) << "\n"
                  << std::endl;
            }

            // define IPs

            dxyFromMyVertex_[nTracks_] = dxyFromMyVertex;
            dxyErrorFromMyVertex_[nTracks_] = s_ip2dpv_err;
            IPTsigFromMyVertex_[nTracks_] = dxyFromMyVertex / s_ip2dpv_err;

            dzFromMyVertex_[nTracks_] = dzFromMyVertex;
            dzErrorFromMyVertex_[nTracks_] = dz_err;
            IPLsigFromMyVertex_[nTracks_] = dzFromMyVertex / dz_err;

            d3DFromMyVertex_[nTracks_] = ip3d_corr;
            d3DErrorFromMyVertex_[nTracks_] = ip3d_err;
            IP3DsigFromMyVertex_[nTracks_] = (ip3d_corr / ip3d_err);

            // fill directly the histograms of residuals

            float trackphi = (theTrack.phi()) * (180. / M_PI);
            float tracketa = theTrack.eta();
            float trackpt = theTrack.pt();
            float trackp = theTrack.p();
            float tracknhits = theTrack.numberOfValidHits();

            // determine the module number and ladder

            int ladder_num = -1.;
            int module_num = -1.;
            int L1BPixHitCount = 0;

            for (auto const& hit : theTrack.recHits()) {
              const DetId& detId = hit->geographicalId();
              unsigned int subid = detId.subdetId();

              if (hit->isValid() && (subid == PixelSubdetector::PixelBarrel)) {
                int layer = tTopo->pxbLayer(detId);
                if (layer == 1) {
                  const SiPixelRecHit* prechit = dynamic_cast<const SiPixelRecHit*>(
                      hit);  //to be used to get the associated cluster and the cluster probability
                  double clusterProbability = prechit->clusterProbability(0);
                  if (clusterProbability > 0) {
                    h_probeL1ClusterProb_->Fill(log10(clusterProbability));
                  }

                  L1BPixHitCount += 1;
                  ladder_num = tTopo->pxbLadder(detId);
                  module_num = tTopo->pxbModule(detId);
                }
              }
            }

            h_probeL1Ladder_->Fill(ladder_num);
            h_probeL1Module_->Fill(module_num);
            h2_probeLayer1Map_->Fill(module_num, ladder_num);
            h_probeHasBPixL1Overlap_->Fill(L1BPixHitCount);

            // residuals vs ladder and module number for map
            if (module_num > 0 && ladder_num > 0) {  // only if we are on BPix Layer 1
              a_dxyL1ResidualsMap[ladder_num - 1][module_num - 1]->Fill(dxyFromMyVertex * cmToum);
              a_dzL1ResidualsMap[ladder_num - 1][module_num - 1]->Fill(dzFromMyVertex * cmToum);
              n_dxyL1ResidualsMap[ladder_num - 1][module_num - 1]->Fill(dxyFromMyVertex / s_ip2dpv_err);
              n_dzL1ResidualsMap[ladder_num - 1][module_num - 1]->Fill(dzFromMyVertex / dz_err);
            }

            // filling the pT-binned distributions

            for (int ipTBin = 0; ipTBin < nPtBins_; ipTBin++) {
              float pTF = mypT_bins_[ipTBin];
              float pTL = mypT_bins_[ipTBin + 1];

              if (debug_)
                edm::LogInfo("PrimaryVertexValidation") << "ipTBin:" << ipTBin << " " << mypT_bins_[ipTBin]
                                                        << " < pT < " << mypT_bins_[ipTBin + 1] << std::endl;

              if (std::abs(tracketa) < 1.5 && (trackpt >= pTF && trackpt < pTL)) {
                if (debug_)
                  edm::LogInfo("PrimaryVertexValidation") << "passes this cut: " << mypT_bins_[ipTBin] << std::endl;
                PVValHelper::fillByIndex(h_dxy_pT_, ipTBin, dxyFromMyVertex * cmToum, "9");
                PVValHelper::fillByIndex(h_dz_pT_, ipTBin, dzFromMyVertex * cmToum, "10");
                PVValHelper::fillByIndex(h_norm_dxy_pT_, ipTBin, dxyFromMyVertex / s_ip2dpv_err, "11");
                PVValHelper::fillByIndex(h_norm_dz_pT_, ipTBin, dzFromMyVertex / dz_err, "12");

                if (std::abs(tracketa) < 1.) {
                  if (debug_)
                    edm::LogInfo("PrimaryVertexValidation")
                        << "passes tight eta cut: " << mypT_bins_[ipTBin] << std::endl;
                  PVValHelper::fillByIndex(h_dxy_Central_pT_, ipTBin, dxyFromMyVertex * cmToum, "13");
                  PVValHelper::fillByIndex(h_dz_Central_pT_, ipTBin, dzFromMyVertex * cmToum, "14");
                  PVValHelper::fillByIndex(h_norm_dxy_Central_pT_, ipTBin, dxyFromMyVertex / s_ip2dpv_err, "15");
                  PVValHelper::fillByIndex(h_norm_dz_Central_pT_, ipTBin, dzFromMyVertex / dz_err, "16");
                }
              }
            }

            // checks on the probe track quality
            if (trackpt >= ptOfProbe_ && std::abs(tracketa) <= etaOfProbe_ && tracknhits >= nHitsOfProbe_ &&
                trackp >= pOfProbe_) {
              std::pair<bool, bool> pixelOcc = pixelHitsCheck((theTTrack));

              if (debug_) {
                if (pixelOcc.first == true)
                  edm::LogInfo("PrimaryVertexValidation") << "has BPIx hits" << std::endl;
                if (pixelOcc.second == true)
                  edm::LogInfo("PrimaryVertexValidation") << "has FPix hits" << std::endl;
              }

              if (!doBPix_ && (pixelOcc.first == true))
                continue;
              if (!doFPix_ && (pixelOcc.second == true))
                continue;

              fillTrackHistos(hDA, "sel", &(theTTrack), vertex, beamSpot, fBfield_);

              // probe checks
              h_probePt_->Fill(theTrack.pt());
              h_probePtRebin_->Fill(theTrack.pt());
              h_probeP_->Fill(theTrack.p());
              h_probeEta_->Fill(theTrack.eta());
              h_probePhi_->Fill(theTrack.phi());
              h2_probeEtaPhi_->Fill(theTrack.eta(), theTrack.phi());
              h2_probeEtaPt_->Fill(theTrack.eta(), theTrack.pt());

              h_probeChi2_->Fill(theTrack.chi2());
              h_probeNormChi2_->Fill(theTrack.normalizedChi2());
              h_probeCharge_->Fill(theTrack.charge());
              h_probeQoverP_->Fill(theTrack.qoverp());
              h_probeHits_->Fill(theTrack.numberOfValidHits());
              h_probeHits1D_->Fill(nRecHit1D);
              h_probeHits2D_->Fill(nRecHit2D);
              h_probeHitsInTIB_->Fill(nhitinTIB);
              h_probeHitsInTOB_->Fill(nhitinTOB);
              h_probeHitsInTID_->Fill(nhitinTID);
              h_probeHitsInTEC_->Fill(nhitinTEC);
              h_probeHitsInBPIX_->Fill(nhitinBPIX);
              h_probeHitsInFPIX_->Fill(nhitinFPIX);

              float dxyRecoV = theTrack.dz(theRecoVertex);
              float dzRecoV = theTrack.dxy(theRecoVertex);
              float dxysigmaRecoV =
                  TMath::Sqrt(theTrack.d0Error() * theTrack.d0Error() + xErrOfflineVertex_ * yErrOfflineVertex_);
              float dzsigmaRecoV =
                  TMath::Sqrt(theTrack.dzError() * theTrack.dzError() + zErrOfflineVertex_ * zErrOfflineVertex_);

              double zTrack = (theTTrack.stateAtBeamLine().trackStateAtPCA()).position().z();
              double zVertex = theFittedVertex.position().z();
              double tantheta = tan((theTTrack.stateAtBeamLine().trackStateAtPCA()).momentum().theta());

              double dz2 = pow(theTrack.dzError(), 2) + wxy2_ / pow(tantheta, 2);
              double restrkz = zTrack - zVertex;
              double pulltrkz = (zTrack - zVertex) / TMath::Sqrt(dz2);

              h_probedxyRecoV_->Fill(dxyRecoV);
              h_probedzRecoV_->Fill(dzRecoV);

              h_probedzRefitV_->Fill(dxyFromMyVertex);
              h_probedxyRefitV_->Fill(dzFromMyVertex);

              h_probed0RefitV_->Fill(d0);
              h_probez0RefitV_->Fill(z0);

              h_probesignIP2DRefitV_->Fill(s_ip2dpv_corr);
              h_probed3DRefitV_->Fill(ip3d_corr);
              h_probereszRefitV_->Fill(restrkz);

              h_probeRecoVSigZ_->Fill(dzRecoV / dzsigmaRecoV);
              h_probeRecoVSigXY_->Fill(dxyRecoV / dxysigmaRecoV);
              h_probeRefitVSigZ_->Fill(dzFromMyVertex / dz_err);
              h_probeRefitVSigXY_->Fill(dxyFromMyVertex / s_ip2dpv_err);
              h_probeRefitVSig3D_->Fill(ip3d_corr / ip3d_err);
              h_probeRefitVLogSig3D_->Fill(log10(ip3d_corr / ip3d_err));
              h_probeRefitVSigResZ_->Fill(pulltrkz);

              a_dxyVsPhi->Fill(trackphi, dxyFromMyVertex * cmToum);
              a_dzVsPhi->Fill(trackphi, z0 * cmToum);
              n_dxyVsPhi->Fill(trackphi, dxyFromMyVertex / s_ip2dpv_err);
              n_dzVsPhi->Fill(trackphi, z0 / z0_error);

              a_dxyVsEta->Fill(tracketa, dxyFromMyVertex * cmToum);
              a_dzVsEta->Fill(tracketa, z0 * cmToum);
              n_dxyVsEta->Fill(tracketa, dxyFromMyVertex / s_ip2dpv_err);
              n_dzVsEta->Fill(tracketa, z0 / z0_error);

              if (ladder_num > 0 && module_num > 0) {
                LogDebug("PrimaryVertexValidation")
                    << " ladder_num: " << ladder_num << " module_num: " << module_num << std::endl;

                PVValHelper::fillByIndex(h_dxy_modZ_, module_num - 1, dxyFromMyVertex * cmToum, "17");
                PVValHelper::fillByIndex(h_dz_modZ_, module_num - 1, dzFromMyVertex * cmToum, "18");
                PVValHelper::fillByIndex(h_norm_dxy_modZ_, module_num - 1, dxyFromMyVertex / s_ip2dpv_err, "19");
                PVValHelper::fillByIndex(h_norm_dz_modZ_, module_num - 1, dzFromMyVertex / dz_err, "20");

                PVValHelper::fillByIndex(h_dxy_ladder_, ladder_num - 1, dxyFromMyVertex * cmToum, "21");

                LogDebug("PrimaryVertexValidation") << "h_dxy_ladder size:" << h_dxy_ladder_.size() << std::endl;

                if (L1BPixHitCount == 1) {
                  PVValHelper::fillByIndex(h_dxy_ladderNoOverlap_, ladder_num - 1, dxyFromMyVertex * cmToum);
                } else {
                  PVValHelper::fillByIndex(h_dxy_ladderOverlap_, ladder_num - 1, dxyFromMyVertex * cmToum);
                }

                h2_probePassingLayer1Map_->Fill(module_num, ladder_num);

                PVValHelper::fillByIndex(h_dz_ladder_, ladder_num - 1, dzFromMyVertex * cmToum, "22");
                PVValHelper::fillByIndex(h_norm_dxy_ladder_, ladder_num - 1, dxyFromMyVertex / s_ip2dpv_err, "23");
                PVValHelper::fillByIndex(h_norm_dz_ladder_, ladder_num - 1, dzFromMyVertex / dz_err, "24");
              }

              // filling the binned distributions
              for (int i = 0; i < nBins_; i++) {
                float phiF = theDetails_.trendbins[PVValHelper::phi][i];
                float phiL = theDetails_.trendbins[PVValHelper::phi][i + 1];

                float etaF = theDetails_.trendbins[PVValHelper::eta][i];
                float etaL = theDetails_.trendbins[PVValHelper::eta][i + 1];

                if (tracketa >= etaF && tracketa < etaL) {
                  PVValHelper::fillByIndex(a_dxyEtaResiduals, i, dxyFromMyVertex * cmToum, "25");
                  PVValHelper::fillByIndex(a_dxEtaResiduals, i, my_dx * cmToum, "26");
                  PVValHelper::fillByIndex(a_dyEtaResiduals, i, my_dy * cmToum, "27");
                  PVValHelper::fillByIndex(a_dzEtaResiduals, i, dzFromMyVertex * cmToum, "28");
                  PVValHelper::fillByIndex(n_dxyEtaResiduals, i, dxyFromMyVertex / s_ip2dpv_err, "29");
                  PVValHelper::fillByIndex(n_dzEtaResiduals, i, dzFromMyVertex / dz_err, "30");
                  PVValHelper::fillByIndex(a_IP2DEtaResiduals, i, s_ip2dpv_corr * cmToum, "31");
                  PVValHelper::fillByIndex(n_IP2DEtaResiduals, i, s_ip2dpv_corr / s_ip2dpv_err, "32");
                  PVValHelper::fillByIndex(a_reszEtaResiduals, i, restrkz * cmToum, "33");
                  PVValHelper::fillByIndex(n_reszEtaResiduals, i, pulltrkz, "34");
                  PVValHelper::fillByIndex(a_d3DEtaResiduals, i, ip3d_corr * cmToum, "35");
                  PVValHelper::fillByIndex(n_d3DEtaResiduals, i, ip3d_corr / ip3d_err, "36");
                  PVValHelper::fillByIndex(a_IP3DEtaResiduals, i, s_ip3dpv_corr * cmToum, "37");
                  PVValHelper::fillByIndex(n_IP3DEtaResiduals, i, s_ip3dpv_corr / s_ip3dpv_err, "38");
                }

                if (trackphi >= phiF && trackphi < phiL) {
                  PVValHelper::fillByIndex(a_dxyPhiResiduals, i, dxyFromMyVertex * cmToum, "39");
                  PVValHelper::fillByIndex(a_dxPhiResiduals, i, my_dx * cmToum, "40");
                  PVValHelper::fillByIndex(a_dyPhiResiduals, i, my_dy * cmToum, "41");
                  PVValHelper::fillByIndex(a_dzPhiResiduals, i, dzFromMyVertex * cmToum, "42");
                  PVValHelper::fillByIndex(n_dxyPhiResiduals, i, dxyFromMyVertex / s_ip2dpv_err, "43");
                  PVValHelper::fillByIndex(n_dzPhiResiduals, i, dzFromMyVertex / dz_err, "44");
                  PVValHelper::fillByIndex(a_IP2DPhiResiduals, i, s_ip2dpv_corr * cmToum, "45");
                  PVValHelper::fillByIndex(n_IP2DPhiResiduals, i, s_ip2dpv_corr / s_ip2dpv_err, "46");
                  PVValHelper::fillByIndex(a_reszPhiResiduals, i, restrkz * cmToum, "47");
                  PVValHelper::fillByIndex(n_reszPhiResiduals, i, pulltrkz, "48");
                  PVValHelper::fillByIndex(a_d3DPhiResiduals, i, ip3d_corr * cmToum, "49");
                  PVValHelper::fillByIndex(n_d3DPhiResiduals, i, ip3d_corr / ip3d_err, "50");
                  PVValHelper::fillByIndex(a_IP3DPhiResiduals, i, s_ip3dpv_corr * cmToum, "51");
                  PVValHelper::fillByIndex(n_IP3DPhiResiduals, i, s_ip3dpv_corr / s_ip3dpv_err, "52");

                  for (int j = 0; j < nBins_; j++) {
                    float etaJ = theDetails_.trendbins[PVValHelper::eta][j];
                    float etaK = theDetails_.trendbins[PVValHelper::eta][j + 1];

                    if (tracketa >= etaJ && tracketa < etaK) {
                      a_dxyResidualsMap[i][j]->Fill(dxyFromMyVertex * cmToum);
                      a_dzResidualsMap[i][j]->Fill(dzFromMyVertex * cmToum);
                      n_dxyResidualsMap[i][j]->Fill(dxyFromMyVertex / s_ip2dpv_err);
                      n_dzResidualsMap[i][j]->Fill(dzFromMyVertex / dz_err);
                      a_d3DResidualsMap[i][j]->Fill(ip3d_corr * cmToum);
                      n_d3DResidualsMap[i][j]->Fill(ip3d_corr / ip3d_err);
                    }
                  }
                }
              }
            }

            if (debug_) {
              edm::LogInfo("PrimaryVertexValidation")
                  << " myVertex.x()= " << myVertex.x() << "\n"
                  << " myVertex.y()= " << myVertex.y() << " \n"
                  << " myVertex.z()= " << myVertex.z() << " \n"
                  << " theTrack.dz(myVertex)= " << theTrack.dz(myVertex) << " \n"
                  << " zPCA -myVertex.z() = " << (theTrack.vertex().z() - myVertex.z());

            }  // ends if debug_
          }    // ends if the fitted vertex is Valid

          //delete theFitter;

        } catch (cms::Exception& er) {
          LogTrace("PrimaryVertexValidation") << "caught std::exception " << er.what() << std::endl;
        }

      }  //ends if theFinalTracks.size() > 2

      else {
        if (debug_)
          edm::LogInfo("PrimaryVertexValidation") << "Not enough tracks to make a vertex.  Returns no vertex info";
      }

      ++nTracks_;
      ++nTracksPerClus_;

      if (debug_)
        edm::LogInfo("PrimaryVertexValidation") << "Track " << i << " : pT = " << theTrack.pt();

    }  // for loop on tracks
  }    // for loop on track clusters

  // Fill the TTree if needed

  if (storeNtuple_) {
    rootTree_->Fill();
  }
}

// ------------ method called to discriminate 1D from 2D hits  ------------
bool PrimaryVertexValidation::isHit2D(const TrackingRecHit& hit, const PVValHelper::detectorPhase& thePhase) const {
  if (hit.dimension() < 2) {
    return false;  // some (muon...) stuff really has RecHit1D
  } else {
    const DetId detId(hit.geographicalId());
    if (detId.det() == DetId::Tracker) {
      if (detId.subdetId() == PixelSubdetector::PixelBarrel || detId.subdetId() == PixelSubdetector::PixelEndcap) {
        return true;                                 // pixel is always 2D
      } else if (thePhase != PVValHelper::phase2) {  // should be SiStrip now
        if (dynamic_cast<const SiStripRecHit2D*>(&hit))
          return false;  // normal hit
        else if (dynamic_cast<const SiStripMatchedRecHit2D*>(&hit))
          return true;  // matched is 2D
        else if (dynamic_cast<const ProjectedSiStripRecHit2D*>(&hit))
          return false;  // crazy hit...
        else {
          edm::LogError("UnkownType") << "@SUB=PrimaryVertexValidation::isHit2D"
                                      << "Tracker hit not in pixel and neither SiStripRecHit2D nor "
                                      << "SiStripMatchedRecHit2D nor ProjectedSiStripRecHit2D.";
          return false;
        }
      } else {
        return false;
      }
    } else {  // not tracker??
      edm::LogWarning("DetectorMismatch") << "@SUB=AlignmentTrackSelector::isHit2D"
                                          << "Hit not in tracker with 'official' dimension >=2.";
      return true;  // dimension() >= 2 so accept that...
    }
  }
  // never reached...
}

// ------------ method to check the presence of pixel hits  ------------
std::pair<bool, bool> PrimaryVertexValidation::pixelHitsCheck(const reco::TransientTrack& track) {
  bool hasBPixHits = false;
  bool hasFPixHits = false;

  const reco::HitPattern& p = track.hitPattern();
  if (p.numberOfValidPixelEndcapHits() != 0) {
    hasFPixHits = true;
  }
  if (p.numberOfValidPixelBarrelHits() != 0) {
    hasBPixHits = true;
  }

  return std::make_pair(hasBPixHits, hasFPixHits);
}

// ------------ method to check the presence of pixel hits  ------------
bool PrimaryVertexValidation::hasFirstLayerPixelHits(const reco::TransientTrack& track) {
  using namespace reco;
  const HitPattern& p = track.hitPattern();
  for (int i = 0; i < p.numberOfAllHits(HitPattern::TRACK_HITS); i++) {
    uint32_t pattern = p.getHitPattern(HitPattern::TRACK_HITS, i);
    if (p.pixelBarrelHitFilter(pattern) || p.pixelEndcapHitFilter(pattern)) {
      if (p.getLayer(pattern) == 1) {
        if (p.validHitFilter(pattern)) {
          return true;
        }
      }
    }
  }
  return false;
}

// ------------ method called once each job before begining the event loop  ------------
void PrimaryVertexValidation::beginJob() {
  edm::LogInfo("PrimaryVertexValidation") << "######################################\n"
                                          << "Begin Job \n"
                                          << "######################################";

  // Define TTree for output
  Nevt_ = 0;
  if (compressionSettings_ > 0) {
    fs->file().SetCompressionSettings(compressionSettings_);
  }

  //  rootFile_ = new TFile(filename_.c_str(),"recreate");
  rootTree_ = fs->make<TTree>("tree", "PV Validation tree");

  // Track Paramters

  if (lightNtupleSwitch_) {
    rootTree_->Branch("EventNumber", &EventNumber_, "EventNumber/i");
    rootTree_->Branch("RunNumber", &RunNumber_, "RunNumber/i");
    rootTree_->Branch("LuminosityBlockNumber", &LuminosityBlockNumber_, "LuminosityBlockNumber/i");
    rootTree_->Branch("nOfflineVertices", &nOfflineVertices_, "nOfflineVertices/I");
    rootTree_->Branch("nTracks", &nTracks_, "nTracks/I");
    rootTree_->Branch("phi", &phi_, "phi[nTracks]/D");
    rootTree_->Branch("eta", &eta_, "eta[nTracks]/D");
    rootTree_->Branch("pt", &pt_, "pt[nTracks]/D");
    rootTree_->Branch("dxyFromMyVertex", &dxyFromMyVertex_, "dxyFromMyVertex[nTracks]/D");
    rootTree_->Branch("dzFromMyVertex", &dzFromMyVertex_, "dzFromMyVertex[nTracks]/D");
    rootTree_->Branch("d3DFromMyVertex", &d3DFromMyVertex_, "d3DFromMyVertex[nTracks]/D");
    rootTree_->Branch("IPTsigFromMyVertex", &IPTsigFromMyVertex_, "IPTsigFromMyVertex_[nTracks]/D");
    rootTree_->Branch("IPLsigFromMyVertex", &IPLsigFromMyVertex_, "IPLsigFromMyVertex_[nTracks]/D");
    rootTree_->Branch("IP3DsigFromMyVertex", &IP3DsigFromMyVertex_, "IP3DsigFromMyVertex_[nTracks]/D");
    rootTree_->Branch("hasRecVertex", &hasRecVertex_, "hasRecVertex[nTracks]/I");
    rootTree_->Branch("isGoodTrack", &isGoodTrack_, "isGoodTrack[nTracks]/I");
    rootTree_->Branch("isHighPurity", &isHighPurity_, "isHighPurity_[nTracks]/I");

  } else {
    rootTree_->Branch("nTracks", &nTracks_, "nTracks/I");
    rootTree_->Branch("nTracksPerClus", &nTracksPerClus_, "nTracksPerClus/I");
    rootTree_->Branch("nClus", &nClus_, "nClus/I");
    rootTree_->Branch("xOfflineVertex", &xOfflineVertex_, "xOfflineVertex/D");
    rootTree_->Branch("yOfflineVertex", &yOfflineVertex_, "yOfflineVertex/D");
    rootTree_->Branch("zOfflineVertex", &zOfflineVertex_, "zOfflineVertex/D");
    rootTree_->Branch("BSx0", &BSx0_, "BSx0/D");
    rootTree_->Branch("BSy0", &BSy0_, "BSy0/D");
    rootTree_->Branch("BSz0", &BSz0_, "BSz0/D");
    rootTree_->Branch("Beamsigmaz", &Beamsigmaz_, "Beamsigmaz/D");
    rootTree_->Branch("Beamdxdz", &Beamdxdz_, "Beamdxdz/D");
    rootTree_->Branch("BeamWidthX", &BeamWidthX_, "BeamWidthX/D");
    rootTree_->Branch("BeamWidthY", &BeamWidthY_, "BeamWidthY/D");
    rootTree_->Branch("pt", &pt_, "pt[nTracks]/D");
    rootTree_->Branch("p", &p_, "p[nTracks]/D");
    rootTree_->Branch("nhits", &nhits_, "nhits[nTracks]/I");
    rootTree_->Branch("nhits1D", &nhits1D_, "nhits1D[nTracks]/I");
    rootTree_->Branch("nhits2D", &nhits2D_, "nhits2D[nTracks]/I");
    rootTree_->Branch("nhitsBPIX", &nhitsBPIX_, "nhitsBPIX[nTracks]/I");
    rootTree_->Branch("nhitsFPIX", &nhitsFPIX_, "nhitsFPIX[nTracks]/I");
    rootTree_->Branch("nhitsTIB", &nhitsTIB_, "nhitsTIB[nTracks]/I");
    rootTree_->Branch("nhitsTID", &nhitsTID_, "nhitsTID[nTracks]/I");
    rootTree_->Branch("nhitsTOB", &nhitsTOB_, "nhitsTOB[nTracks]/I");
    rootTree_->Branch("nhitsTEC", &nhitsTEC_, "nhitsTEC[nTracks]/I");
    rootTree_->Branch("eta", &eta_, "eta[nTracks]/D");
    rootTree_->Branch("theta", &theta_, "theta[nTracks]/D");
    rootTree_->Branch("phi", &phi_, "phi[nTracks]/D");
    rootTree_->Branch("chi2", &chi2_, "chi2[nTracks]/D");
    rootTree_->Branch("chi2ndof", &chi2ndof_, "chi2ndof[nTracks]/D");
    rootTree_->Branch("charge", &charge_, "charge[nTracks]/I");
    rootTree_->Branch("qoverp", &qoverp_, "qoverp[nTracks]/D");
    rootTree_->Branch("dz", &dz_, "dz[nTracks]/D");
    rootTree_->Branch("dxy", &dxy_, "dxy[nTracks]/D");
    rootTree_->Branch("dzBs", &dzBs_, "dzBs[nTracks]/D");
    rootTree_->Branch("dxyBs", &dxyBs_, "dxyBs[nTracks]/D");
    rootTree_->Branch("xPCA", &xPCA_, "xPCA[nTracks]/D");
    rootTree_->Branch("yPCA", &yPCA_, "yPCA[nTracks]/D");
    rootTree_->Branch("zPCA", &zPCA_, "zPCA[nTracks]/D");
    rootTree_->Branch("xUnbiasedVertex", &xUnbiasedVertex_, "xUnbiasedVertex[nTracks]/D");
    rootTree_->Branch("yUnbiasedVertex", &yUnbiasedVertex_, "yUnbiasedVertex[nTracks]/D");
    rootTree_->Branch("zUnbiasedVertex", &zUnbiasedVertex_, "zUnbiasedVertex[nTracks]/D");
    rootTree_->Branch("chi2normUnbiasedVertex", &chi2normUnbiasedVertex_, "chi2normUnbiasedVertex[nTracks]/F");
    rootTree_->Branch("chi2UnbiasedVertex", &chi2UnbiasedVertex_, "chi2UnbiasedVertex[nTracks]/F");
    rootTree_->Branch("DOFUnbiasedVertex", &DOFUnbiasedVertex_, " DOFUnbiasedVertex[nTracks]/F");
    rootTree_->Branch("chi2ProbUnbiasedVertex", &chi2ProbUnbiasedVertex_, "chi2ProbUnbiasedVertex[nTracks]/F");
    rootTree_->Branch(
        "sumOfWeightsUnbiasedVertex", &sumOfWeightsUnbiasedVertex_, "sumOfWeightsUnbiasedVertex[nTracks]/F");
    rootTree_->Branch("tracksUsedForVertexing", &tracksUsedForVertexing_, "tracksUsedForVertexing[nTracks]/I");
    rootTree_->Branch("dxyFromMyVertex", &dxyFromMyVertex_, "dxyFromMyVertex[nTracks]/D");
    rootTree_->Branch("dzFromMyVertex", &dzFromMyVertex_, "dzFromMyVertex[nTracks]/D");
    rootTree_->Branch("dxyErrorFromMyVertex", &dxyErrorFromMyVertex_, "dxyErrorFromMyVertex_[nTracks]/D");
    rootTree_->Branch("dzErrorFromMyVertex", &dzErrorFromMyVertex_, "dzErrorFromMyVertex_[nTracks]/D");
    rootTree_->Branch("IPTsigFromMyVertex", &IPTsigFromMyVertex_, "IPTsigFromMyVertex_[nTracks]/D");
    rootTree_->Branch("IPLsigFromMyVertex", &IPLsigFromMyVertex_, "IPLsigFromMyVertex_[nTracks]/D");
    rootTree_->Branch("hasRecVertex", &hasRecVertex_, "hasRecVertex[nTracks]/I");
    rootTree_->Branch("isGoodTrack", &isGoodTrack_, "isGoodTrack[nTracks]/I");
  }

  // event histograms
  TFileDirectory EventFeatures = fs->mkdir("EventFeatures");

  TH1F::SetDefaultSumw2(kTRUE);

  h_lumiFromConfig =
      EventFeatures.make<TH1F>("h_lumiFromConfig", "luminosity from config;;luminosity of present run", 1, -0.5, 0.5);
  h_lumiFromConfig->SetBinContent(1, intLumi_);

  h_runFromConfig = EventFeatures.make<TH1I>("h_runFromConfig",
                                             "run number from config;;run number (from configuration)",
                                             runControlNumbers_.size(),
                                             0.,
                                             runControlNumbers_.size());

  for (const auto& run : runControlNumbers_ | boost::adaptors::indexed(1)) {
    h_runFromConfig->SetBinContent(run.index(), run.value());
  }

  h_runFromEvent =
      EventFeatures.make<TH1I>("h_runFromEvent", "run number from event;;run number (from event)", 1, -0.5, 0.5);
  h_nTracks =
      EventFeatures.make<TH1F>("h_nTracks", "number of tracks per event;n_{tracks}/event;n_{events}", 300, -0.5, 299.5);
  h_nClus =
      EventFeatures.make<TH1F>("h_nClus", "number of track clusters;n_{clusters}/event;n_{events}", 50, -0.5, 49.5);
  h_nOfflineVertices = EventFeatures.make<TH1F>(
      "h_nOfflineVertices", "number of offline reconstructed vertices;n_{vertices}/event;n_{events}", 50, -0.5, 49.5);
  h_runNumber = EventFeatures.make<TH1F>("h_runNumber", "run number;run number;n_{events}", 100000, 250000., 350000.);
  h_xOfflineVertex = EventFeatures.make<TH1F>(
      "h_xOfflineVertex", "x-coordinate of offline vertex;x_{vertex};n_{events}", 100, -0.1, 0.1);
  h_yOfflineVertex = EventFeatures.make<TH1F>(
      "h_yOfflineVertex", "y-coordinate of offline vertex;y_{vertex};n_{events}", 100, -0.1, 0.1);
  h_zOfflineVertex = EventFeatures.make<TH1F>(
      "h_zOfflineVertex", "z-coordinate of offline vertex;z_{vertex};n_{events}", 100, -30., 30.);
  h_xErrOfflineVertex = EventFeatures.make<TH1F>(
      "h_xErrOfflineVertex", "x-coordinate error of offline vertex;err_{x}^{vtx};n_{events}", 100, 0., 0.01);
  h_yErrOfflineVertex = EventFeatures.make<TH1F>(
      "h_yErrOfflineVertex", "y-coordinate error of offline vertex;err_{y}^{vtx};n_{events}", 100, 0., 0.01);
  h_zErrOfflineVertex = EventFeatures.make<TH1F>(
      "h_zErrOfflineVertex", "z-coordinate error of offline vertex;err_{z}^{vtx};n_{events}", 100, 0., 10.);
  h_BSx0 = EventFeatures.make<TH1F>("h_BSx0", "x-coordinate of reco beamspot;x^{BS}_{0};n_{events}", 100, -0.1, 0.1);
  h_BSy0 = EventFeatures.make<TH1F>("h_BSy0", "y-coordinate of reco beamspot;y^{BS}_{0};n_{events}", 100, -0.1, 0.1);
  h_BSz0 = EventFeatures.make<TH1F>("h_BSz0", "z-coordinate of reco beamspot;z^{BS}_{0};n_{events}", 100, -1., 1.);
  h_Beamsigmaz =
      EventFeatures.make<TH1F>("h_Beamsigmaz", "z-coordinate beam width;#sigma_{Z}^{beam};n_{events}", 100, 0., 1.);
  h_BeamWidthX =
      EventFeatures.make<TH1F>("h_BeamWidthX", "x-coordinate beam width;#sigma_{X}^{beam};n_{events}", 100, 0., 0.01);
  h_BeamWidthY =
      EventFeatures.make<TH1F>("h_BeamWidthY", "y-coordinate beam width;#sigma_{Y}^{beam};n_{events}", 100, 0., 0.01);

  h_etaMax = EventFeatures.make<TH1F>("etaMax", "etaMax", 1, -0.5, 0.5);
  h_pTinfo = EventFeatures.make<TH1F>("pTinfo", "pTinfo", 3, -1.5, 1.5);
  h_pTinfo->GetXaxis()->SetBinLabel(1, "n. bins");
  h_pTinfo->GetXaxis()->SetBinLabel(2, "pT min");
  h_pTinfo->GetXaxis()->SetBinLabel(3, "pT max");

  h_nbins = EventFeatures.make<TH1F>("nbins", "nbins", 1, -0.5, 0.5);
  h_nLadders = EventFeatures.make<TH1F>("nladders", "n. ladders", 1, -0.5, 0.5);
  h_nModZ = EventFeatures.make<TH1F>("nModZ", "n. modules along z", 1, -0.5, 0.5);

  // probe track histograms
  TFileDirectory ProbeFeatures = fs->mkdir("ProbeTrackFeatures");

  h_probePt_ = ProbeFeatures.make<TH1F>("h_probePt", "p_{T} of probe track;track p_{T} (GeV); tracks", 100, 0., 50.);
  h_probePtRebin_ = ProbeFeatures.make<TH1F>(
      "h_probePtRebin", "p_{T} of probe track;track p_{T} (GeV); tracks", mypT_bins_.size() - 1, mypT_bins_.data());
  h_probeP_ = ProbeFeatures.make<TH1F>("h_probeP", "momentum of probe track;track p (GeV); tracks", 100, 0., 100.);
  h_probeEta_ = ProbeFeatures.make<TH1F>(
      "h_probeEta", "#eta of the probe track;track #eta;tracks", 54, -etaOfProbe_, etaOfProbe_);
  h_probePhi_ = ProbeFeatures.make<TH1F>("h_probePhi", "#phi of probe track;track #phi (rad);tracks", 100, -3.15, 3.15);

  h2_probeEtaPhi_ =
      ProbeFeatures.make<TH2F>("h2_probeEtaPhi",
                               "probe track #phi vs #eta;#eta of probe track;track #phi of probe track (rad); tracks",
                               54,
                               -etaOfProbe_,
                               etaOfProbe_,
                               100,
                               -M_PI,
                               M_PI);
  h2_probeEtaPt_ = ProbeFeatures.make<TH2F>("h2_probeEtaPt",
                                            "probe track p_{T} vs #eta;#eta of probe track;track p_{T} (GeV); tracks",
                                            54,
                                            -etaOfProbe_,
                                            etaOfProbe_,
                                            100,
                                            0.,
                                            50.);

  h_probeChi2_ =
      ProbeFeatures.make<TH1F>("h_probeChi2", "#chi^{2} of probe track;track #chi^{2}; tracks", 100, 0., 100.);
  h_probeNormChi2_ = ProbeFeatures.make<TH1F>(
      "h_probeNormChi2", " normalized #chi^{2} of probe track;track #chi^{2}/ndof; tracks", 100, 0., 10.);
  h_probeCharge_ =
      ProbeFeatures.make<TH1F>("h_probeCharge", "charge of probe track;track charge Q;tracks", 3, -1.5, 1.5);
  h_probeQoverP_ =
      ProbeFeatures.make<TH1F>("h_probeQoverP", "q/p of probe track; track Q/p (GeV^{-1});tracks", 200, -1., 1.);
  h_probedzRecoV_ = ProbeFeatures.make<TH1F>(
      "h_probedzRecoV", "d_{z}(V_{offline}) of probe track;track d_{z}(V_{off}) (cm);tracks", 200, -1., 1.);
  h_probedxyRecoV_ = ProbeFeatures.make<TH1F>(
      "h_probedxyRecoV", "d_{xy}(V_{offline}) of probe track;track d_{xy}(V_{off}) (cm);tracks", 200, -1., 1.);
  h_probedzRefitV_ = ProbeFeatures.make<TH1F>(
      "h_probedzRefitV", "d_{z}(V_{refit}) of probe track;track d_{z}(V_{fit}) (cm);tracks", 200, -0.5, 0.5);
  h_probesignIP2DRefitV_ = ProbeFeatures.make<TH1F>(
      "h_probesignIPRefitV", "ip_{2D}(V_{refit}) of probe track;track ip_{2D}(V_{fit}) (cm);tracks", 200, -1., 1.);
  h_probedxyRefitV_ = ProbeFeatures.make<TH1F>(
      "h_probedxyRefitV", "d_{xy}(V_{refit}) of probe track;track d_{xy}(V_{fit}) (cm);tracks", 200, -0.5, 0.5);

  h_probez0RefitV_ = ProbeFeatures.make<TH1F>(
      "h_probez0RefitV", "z_{0}(V_{refit}) of probe track;track z_{0}(V_{fit}) (cm);tracks", 200, -1., 1.);
  h_probed0RefitV_ = ProbeFeatures.make<TH1F>(
      "h_probed0RefitV", "d_{0}(V_{refit}) of probe track;track d_{0}(V_{fit}) (cm);tracks", 200, -1., 1.);

  h_probed3DRefitV_ = ProbeFeatures.make<TH1F>(
      "h_probed3DRefitV", "d_{3D}(V_{refit}) of probe track;track d_{3D}(V_{fit}) (cm);tracks", 200, 0., 1.);
  h_probereszRefitV_ = ProbeFeatures.make<TH1F>(
      "h_probeReszRefitV", "z_{track} -z_{V_{refit}};track res_{z}(V_{refit}) (cm);tracks", 200, -1., 1.);

  h_probeRecoVSigZ_ = ProbeFeatures.make<TH1F>(
      "h_probeRecoVSigZ", "Longitudinal DCA Significance (reco);d_{z}(V_{off})/#sigma_{dz};tracks", 100, -8, 8);
  h_probeRecoVSigXY_ = ProbeFeatures.make<TH1F>(
      "h_probeRecoVSigXY", "Transverse DCA Significance (reco);d_{xy}(V_{off})/#sigma_{dxy};tracks", 100, -8, 8);
  h_probeRefitVSigZ_ = ProbeFeatures.make<TH1F>(
      "h_probeRefitVSigZ", "Longitudinal DCA Significance (refit);d_{z}(V_{fit})/#sigma_{dz};tracks", 100, -8, 8);
  h_probeRefitVSigXY_ = ProbeFeatures.make<TH1F>(
      "h_probeRefitVSigXY", "Transverse DCA Significance (refit);d_{xy}(V_{fit})/#sigma_{dxy};tracks", 100, -8, 8);
  h_probeRefitVSig3D_ = ProbeFeatures.make<TH1F>(
      "h_probeRefitVSig3D", "3D DCA Significance (refit);d_{3D}/#sigma_{3D};tracks", 100, 0., 20.);
  h_probeRefitVLogSig3D_ =
      ProbeFeatures.make<TH1F>("h_probeRefitVLogSig3D",
                               "log_{10}(3D DCA-Significance) (refit);log_{10}(d_{3D}/#sigma_{3D});tracks",
                               100,
                               -5.,
                               4.);
  h_probeRefitVSigResZ_ = ProbeFeatures.make<TH1F>(
      "h_probeRefitVSigResZ",
      "Longitudinal residual significance (refit);(z_{track} -z_{V_{fit}})/#sigma_{res_{z}};tracks",
      100,
      -8,
      8);

  h_probeHits_ = ProbeFeatures.make<TH1F>("h_probeNRechits", "N_{hits}     ;N_{hits}    ;tracks", 40, -0.5, 39.5);
  h_probeHits1D_ = ProbeFeatures.make<TH1F>("h_probeNRechits1D", "N_{hits} 1D  ;N_{hits} 1D ;tracks", 40, -0.5, 39.5);
  h_probeHits2D_ = ProbeFeatures.make<TH1F>("h_probeNRechits2D", "N_{hits} 2D  ;N_{hits} 2D ;tracks", 40, -0.5, 39.5);
  h_probeHitsInTIB_ =
      ProbeFeatures.make<TH1F>("h_probeNRechitsTIB", "N_{hits} TIB ;N_{hits} TIB;tracks", 40, -0.5, 39.5);
  h_probeHitsInTOB_ =
      ProbeFeatures.make<TH1F>("h_probeNRechitsTOB", "N_{hits} TOB ;N_{hits} TOB;tracks", 40, -0.5, 39.5);
  h_probeHitsInTID_ =
      ProbeFeatures.make<TH1F>("h_probeNRechitsTID", "N_{hits} TID ;N_{hits} TID;tracks", 40, -0.5, 39.5);
  h_probeHitsInTEC_ =
      ProbeFeatures.make<TH1F>("h_probeNRechitsTEC", "N_{hits} TEC ;N_{hits} TEC;tracks", 40, -0.5, 39.5);
  h_probeHitsInBPIX_ =
      ProbeFeatures.make<TH1F>("h_probeNRechitsBPIX", "N_{hits} BPIX;N_{hits} BPIX;tracks", 40, -0.5, 39.5);
  h_probeHitsInFPIX_ =
      ProbeFeatures.make<TH1F>("h_probeNRechitsFPIX", "N_{hits} FPIX;N_{hits} FPIX;tracks", 40, -0.5, 39.5);

  h_probeL1Ladder_ = ProbeFeatures.make<TH1F>(
      "h_probeL1Ladder", "Ladder number (L1 hit); ladder number", nLadders_ + 2, -1.5, nLadders_ + 0.5);
  h_probeL1Module_ = ProbeFeatures.make<TH1F>(
      "h_probeL1Module", "Module number (L1 hit); module number", nModZ_ + 2, -1.5, nModZ_ + 0.5);

  h2_probeLayer1Map_ = ProbeFeatures.make<TH2F>("h2_probeLayer1Map",
                                                "Position in Layer 1 of first hit;module number;ladder number",
                                                nModZ_,
                                                0.5,
                                                nModZ_ + 0.5,
                                                nLadders_,
                                                0.5,
                                                nLadders_ + 0.5);

  h2_probePassingLayer1Map_ = ProbeFeatures.make<TH2F>("h2_probePassingLayer1Map",
                                                       "Position in Layer 1 of first hit;module number;ladder number",
                                                       nModZ_,
                                                       0.5,
                                                       nModZ_ + 0.5,
                                                       nLadders_,
                                                       0.5,
                                                       nLadders_ + 0.5);
  h_probeHasBPixL1Overlap_ =
      ProbeFeatures.make<TH1I>("h_probeHasBPixL1Overlap", "n. hits in L1;n. L1-BPix hits;tracks", 5, -0.5, 4.5);
  h_probeL1ClusterProb_ = ProbeFeatures.make<TH1F>(
      "h_probeL1ClusterProb",
      "log_{10}(Cluster Probability) for Layer1 hits;log_{10}(cluster probability); n. Layer1 hits",
      100,
      -10.,
      0.);

  // refit vertex features
  TFileDirectory RefitVertexFeatures = fs->mkdir("RefitVertexFeatures");
  h_fitVtxNtracks_ = RefitVertexFeatures.make<TH1F>(
      "h_fitVtxNtracks", "N_{trks} used in vertex fit;N^{fit}_{tracks};vertices", 100, -0.5, 99.5);
  h_fitVtxNdof_ = RefitVertexFeatures.make<TH1F>(
      "h_fitVtxNdof", "N_{DOF} of vertex fit;N_{DOF} of refit vertex;vertices", 100, -0.5, 99.5);
  h_fitVtxChi2_ = RefitVertexFeatures.make<TH1F>(
      "h_fitVtxChi2", "#chi^{2} of vertex fit;vertex #chi^{2};vertices", 100, -0.5, 99.5);
  h_fitVtxChi2ndf_ = RefitVertexFeatures.make<TH1F>(
      "h_fitVtxChi2ndf", "#chi^{2}/ndf of vertex fit;vertex #chi^{2}/ndf;vertices", 100, -0.5, 9.5);
  h_fitVtxChi2Prob_ = RefitVertexFeatures.make<TH1F>(
      "h_fitVtxChi2Prob", "Prob(#chi^{2},ndf) of vertex fit;Prob(#chi^{2},ndf);vertices", 40, 0., 1.);
  h_fitVtxTrackWeights_ = RefitVertexFeatures.make<TH1F>(
      "h_fitVtxTrackWeights", "track weights associated to track;track weights;tracks", 40, 0., 1.);
  h_fitVtxTrackAverageWeight_ = RefitVertexFeatures.make<TH1F>(
      "h_fitVtxTrackAverageWeight_", "average track weight per vertex;#LT track weight #GT;vertices", 40, 0., 1.);

  if (useTracksFromRecoVtx_) {
    TFileDirectory RecoVertexFeatures = fs->mkdir("RecoVertexFeatures");
    h_recoVtxNtracks_ =
        RecoVertexFeatures.make<TH1F>("h_recoVtxNtracks", "N^{vtx}_{trks};N^{vtx}_{trks};vertices", 100, -0.5, 99.5);
    h_recoVtxChi2ndf_ =
        RecoVertexFeatures.make<TH1F>("h_recoVtxChi2ndf", "#chi^{2}/ndf vtx;#chi^{2}/ndf vtx;vertices", 10, -0.5, 9.5);
    h_recoVtxChi2Prob_ = RecoVertexFeatures.make<TH1F>(
        "h_recoVtxChi2Prob", "Prob(#chi^{2},ndf);Prob(#chi^{2},ndf);vertices", 40, 0., 1.);
    h_recoVtxSumPt_ =
        RecoVertexFeatures.make<TH1F>("h_recoVtxSumPt", "Sum(p^{trks}_{T});Sum(p^{trks}_{T});vertices", 100, 0., 200.);
  }

  TFileDirectory DA = fs->mkdir("DA");
  //DA.cd();
  hDA = bookVertexHistograms(DA);
  //for(std::map<std::string,TH1*>::const_iterator hist=hDA.begin(); hist!=hDA.end(); hist++){
  //hist->second->SetDirectory(DA);
  // DA.make<TH1F>(hist->second);
  // }

  // initialize the residuals histograms

  const float dxymax_phi = theDetails_.getHigh(PVValHelper::dxy, PVValHelper::phi);
  const float dzmax_phi = theDetails_.getHigh(PVValHelper::dz, PVValHelper::eta);
  const float dxymax_eta = theDetails_.getHigh(PVValHelper::dxy, PVValHelper::phi);
  const float dzmax_eta = theDetails_.getHigh(PVValHelper::dz, PVValHelper::eta);
  //const float d3Dmax_phi = theDetails_.getHigh(PVValHelper::d3D,PVValHelper::phi);
  const float d3Dmax_eta = theDetails_.getHigh(PVValHelper::d3D, PVValHelper::eta);

  ///////////////////////////////////////////////////////////////////
  //
  // Unbiased track-to-vertex residuals
  // The vertex is refit without the probe track
  //
  ///////////////////////////////////////////////////////////////////

  //    _   _            _      _         ___        _    _           _
  //   /_\ | |__ ___ ___| |_  _| |_ ___  | _ \___ __(_)__| |_  _ __ _| |___
  //  / _ \| '_ (_-</ _ \ | || |  _/ -_) |   / -_|_-< / _` | || / _` | (_-<
  // /_/ \_\_.__/__/\___/_|\_,_|\__\___| |_|_\___/__/_\__,_|\_,_\__,_|_/__/
  //

  TFileDirectory AbsTransPhiRes = fs->mkdir("Abs_Transv_Phi_Residuals");
  a_dxyPhiResiduals = bookResidualsHistogram(AbsTransPhiRes, nBins_, PVValHelper::dxy, PVValHelper::phi);
  a_dxPhiResiduals = bookResidualsHistogram(AbsTransPhiRes, nBins_, PVValHelper::dx, PVValHelper::phi);
  a_dyPhiResiduals = bookResidualsHistogram(AbsTransPhiRes, nBins_, PVValHelper::dy, PVValHelper::phi);
  a_IP2DPhiResiduals = bookResidualsHistogram(AbsTransPhiRes, nBins_, PVValHelper::IP2D, PVValHelper::phi);

  TFileDirectory AbsTransEtaRes = fs->mkdir("Abs_Transv_Eta_Residuals");
  a_dxyEtaResiduals = bookResidualsHistogram(AbsTransEtaRes, nBins_, PVValHelper::dxy, PVValHelper::eta);
  a_dxEtaResiduals = bookResidualsHistogram(AbsTransEtaRes, nBins_, PVValHelper::dx, PVValHelper::eta);
  a_dyEtaResiduals = bookResidualsHistogram(AbsTransEtaRes, nBins_, PVValHelper::dy, PVValHelper::eta);
  a_IP2DEtaResiduals = bookResidualsHistogram(AbsTransEtaRes, nBins_, PVValHelper::IP2D, PVValHelper::eta);

  TFileDirectory AbsLongPhiRes = fs->mkdir("Abs_Long_Phi_Residuals");
  a_dzPhiResiduals = bookResidualsHistogram(AbsLongPhiRes, nBins_, PVValHelper::dz, PVValHelper::phi);
  a_reszPhiResiduals = bookResidualsHistogram(AbsLongPhiRes, nBins_, PVValHelper::resz, PVValHelper::phi);

  TFileDirectory AbsLongEtaRes = fs->mkdir("Abs_Long_Eta_Residuals");
  a_dzEtaResiduals = bookResidualsHistogram(AbsLongEtaRes, nBins_, PVValHelper::dz, PVValHelper::eta);
  a_reszEtaResiduals = bookResidualsHistogram(AbsLongEtaRes, nBins_, PVValHelper::resz, PVValHelper::eta);

  TFileDirectory Abs3DPhiRes = fs->mkdir("Abs_3D_Phi_Residuals");
  a_d3DPhiResiduals = bookResidualsHistogram(Abs3DPhiRes, nBins_, PVValHelper::d3D, PVValHelper::phi);
  a_IP3DPhiResiduals = bookResidualsHistogram(Abs3DPhiRes, nBins_, PVValHelper::IP3D, PVValHelper::phi);

  TFileDirectory Abs3DEtaRes = fs->mkdir("Abs_3D_Eta_Residuals");
  a_d3DEtaResiduals = bookResidualsHistogram(Abs3DEtaRes, nBins_, PVValHelper::d3D, PVValHelper::eta);
  a_IP3DEtaResiduals = bookResidualsHistogram(Abs3DEtaRes, nBins_, PVValHelper::IP3D, PVValHelper::eta);

  TFileDirectory NormTransPhiRes = fs->mkdir("Norm_Transv_Phi_Residuals");
  n_dxyPhiResiduals = bookResidualsHistogram(NormTransPhiRes, nBins_, PVValHelper::norm_dxy, PVValHelper::phi, true);
  n_IP2DPhiResiduals = bookResidualsHistogram(NormTransPhiRes, nBins_, PVValHelper::norm_IP2D, PVValHelper::phi, true);

  TFileDirectory NormTransEtaRes = fs->mkdir("Norm_Transv_Eta_Residuals");
  n_dxyEtaResiduals = bookResidualsHistogram(NormTransEtaRes, nBins_, PVValHelper::norm_dxy, PVValHelper::eta, true);
  n_IP2DEtaResiduals = bookResidualsHistogram(NormTransEtaRes, nBins_, PVValHelper::norm_IP2D, PVValHelper::eta, true);

  TFileDirectory NormLongPhiRes = fs->mkdir("Norm_Long_Phi_Residuals");
  n_dzPhiResiduals = bookResidualsHistogram(NormLongPhiRes, nBins_, PVValHelper::norm_dz, PVValHelper::phi, true);
  n_reszPhiResiduals = bookResidualsHistogram(NormLongPhiRes, nBins_, PVValHelper::norm_resz, PVValHelper::phi, true);

  TFileDirectory NormLongEtaRes = fs->mkdir("Norm_Long_Eta_Residuals");
  n_dzEtaResiduals = bookResidualsHistogram(NormLongEtaRes, nBins_, PVValHelper::norm_dz, PVValHelper::eta, true);
  n_reszEtaResiduals = bookResidualsHistogram(NormLongEtaRes, nBins_, PVValHelper::norm_resz, PVValHelper::eta, true);

  TFileDirectory Norm3DPhiRes = fs->mkdir("Norm_3D_Phi_Residuals");
  n_d3DPhiResiduals = bookResidualsHistogram(Norm3DPhiRes, nBins_, PVValHelper::norm_d3D, PVValHelper::phi, true);
  n_IP3DPhiResiduals = bookResidualsHistogram(Norm3DPhiRes, nBins_, PVValHelper::norm_IP3D, PVValHelper::phi, true);

  TFileDirectory Norm3DEtaRes = fs->mkdir("Norm_3D_Eta_Residuals");
  n_d3DEtaResiduals = bookResidualsHistogram(Norm3DEtaRes, nBins_, PVValHelper::norm_d3D, PVValHelper::eta, true);
  n_IP3DEtaResiduals = bookResidualsHistogram(Norm3DEtaRes, nBins_, PVValHelper::norm_IP3D, PVValHelper::eta, true);

  TFileDirectory AbsDoubleDiffRes = fs->mkdir("Abs_DoubleDiffResiduals");
  TFileDirectory NormDoubleDiffRes = fs->mkdir("Norm_DoubleDiffResiduals");

  TFileDirectory AbsL1Map = fs->mkdir("Abs_L1Residuals");
  TFileDirectory NormL1Map = fs->mkdir("Norm_L1Residuals");

  // book residuals vs pT histograms

  TFileDirectory AbsTranspTRes = fs->mkdir("Abs_Transv_pT_Residuals");
  h_dxy_pT_ = bookResidualsHistogram(AbsTranspTRes, nPtBins_, PVValHelper::dxy, PVValHelper::pT);

  TFileDirectory AbsLongpTRes = fs->mkdir("Abs_Long_pT_Residuals");
  h_dz_pT_ = bookResidualsHistogram(AbsLongpTRes, nPtBins_, PVValHelper::dz, PVValHelper::pT);

  TFileDirectory NormTranspTRes = fs->mkdir("Norm_Transv_pT_Residuals");
  h_norm_dxy_pT_ = bookResidualsHistogram(NormTranspTRes, nPtBins_, PVValHelper::norm_dxy, PVValHelper::pT, true);

  TFileDirectory NormLongpTRes = fs->mkdir("Norm_Long_pT_Residuals");
  h_norm_dz_pT_ = bookResidualsHistogram(NormLongpTRes, nPtBins_, PVValHelper::norm_dz, PVValHelper::pT, true);

  // book residuals vs pT histograms in central region (|eta|<1.0)

  TFileDirectory AbsTranspTCentralRes = fs->mkdir("Abs_Transv_pTCentral_Residuals");
  h_dxy_Central_pT_ = bookResidualsHistogram(AbsTranspTCentralRes, nPtBins_, PVValHelper::dxy, PVValHelper::pTCentral);

  TFileDirectory AbsLongpTCentralRes = fs->mkdir("Abs_Long_pTCentral_Residuals");
  h_dz_Central_pT_ = bookResidualsHistogram(AbsLongpTCentralRes, nPtBins_, PVValHelper::dz, PVValHelper::pTCentral);

  TFileDirectory NormTranspTCentralRes = fs->mkdir("Norm_Transv_pTCentral_Residuals");
  h_norm_dxy_Central_pT_ =
      bookResidualsHistogram(NormTranspTCentralRes, nPtBins_, PVValHelper::norm_dxy, PVValHelper::pTCentral, true);

  TFileDirectory NormLongpTCentralRes = fs->mkdir("Norm_Long_pTCentral_Residuals");
  h_norm_dz_Central_pT_ =
      bookResidualsHistogram(NormLongpTCentralRes, nPtBins_, PVValHelper::norm_dz, PVValHelper::pTCentral, true);

  // book residuals vs module number

  TFileDirectory AbsTransModZRes = fs->mkdir("Abs_Transv_modZ_Residuals");
  h_dxy_modZ_ = bookResidualsHistogram(AbsTransModZRes, nModZ_, PVValHelper::dxy, PVValHelper::modZ);

  TFileDirectory AbsLongModZRes = fs->mkdir("Abs_Long_modZ_Residuals");
  h_dz_modZ_ = bookResidualsHistogram(AbsLongModZRes, nModZ_, PVValHelper::dz, PVValHelper::modZ);

  //  _  _                    _ _           _   ___        _    _           _
  // | \| |___ _ _ _ __  __ _| (_)______ __| | | _ \___ __(_)__| |_  _ __ _| |___
  // | .` / _ \ '_| '  \/ _` | | |_ / -_) _` | |   / -_|_-< / _` | || / _` | (_-<
  // |_|\_\___/_| |_|_|_\__,_|_|_/__\___\__,_| |_|_\___/__/_\__,_|\_,_\__,_|_/__/
  //

  TFileDirectory NormTransModZRes = fs->mkdir("Norm_Transv_modZ_Residuals");
  h_norm_dxy_modZ_ = bookResidualsHistogram(NormTransModZRes, nModZ_, PVValHelper::norm_dxy, PVValHelper::modZ, true);

  TFileDirectory NormLongModZRes = fs->mkdir("Norm_Long_modZ_Residuals");
  h_norm_dz_modZ_ = bookResidualsHistogram(NormLongModZRes, nModZ_, PVValHelper::norm_dz, PVValHelper::modZ, true);

  TFileDirectory AbsTransLadderRes = fs->mkdir("Abs_Transv_ladder_Residuals");
  h_dxy_ladder_ = bookResidualsHistogram(AbsTransLadderRes, nLadders_, PVValHelper::dxy, PVValHelper::ladder);

  TFileDirectory AbsTransLadderResOverlap = fs->mkdir("Abs_Transv_ladderOverlap_Residuals");
  h_dxy_ladderOverlap_ =
      bookResidualsHistogram(AbsTransLadderResOverlap, nLadders_, PVValHelper::dxy, PVValHelper::ladder);

  TFileDirectory AbsTransLadderResNoOverlap = fs->mkdir("Abs_Transv_ladderNoOverlap_Residuals");
  h_dxy_ladderNoOverlap_ =
      bookResidualsHistogram(AbsTransLadderResNoOverlap, nLadders_, PVValHelper::dxy, PVValHelper::ladder);

  TFileDirectory AbsLongLadderRes = fs->mkdir("Abs_Long_ladder_Residuals");
  h_dz_ladder_ = bookResidualsHistogram(AbsLongLadderRes, nLadders_, PVValHelper::dz, PVValHelper::ladder);

  TFileDirectory NormTransLadderRes = fs->mkdir("Norm_Transv_ladder_Residuals");
  h_norm_dxy_ladder_ =
      bookResidualsHistogram(NormTransLadderRes, nLadders_, PVValHelper::norm_dxy, PVValHelper::ladder, true);

  TFileDirectory NormLongLadderRes = fs->mkdir("Norm_Long_ladder_Residuals");
  h_norm_dz_ladder_ =
      bookResidualsHistogram(NormLongLadderRes, nLadders_, PVValHelper::norm_dz, PVValHelper::ladder, true);

  // book residuals as function of nLadders and nModules

  for (unsigned int iLadder = 0; iLadder < nLadders_; iLadder++) {
    for (unsigned int iModule = 0; iModule < nModZ_; iModule++) {
      a_dxyL1ResidualsMap[iLadder][iModule] =
          AbsL1Map.make<TH1F>(Form("histo_dxy_ladder%i_module%i", iLadder, iModule),
                              Form("d_{xy} ladder=%i module=%i;d_{xy} [#mum];tracks", iLadder, iModule),
                              theDetails_.histobins,
                              -dzmax_eta,
                              dzmax_eta);

      a_dzL1ResidualsMap[iLadder][iModule] =
          AbsL1Map.make<TH1F>(Form("histo_dz_ladder%i_module%i", iLadder, iModule),
                              Form("d_{z} ladder=%i module=%i;d_{z} [#mum];tracks", iLadder, iModule),
                              theDetails_.histobins,
                              -dzmax_eta,
                              dzmax_eta);

      n_dxyL1ResidualsMap[iLadder][iModule] =
          NormL1Map.make<TH1F>(Form("histo_norm_dxy_ladder%i_module%i", iLadder, iModule),
                               Form("d_{xy} ladder=%i module=%i;d_{xy}/#sigma_{d_{xy}};tracks", iLadder, iModule),
                               theDetails_.histobins,
                               -dzmax_eta / 100,
                               dzmax_eta / 100);

      n_dzL1ResidualsMap[iLadder][iModule] =
          NormL1Map.make<TH1F>(Form("histo_norm_dz_ladder%i_module%i", iLadder, iModule),
                               Form("d_{z} ladder=%i module=%i;d_{z}/#sigma_{d_{z}};tracks", iLadder, iModule),
                               theDetails_.histobins,
                               -dzmax_eta / 100,
                               dzmax_eta / 100);
    }
  }

  // book residuals as function of phi and eta

  for (int i = 0; i < nBins_; ++i) {
    float phiF = theDetails_.trendbins[PVValHelper::phi][i];
    float phiL = theDetails_.trendbins[PVValHelper::phi][i + 1];

    //  ___           _    _     ___  _  __  __   ___        _    _           _
    // |   \ ___ _  _| |__| |___|   \(_)/ _|/ _| | _ \___ __(_)__| |_  _ __ _| |___
    // | |) / _ \ || | '_ \ / -_) |) | |  _|  _| |   / -_|_-< / _` | || / _` | (_-<
    // |___/\___/\_,_|_.__/_\___|___/|_|_| |_|   |_|_\___/__/_\__,_|\_,_\__,_|_/__/

    for (int j = 0; j < nBins_; ++j) {
      float etaF = theDetails_.trendbins[PVValHelper::eta][j];
      float etaL = theDetails_.trendbins[PVValHelper::eta][j + 1];

      a_dxyResidualsMap[i][j] = AbsDoubleDiffRes.make<TH1F>(
          Form("histo_dxy_eta_plot%i_phi_plot%i", i, j),
          Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{xy};tracks", etaF, etaL, phiF, phiL),
          theDetails_.histobins,
          -dzmax_eta,
          dzmax_eta);

      a_dzResidualsMap[i][j] = AbsDoubleDiffRes.make<TH1F>(
          Form("histo_dz_eta_plot%i_phi_plot%i", i, j),
          Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{z};tracks", etaF, etaL, phiF, phiL),
          theDetails_.histobins,
          -dzmax_eta,
          dzmax_eta);

      a_d3DResidualsMap[i][j] = AbsDoubleDiffRes.make<TH1F>(
          Form("histo_d3D_eta_plot%i_phi_plot%i", i, j),
          Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{3D};tracks", etaF, etaL, phiF, phiL),
          theDetails_.histobins,
          0.,
          d3Dmax_eta);

      n_dxyResidualsMap[i][j] = NormDoubleDiffRes.make<TH1F>(
          Form("histo_norm_dxy_eta_plot%i_phi_plot%i", i, j),
          Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{xy}/#sigma_{d_{xy}};tracks",
               etaF,
               etaL,
               phiF,
               phiL),
          theDetails_.histobins,
          -dzmax_eta / 100,
          dzmax_eta / 100);

      n_dzResidualsMap[i][j] = NormDoubleDiffRes.make<TH1F>(
          Form("histo_norm_dz_eta_plot%i_phi_plot%i", i, j),
          Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{z}/#sigma_{d_{z}};tracks",
               etaF,
               etaL,
               phiF,
               phiL),
          theDetails_.histobins,
          -dzmax_eta / 100,
          dzmax_eta / 100);

      n_d3DResidualsMap[i][j] = NormDoubleDiffRes.make<TH1F>(
          Form("histo_norm_d3D_eta_plot%i_phi_plot%i", i, j),
          Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{3D}/#sigma_{d_{3D}};tracks",
               etaF,
               etaL,
               phiF,
               phiL),
          theDetails_.histobins,
          0.,
          d3Dmax_eta);
    }
  }

  // declaration of the directories

  TFileDirectory BiasVsParameter = fs->mkdir("BiasVsParameter");

  a_dxyVsPhi = BiasVsParameter.make<TH2F>("h2_dxy_vs_phi",
                                          "d_{xy} vs track #phi;track #phi [rad];track d_{xy}(PV) [#mum]",
                                          nBins_,
                                          -M_PI,
                                          M_PI,
                                          theDetails_.histobins,
                                          -dxymax_phi,
                                          dxymax_phi);

  a_dzVsPhi = BiasVsParameter.make<TH2F>("h2_dz_vs_phi",
                                         "d_{z} vs track #phi;track #phi [rad];track d_{z}(PV) [#mum]",
                                         nBins_,
                                         -M_PI,
                                         M_PI,
                                         theDetails_.histobins,
                                         -dzmax_phi,
                                         dzmax_phi);

  n_dxyVsPhi = BiasVsParameter.make<TH2F>(
      "h2_n_dxy_vs_phi",
      "d_{xy}/#sigma_{d_{xy}} vs track #phi;track #phi [rad];track d_{xy}(PV)/#sigma_{d_{xy}}",
      nBins_,
      -M_PI,
      M_PI,
      theDetails_.histobins,
      -dxymax_phi / 100.,
      dxymax_phi / 100.);

  n_dzVsPhi =
      BiasVsParameter.make<TH2F>("h2_n_dz_vs_phi",
                                 "d_{z}/#sigma_{d_{z}} vs track #phi;track #phi [rad];track d_{z}(PV)/#sigma_{d_{z}}",
                                 nBins_,
                                 -M_PI,
                                 M_PI,
                                 theDetails_.histobins,
                                 -dzmax_phi / 100.,
                                 dzmax_phi / 100.);

  a_dxyVsEta = BiasVsParameter.make<TH2F>("h2_dxy_vs_eta",
                                          "d_{xy} vs track #eta;track #eta;track d_{xy}(PV) [#mum]",
                                          nBins_,
                                          -etaOfProbe_,
                                          etaOfProbe_,
                                          theDetails_.histobins,
                                          -dxymax_eta,
                                          dzmax_eta);

  a_dzVsEta = BiasVsParameter.make<TH2F>("h2_dz_vs_eta",
                                         "d_{z} vs track #eta;track #eta;track d_{z}(PV) [#mum]",
                                         nBins_,
                                         -etaOfProbe_,
                                         etaOfProbe_,
                                         theDetails_.histobins,
                                         -dzmax_eta,
                                         dzmax_eta);

  n_dxyVsEta =
      BiasVsParameter.make<TH2F>("h2_n_dxy_vs_eta",
                                 "d_{xy}/#sigma_{d_{xy}} vs track #eta;track #eta;track d_{xy}(PV)/#sigma_{d_{xy}}",
                                 nBins_,
                                 -etaOfProbe_,
                                 etaOfProbe_,
                                 theDetails_.histobins,
                                 -dxymax_eta / 100.,
                                 dxymax_eta / 100.);

  n_dzVsEta = BiasVsParameter.make<TH2F>("h2_n_dz_vs_eta",
                                         "d_{z}/#sigma_{d_{z}} vs track #eta;track #eta;track d_{z}(PV)/#sigma_{d_{z}}",
                                         nBins_,
                                         -etaOfProbe_,
                                         etaOfProbe_,
                                         theDetails_.histobins,
                                         -dzmax_eta / 100.,
                                         dzmax_eta / 100.);

  MeanTrendsDir = fs->mkdir("MeanTrends");
  WidthTrendsDir = fs->mkdir("WidthTrends");
  MedianTrendsDir = fs->mkdir("MedianTrends");
  MADTrendsDir = fs->mkdir("MADTrends");

  Mean2DMapsDir = fs->mkdir("MeanMaps");
  Width2DMapsDir = fs->mkdir("WidthMaps");

  double highedge = nBins_ - 0.5;
  double lowedge = -0.5;

  // means and widths from the fit

  a_dxyPhiMeanTrend =
      MeanTrendsDir.make<TH1F>("means_dxy_phi",
                               "#LT d_{xy} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{xy} #GT [#mum]",
                               nBins_,
                               lowedge,
                               highedge);

  a_dxyPhiWidthTrend =
      WidthTrendsDir.make<TH1F>("widths_dxy_phi",
                                "#sigma_{d_{xy}} vs #phi sector;#varphi (sector) [degrees];#sigma_{d_{xy}} [#mum]",
                                nBins_,
                                lowedge,
                                highedge);

  a_dzPhiMeanTrend =
      MeanTrendsDir.make<TH1F>("means_dz_phi",
                               "#LT d_{z} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{z} #GT [#mum]",
                               nBins_,
                               lowedge,
                               highedge);

  a_dzPhiWidthTrend =
      WidthTrendsDir.make<TH1F>("widths_dz_phi",
                                "#sigma_{d_{z}} vs #phi sector;#varphi (sector) [degrees];#sigma_{d_{z}} [#mum]",
                                nBins_,
                                lowedge,
                                highedge);

  a_dxyEtaMeanTrend = MeanTrendsDir.make<TH1F>(
      "means_dxy_eta", "#LT d_{xy} #GT vs #eta sector;#eta (sector);#LT d_{xy} #GT [#mum]", nBins_, lowedge, highedge);

  a_dxyEtaWidthTrend = WidthTrendsDir.make<TH1F>("widths_dxy_eta",
                                                 "#sigma_{d_{xy}} vs #eta sector;#eta (sector);#sigma_{d_{xy}} [#mum]",
                                                 nBins_,
                                                 lowedge,
                                                 highedge);

  a_dzEtaMeanTrend = MeanTrendsDir.make<TH1F>(
      "means_dz_eta", "#LT d_{z} #GT vs #eta sector;#eta (sector);#LT d_{z} #GT [#mum]", nBins_, lowedge, highedge);

  a_dzEtaWidthTrend = WidthTrendsDir.make<TH1F>(
      "widths_dz_eta", "#sigma_{d_{xy}} vs #eta sector;#eta (sector);#sigma_{d_{z}} [#mum]", nBins_, lowedge, highedge);

  n_dxyPhiMeanTrend = MeanTrendsDir.make<TH1F>(
      "norm_means_dxy_phi",
      "#LT d_{xy}/#sigma_{d_{xy}} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{xy}/#sigma_{d_{xy}} #GT",
      nBins_,
      lowedge,
      highedge);

  n_dxyPhiWidthTrend = WidthTrendsDir.make<TH1F>(
      "norm_widths_dxy_phi",
      "width(d_{xy}/#sigma_{d_{xy}}) vs #phi sector;#varphi (sector) [degrees]; width(d_{xy}/#sigma_{d_{xy}})",
      nBins_,
      lowedge,
      highedge);

  n_dzPhiMeanTrend = MeanTrendsDir.make<TH1F>(
      "norm_means_dz_phi",
      "#LT d_{z}/#sigma_{d_{z}} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{z}/#sigma_{d_{z}} #GT",
      nBins_,
      lowedge,
      highedge);

  n_dzPhiWidthTrend = WidthTrendsDir.make<TH1F>(
      "norm_widths_dz_phi",
      "width(d_{z}/#sigma_{d_{z}}) vs #phi sector;#varphi (sector) [degrees];width(d_{z}/#sigma_{d_{z}})",
      nBins_,
      lowedge,
      highedge);

  n_dxyEtaMeanTrend = MeanTrendsDir.make<TH1F>(
      "norm_means_dxy_eta",
      "#LT d_{xy}/#sigma_{d_{xy}} #GT vs #eta sector;#eta (sector);#LT d_{xy}/#sigma_{d_{z}} #GT",
      nBins_,
      lowedge,
      highedge);

  n_dxyEtaWidthTrend = WidthTrendsDir.make<TH1F>(
      "norm_widths_dxy_eta",
      "width(d_{xy}/#sigma_{d_{xy}}) vs #eta sector;#eta (sector);width(d_{xy}/#sigma_{d_{z}})",
      nBins_,
      lowedge,
      highedge);

  n_dzEtaMeanTrend =
      MeanTrendsDir.make<TH1F>("norm_means_dz_eta",
                               "#LT d_{z}/#sigma_{d_{z}} #GT vs #eta sector;#eta (sector);#LT d_{z}/#sigma_{d_{z}} #GT",
                               nBins_,
                               lowedge,
                               highedge);

  n_dzEtaWidthTrend =
      WidthTrendsDir.make<TH1F>("norm_widths_dz_eta",
                                "width(d_{z}/#sigma_{d_{z}}) vs #eta sector;#eta (sector);width(d_{z}/#sigma_{d_{z}})",
                                nBins_,
                                lowedge,
                                highedge);

  // means and widhts vs pT and pTCentral

  a_dxypTMeanTrend = MeanTrendsDir.make<TH1F>("means_dxy_pT",
                                              "#LT d_{xy} #GT vs pT;p_{T} [GeV];#LT d_{xy} #GT [#mum]",
                                              mypT_bins_.size() - 1,
                                              mypT_bins_.data());

  a_dxypTWidthTrend = WidthTrendsDir.make<TH1F>("widths_dxy_pT",
                                                "#sigma_{d_{xy}} vs pT;p_{T} [GeV];#sigma_{d_{xy}} [#mum]",
                                                mypT_bins_.size() - 1,
                                                mypT_bins_.data());

  a_dzpTMeanTrend = MeanTrendsDir.make<TH1F>(
      "means_dz_pT", "#LT d_{z} #GT vs pT;p_{T} [GeV];#LT d_{z} #GT [#mum]", mypT_bins_.size() - 1, mypT_bins_.data());

  a_dzpTWidthTrend = WidthTrendsDir.make<TH1F>("widths_dz_pT",
                                               "#sigma_{d_{z}} vs pT;p_{T} [GeV];#sigma_{d_{z}} [#mum]",
                                               mypT_bins_.size() - 1,
                                               mypT_bins_.data());

  n_dxypTMeanTrend =
      MeanTrendsDir.make<TH1F>("norm_means_dxy_pT",
                               "#LT d_{xy}/#sigma_{d_{xy}} #GT vs pT;p_{T} [GeV];#LT d_{xy}/#sigma_{d_{xy}} #GT",
                               mypT_bins_.size() - 1,
                               mypT_bins_.data());

  n_dxypTWidthTrend =
      WidthTrendsDir.make<TH1F>("norm_widths_dxy_pT",
                                "width(d_{xy}/#sigma_{d_{xy}}) vs pT;p_{T} [GeV]; width(d_{xy}/#sigma_{d_{xy}})",
                                mypT_bins_.size() - 1,
                                mypT_bins_.data());

  n_dzpTMeanTrend =
      MeanTrendsDir.make<TH1F>("norm_means_dz_pT",
                               "#LT d_{z}/#sigma_{d_{z}} #GT vs pT;p_{T} [GeV];#LT d_{z}/#sigma_{d_{z}} #GT",
                               mypT_bins_.size() - 1,
                               mypT_bins_.data());

  n_dzpTWidthTrend =
      WidthTrendsDir.make<TH1F>("norm_widths_dz_pT",
                                "width(d_{z}/#sigma_{d_{z}}) vs pT;p_{T} [GeV];width(d_{z}/#sigma_{d_{z}})",
                                mypT_bins_.size() - 1,
                                mypT_bins_.data());

  a_dxypTCentralMeanTrend =
      MeanTrendsDir.make<TH1F>("means_dxy_pTCentral",
                               "#LT d_{xy} #GT vs p_{T};p_{T}(|#eta|<1.) [GeV];#LT d_{xy} #GT [#mum]",
                               mypT_bins_.size() - 1,
                               mypT_bins_.data());

  a_dxypTCentralWidthTrend =
      WidthTrendsDir.make<TH1F>("widths_dxy_pTCentral",
                                "#sigma_{d_{xy}} vs p_{T};p_{T}(|#eta|<1.) [GeV];#sigma_{d_{xy}} [#mum]",
                                mypT_bins_.size() - 1,
                                mypT_bins_.data());

  a_dzpTCentralMeanTrend =
      MeanTrendsDir.make<TH1F>("means_dz_pTCentral",
                               "#LT d_{z} #GT vs p_{T};p_{T}(|#eta|<1.) [GeV];#LT d_{z} #GT [#mum]",
                               mypT_bins_.size() - 1,
                               mypT_bins_.data());

  a_dzpTCentralWidthTrend =
      WidthTrendsDir.make<TH1F>("widths_dz_pTCentral",
                                "#sigma_{d_{z}} vs p_{T};p_{T}(|#eta|<1.) [GeV];#sigma_{d_{z}} [#mum]",
                                mypT_bins_.size() - 1,
                                mypT_bins_.data());

  n_dxypTCentralMeanTrend = MeanTrendsDir.make<TH1F>(
      "norm_means_dxy_pTCentral",
      "#LT d_{xy}/#sigma_{d_{xy}} #GT vs p_{T};p_{T}(|#eta|<1.) [GeV];#LT d_{xy}/#sigma_{d_{z}} #GT",
      mypT_bins_.size() - 1,
      mypT_bins_.data());

  n_dxypTCentralWidthTrend = WidthTrendsDir.make<TH1F>(
      "norm_widths_dxy_pTCentral",
      "width(d_{xy}/#sigma_{d_{xy}}) vs p_{T};p_{T}(|#eta|<1.) [GeV];width(d_{xy}/#sigma_{d_{z}})",
      mypT_bins_.size() - 1,
      mypT_bins_.data());

  n_dzpTCentralMeanTrend = MeanTrendsDir.make<TH1F>(
      "norm_means_dz_pTCentral",
      "#LT d_{z}/#sigma_{d_{z}} #GT vs p_{T};p_{T}(|#eta|<1.) [GeV];#LT d_{z}/#sigma_{d_{z}} #GT",
      mypT_bins_.size() - 1,
      mypT_bins_.data());

  n_dzpTCentralWidthTrend = WidthTrendsDir.make<TH1F>(
      "norm_widths_dz_pTCentral",
      "width(d_{z}/#sigma_{d_{z}}) vs p_{T};p_{T}(|#eta|<1.) [GeV];width(d_{z}/#sigma_{d_{z}})",
      mypT_bins_.size() - 1,
      mypT_bins_.data());

  // 2D maps

  a_dxyMeanMap = Mean2DMapsDir.make<TH2F>("means_dxy_map",
                                          "#LT d_{xy} #GT map;#eta (sector);#varphi (sector) [degrees]",
                                          nBins_,
                                          lowedge,
                                          highedge,
                                          nBins_,
                                          lowedge,
                                          highedge);

  a_dzMeanMap = Mean2DMapsDir.make<TH2F>("means_dz_map",
                                         "#LT d_{z} #GT map;#eta (sector);#varphi (sector) [degrees]",
                                         nBins_,
                                         lowedge,
                                         highedge,
                                         nBins_,
                                         lowedge,
                                         highedge);

  n_dxyMeanMap = Mean2DMapsDir.make<TH2F>("norm_means_dxy_map",
                                          "#LT d_{xy}/#sigma_{d_{xy}} #GT map;#eta (sector);#varphi (sector) [degrees]",
                                          nBins_,
                                          lowedge,
                                          highedge,
                                          nBins_,
                                          lowedge,
                                          highedge);

  n_dzMeanMap = Mean2DMapsDir.make<TH2F>("norm_means_dz_map",
                                         "#LT d_{z}/#sigma_{d_{z}} #GT map;#eta (sector);#varphi (sector) [degrees]",
                                         nBins_,
                                         lowedge,
                                         highedge,
                                         nBins_,
                                         lowedge,
                                         highedge);

  a_dxyWidthMap = Width2DMapsDir.make<TH2F>("widths_dxy_map",
                                            "#sigma_{d_{xy}} map;#eta (sector);#varphi (sector) [degrees]",
                                            nBins_,
                                            lowedge,
                                            highedge,
                                            nBins_,
                                            lowedge,
                                            highedge);

  a_dzWidthMap = Width2DMapsDir.make<TH2F>("widths_dz_map",
                                           "#sigma_{d_{z}} map;#eta (sector);#varphi (sector) [degrees]",
                                           nBins_,
                                           lowedge,
                                           highedge,
                                           nBins_,
                                           lowedge,
                                           highedge);

  n_dxyWidthMap =
      Width2DMapsDir.make<TH2F>("norm_widths_dxy_map",
                                "width(d_{xy}/#sigma_{d_{xy}}) map;#eta (sector);#varphi (sector) [degrees]",
                                nBins_,
                                lowedge,
                                highedge,
                                nBins_,
                                lowedge,
                                highedge);

  n_dzWidthMap = Width2DMapsDir.make<TH2F>("norm_widths_dz_map",
                                           "width(d_{z}/#sigma_{d_{z}}) map;#eta (sector);#varphi (sector) [degrees]",
                                           nBins_,
                                           lowedge,
                                           highedge,
                                           nBins_,
                                           lowedge,
                                           highedge);

  // medians and MADs

  a_dxyPhiMedianTrend =
      MedianTrendsDir.make<TH1F>("medians_dxy_phi",
                                 "Median of d_{xy} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{xy}) [#mum]",
                                 nBins_,
                                 lowedge,
                                 highedge);

  a_dxyPhiMADTrend = MADTrendsDir.make<TH1F>(
      "MADs_dxy_phi",
      "Median absolute deviation of d_{xy} vs #phi sector;#varphi (sector) [degrees];MAD(d_{xy}) [#mum]",
      nBins_,
      lowedge,
      highedge);

  a_dzPhiMedianTrend =
      MedianTrendsDir.make<TH1F>("medians_dz_phi",
                                 "Median of d_{z} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{z}) [#mum]",
                                 nBins_,
                                 lowedge,
                                 highedge);

  a_dzPhiMADTrend = MADTrendsDir.make<TH1F>(
      "MADs_dz_phi",
      "Median absolute deviation of d_{z} vs #phi sector;#varphi (sector) [degrees];MAD(d_{z}) [#mum]",
      nBins_,
      lowedge,
      highedge);

  a_dxyEtaMedianTrend =
      MedianTrendsDir.make<TH1F>("medians_dxy_eta",
                                 "Median of d_{xy} vs #eta sector;#eta (sector);#mu_{1/2}(d_{xy}) [#mum]",
                                 nBins_,
                                 lowedge,
                                 highedge);

  a_dxyEtaMADTrend =
      MADTrendsDir.make<TH1F>("MADs_dxy_eta",
                              "Median absolute deviation of d_{xy} vs #eta sector;#eta (sector);MAD(d_{xy}) [#mum]",
                              nBins_,
                              lowedge,
                              highedge);

  a_dzEtaMedianTrend =
      MedianTrendsDir.make<TH1F>("medians_dz_eta",
                                 "Median of d_{z} vs #eta sector;#eta (sector);#mu_{1/2}(d_{z}) [#mum]",
                                 nBins_,
                                 lowedge,
                                 highedge);

  a_dzEtaMADTrend =
      MADTrendsDir.make<TH1F>("MADs_dz_eta",
                              "Median absolute deviation of d_{z} vs #eta sector;#eta (sector);MAD(d_{z}) [#mum]",
                              nBins_,
                              lowedge,
                              highedge);

  n_dxyPhiMedianTrend = MedianTrendsDir.make<TH1F>(
      "norm_medians_dxy_phi",
      "Median of d_{xy}/#sigma_{d_{xy}} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{xy}/#sigma_{d_{xy}})",
      nBins_,
      lowedge,
      highedge);

  n_dxyPhiMADTrend = MADTrendsDir.make<TH1F>("norm_MADs_dxy_phi",
                                             "Median absolute deviation of d_{xy}/#sigma_{d_{xy}} vs #phi "
                                             "sector;#varphi (sector) [degrees]; MAD(d_{xy}/#sigma_{d_{xy}})",
                                             nBins_,
                                             lowedge,
                                             highedge);

  n_dzPhiMedianTrend = MedianTrendsDir.make<TH1F>(
      "norm_medians_dz_phi",
      "Median of d_{z}/#sigma_{d_{z}} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{z}/#sigma_{d_{z}})",
      nBins_,
      lowedge,
      highedge);

  n_dzPhiMADTrend = MADTrendsDir.make<TH1F>("norm_MADs_dz_phi",
                                            "Median absolute deviation of d_{z}/#sigma_{d_{z}} vs #phi sector;#varphi "
                                            "(sector) [degrees];MAD(d_{z}/#sigma_{d_{z}})",
                                            nBins_,
                                            lowedge,
                                            highedge);

  n_dxyEtaMedianTrend = MedianTrendsDir.make<TH1F>(
      "norm_medians_dxy_eta",
      "Median of d_{xy}/#sigma_{d_{xy}} vs #eta sector;#eta (sector);#mu_{1/2}(d_{xy}/#sigma_{d_{z}})",
      nBins_,
      lowedge,
      highedge);

  n_dxyEtaMADTrend = MADTrendsDir.make<TH1F>(
      "norm_MADs_dxy_eta",
      "Median absolute deviation of d_{xy}/#sigma_{d_{xy}} vs #eta sector;#eta (sector);MAD(d_{xy}/#sigma_{d_{z}})",
      nBins_,
      lowedge,
      highedge);

  n_dzEtaMedianTrend = MedianTrendsDir.make<TH1F>(
      "norm_medians_dz_eta",
      "Median of d_{z}/#sigma_{d_{z}} vs #eta sector;#eta (sector);#mu_{1/2}(d_{z}/#sigma_{d_{z}})",
      nBins_,
      lowedge,
      highedge);

  n_dzEtaMADTrend = MADTrendsDir.make<TH1F>(
      "norm_MADs_dz_eta",
      "Median absolute deviation of d_{z}/#sigma_{d_{z}} vs #eta sector;#eta (sector);MAD(d_{z}/#sigma_{d_{z}})",
      nBins_,
      lowedge,
      highedge);

  ///////////////////////////////////////////////////////////////////
  //
  // plots of biased residuals
  // The vertex includes the probe track
  //
  ///////////////////////////////////////////////////////////////////

  if (useTracksFromRecoVtx_) {
    TFileDirectory AbsTransPhiBiasRes = fs->mkdir("Abs_Transv_Phi_BiasResiduals");
    a_dxyPhiBiasResiduals = bookResidualsHistogram(AbsTransPhiBiasRes, nBins_, PVValHelper::dxy, PVValHelper::phi);

    TFileDirectory AbsTransEtaBiasRes = fs->mkdir("Abs_Transv_Eta_BiasResiduals");
    a_dxyEtaBiasResiduals = bookResidualsHistogram(AbsTransEtaBiasRes, nBins_, PVValHelper::dxy, PVValHelper::eta);

    TFileDirectory AbsLongPhiBiasRes = fs->mkdir("Abs_Long_Phi_BiasResiduals");
    a_dzPhiBiasResiduals = bookResidualsHistogram(AbsLongPhiBiasRes, nBins_, PVValHelper::dz, PVValHelper::phi);

    TFileDirectory AbsLongEtaBiasRes = fs->mkdir("Abs_Long_Eta_BiasResiduals");
    a_dzEtaBiasResiduals = bookResidualsHistogram(AbsLongEtaBiasRes, nBins_, PVValHelper::dz, PVValHelper::eta);

    TFileDirectory NormTransPhiBiasRes = fs->mkdir("Norm_Transv_Phi_BiasResiduals");
    n_dxyPhiBiasResiduals = bookResidualsHistogram(NormTransPhiBiasRes, nBins_, PVValHelper::dxy, PVValHelper::phi);

    TFileDirectory NormTransEtaBiasRes = fs->mkdir("Norm_Transv_Eta_BiasResiduals");
    n_dxyEtaBiasResiduals = bookResidualsHistogram(NormTransEtaBiasRes, nBins_, PVValHelper::dxy, PVValHelper::eta);

    TFileDirectory NormLongPhiBiasRes = fs->mkdir("Norm_Long_Phi_BiasResiduals");
    n_dzPhiBiasResiduals = bookResidualsHistogram(NormLongPhiBiasRes, nBins_, PVValHelper::dz, PVValHelper::phi);

    TFileDirectory NormLongEtaBiasRes = fs->mkdir("Norm_Long_Eta_BiasResiduals");
    n_dzEtaBiasResiduals = bookResidualsHistogram(NormLongEtaBiasRes, nBins_, PVValHelper::dz, PVValHelper::eta);

    TFileDirectory AbsDoubleDiffBiasRes = fs->mkdir("Abs_DoubleDiffBiasResiduals");
    TFileDirectory NormDoubleDiffBiasRes = fs->mkdir("Norm_DoubleDiffBiasResiduals");

    for (int i = 0; i < nBins_; ++i) {
      float phiF = theDetails_.trendbins[PVValHelper::phi][i];
      float phiL = theDetails_.trendbins[PVValHelper::phi][i + 1];

      for (int j = 0; j < nBins_; ++j) {
        float etaF = theDetails_.trendbins[PVValHelper::eta][j];
        float etaL = theDetails_.trendbins[PVValHelper::eta][j + 1];

        a_dxyBiasResidualsMap[i][j] = AbsDoubleDiffBiasRes.make<TH1F>(
            Form("histo_dxy_eta_plot%i_phi_plot%i", i, j),
            Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{xy} [#mum];tracks", etaF, etaL, phiF, phiL),
            theDetails_.histobins,
            -dzmax_eta,
            dzmax_eta);

        a_dzBiasResidualsMap[i][j] = AbsDoubleDiffBiasRes.make<TH1F>(
            Form("histo_dxy_eta_plot%i_phi_plot%i", i, j),
            Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{z} [#mum];tracks", etaF, etaL, phiF, phiL),
            theDetails_.histobins,
            -dzmax_eta,
            dzmax_eta);

        n_dxyBiasResidualsMap[i][j] = NormDoubleDiffBiasRes.make<TH1F>(
            Form("histo_norm_dxy_eta_plot%i_phi_plot%i", i, j),
            Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{xy}/#sigma_{d_{xy}};tracks",
                 etaF,
                 etaL,
                 phiF,
                 phiL),
            theDetails_.histobins,
            -dzmax_eta / 100,
            dzmax_eta / 100);

        n_dzBiasResidualsMap[i][j] = NormDoubleDiffBiasRes.make<TH1F>(
            Form("histo_norm_dxy_eta_plot%i_phi_plot%i", i, j),
            Form("%.2f<#eta_{tk}<%.2f %.2f#circ<#varphi_{tk}<%.2f#circ;d_{z}/#sigma_{d_{z}};tracks",
                 etaF,
                 etaL,
                 phiF,
                 phiL),
            theDetails_.histobins,
            -dzmax_eta / 100,
            dzmax_eta / 100);
      }
    }

    // declaration of the directories

    TFileDirectory MeanBiasTrendsDir = fs->mkdir("MeanBiasTrends");
    TFileDirectory WidthBiasTrendsDir = fs->mkdir("WidthBiasTrends");
    TFileDirectory MedianBiasTrendsDir = fs->mkdir("MedianBiasTrends");
    TFileDirectory MADBiasTrendsDir = fs->mkdir("MADBiasTrends");

    TFileDirectory Mean2DBiasMapsDir = fs->mkdir("MeanBiasMaps");
    TFileDirectory Width2DBiasMapsDir = fs->mkdir("WidthBiasMaps");

    // means and widths from the fit

    a_dxyPhiMeanBiasTrend =
        MeanBiasTrendsDir.make<TH1F>("means_dxy_phi",
                                     "#LT d_{xy} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{xy} #GT [#mum]",
                                     nBins_,
                                     lowedge,
                                     highedge);

    a_dxyPhiWidthBiasTrend = WidthBiasTrendsDir.make<TH1F>(
        "widths_dxy_phi",
        "#sigma_{d_{xy}} vs #phi sector;#varphi (sector) [degrees];#sigma_{d_{xy}} [#mum]",
        nBins_,
        lowedge,
        highedge);

    a_dzPhiMeanBiasTrend =
        MeanBiasTrendsDir.make<TH1F>("means_dz_phi",
                                     "#LT d_{z} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{z} #GT [#mum]",
                                     nBins_,
                                     lowedge,
                                     highedge);

    a_dzPhiWidthBiasTrend =
        WidthBiasTrendsDir.make<TH1F>("widths_dz_phi",
                                      "#sigma_{d_{z}} vs #phi sector;#varphi (sector) [degrees];#sigma_{d_{z}} [#mum]",
                                      nBins_,
                                      lowedge,
                                      highedge);

    a_dxyEtaMeanBiasTrend = MeanBiasTrendsDir.make<TH1F>(
        "means_dxy_eta", "#LT d_{xy} #GT vs #eta sector;#eta (sector);#LT d_{xy} #GT [#mum]", nBins_, lowedge, highedge);

    a_dxyEtaWidthBiasTrend =
        WidthBiasTrendsDir.make<TH1F>("widths_dxy_eta",
                                      "#sigma_{d_{xy}} vs #eta sector;#eta (sector);#sigma_{d_{xy}} [#mum]",
                                      nBins_,
                                      lowedge,
                                      highedge);

    a_dzEtaMeanBiasTrend = MeanBiasTrendsDir.make<TH1F>(
        "means_dz_eta", "#LT d_{z} #GT vs #eta sector;#eta (sector);#LT d_{z} #GT [#mum]", nBins_, lowedge, highedge);

    a_dzEtaWidthBiasTrend =
        WidthBiasTrendsDir.make<TH1F>("widths_dz_eta",
                                      "#sigma_{d_{xy}} vs #eta sector;#eta (sector);#sigma_{d_{z}} [#mum]",
                                      nBins_,
                                      lowedge,
                                      highedge);

    n_dxyPhiMeanBiasTrend = MeanBiasTrendsDir.make<TH1F>(
        "norm_means_dxy_phi",
        "#LT d_{xy}/#sigma_{d_{xy}} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{xy}/#sigma_{d_{xy}} #GT",
        nBins_,
        lowedge,
        highedge);

    n_dxyPhiWidthBiasTrend = WidthBiasTrendsDir.make<TH1F>(
        "norm_widths_dxy_phi",
        "width(d_{xy}/#sigma_{d_{xy}}) vs #phi sector;#varphi (sector) [degrees]; width(d_{xy}/#sigma_{d_{xy}})",
        nBins_,
        lowedge,
        highedge);

    n_dzPhiMeanBiasTrend = MeanBiasTrendsDir.make<TH1F>(
        "norm_means_dz_phi",
        "#LT d_{z}/#sigma_{d_{z}} #GT vs #phi sector;#varphi (sector) [degrees];#LT d_{z}/#sigma_{d_{z}} #GT",
        nBins_,
        lowedge,
        highedge);

    n_dzPhiWidthBiasTrend = WidthBiasTrendsDir.make<TH1F>(
        "norm_widths_dz_phi",
        "width(d_{z}/#sigma_{d_{z}}) vs #phi sector;#varphi (sector) [degrees];width(d_{z}/#sigma_{d_{z}})",
        nBins_,
        lowedge,
        highedge);

    n_dxyEtaMeanBiasTrend = MeanBiasTrendsDir.make<TH1F>(
        "norm_means_dxy_eta",
        "#LT d_{xy}/#sigma_{d_{xy}} #GT vs #eta sector;#eta (sector);#LT d_{xy}/#sigma_{d_{z}} #GT",
        nBins_,
        lowedge,
        highedge);

    n_dxyEtaWidthBiasTrend = WidthBiasTrendsDir.make<TH1F>(
        "norm_widths_dxy_eta",
        "width(d_{xy}/#sigma_{d_{xy}}) vs #eta sector;#eta (sector);width(d_{xy}/#sigma_{d_{z}})",
        nBins_,
        lowedge,
        highedge);

    n_dzEtaMeanBiasTrend = MeanBiasTrendsDir.make<TH1F>(
        "norm_means_dz_eta",
        "#LT d_{z}/#sigma_{d_{z}} #GT vs #eta sector;#eta (sector);#LT d_{z}/#sigma_{d_{z}} #GT",
        nBins_,
        lowedge,
        highedge);

    n_dzEtaWidthBiasTrend = WidthBiasTrendsDir.make<TH1F>(
        "norm_widths_dz_eta",
        "width(d_{z}/#sigma_{d_{z}}) vs #eta sector;#eta (sector);width(d_{z}/#sigma_{d_{z}})",
        nBins_,
        lowedge,
        highedge);

    // 2D maps

    a_dxyMeanBiasMap = Mean2DBiasMapsDir.make<TH2F>("means_dxy_map",
                                                    "#LT d_{xy} #GT map;#eta (sector);#varphi (sector) [degrees]",
                                                    nBins_,
                                                    lowedge,
                                                    highedge,
                                                    nBins_,
                                                    lowedge,
                                                    highedge);

    a_dzMeanBiasMap = Mean2DBiasMapsDir.make<TH2F>("means_dz_map",
                                                   "#LT d_{z} #GT map;#eta (sector);#varphi (sector) [degrees]",
                                                   nBins_,
                                                   lowedge,
                                                   highedge,
                                                   nBins_,
                                                   lowedge,
                                                   highedge);

    n_dxyMeanBiasMap =
        Mean2DBiasMapsDir.make<TH2F>("norm_means_dxy_map",
                                     "#LT d_{xy}/#sigma_{d_{xy}} #GT map;#eta (sector);#varphi (sector) [degrees]",
                                     nBins_,
                                     lowedge,
                                     highedge,
                                     nBins_,
                                     lowedge,
                                     highedge);

    n_dzMeanBiasMap =
        Mean2DBiasMapsDir.make<TH2F>("norm_means_dz_map",
                                     "#LT d_{z}/#sigma_{d_{z}} #GT map;#eta (sector);#varphi (sector) [degrees]",
                                     nBins_,
                                     lowedge,
                                     highedge,
                                     nBins_,
                                     lowedge,
                                     highedge);

    a_dxyWidthBiasMap = Width2DBiasMapsDir.make<TH2F>("widths_dxy_map",
                                                      "#sigma_{d_{xy}} map;#eta (sector);#varphi (sector) [degrees]",
                                                      nBins_,
                                                      lowedge,
                                                      highedge,
                                                      nBins_,
                                                      lowedge,
                                                      highedge);

    a_dzWidthBiasMap = Width2DBiasMapsDir.make<TH2F>("widths_dz_map",
                                                     "#sigma_{d_{z}} map;#eta (sector);#varphi (sector) [degrees]",
                                                     nBins_,
                                                     lowedge,
                                                     highedge,
                                                     nBins_,
                                                     lowedge,
                                                     highedge);

    n_dxyWidthBiasMap =
        Width2DBiasMapsDir.make<TH2F>("norm_widths_dxy_map",
                                      "width(d_{xy}/#sigma_{d_{xy}}) map;#eta (sector);#varphi (sector) [degrees]",
                                      nBins_,
                                      lowedge,
                                      highedge,
                                      nBins_,
                                      lowedge,
                                      highedge);

    n_dzWidthBiasMap =
        Width2DBiasMapsDir.make<TH2F>("norm_widths_dz_map",
                                      "width(d_{z}/#sigma_{d_{z}}) map;#eta (sector);#varphi (sector) [degrees]",
                                      nBins_,
                                      lowedge,
                                      highedge,
                                      nBins_,
                                      lowedge,
                                      highedge);

    // medians and MADs

    a_dxyPhiMedianBiasTrend = MedianBiasTrendsDir.make<TH1F>(
        "medians_dxy_phi",
        "Median of d_{xy} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{xy}) [#mum]",
        nBins_,
        lowedge,
        highedge);

    a_dxyPhiMADBiasTrend = MADBiasTrendsDir.make<TH1F>(
        "MADs_dxy_phi",
        "Median absolute deviation of d_{xy} vs #phi sector;#varphi (sector) [degrees];MAD(d_{xy}) [#mum]",
        nBins_,
        lowedge,
        highedge);

    a_dzPhiMedianBiasTrend = MedianBiasTrendsDir.make<TH1F>(
        "medians_dz_phi",
        "Median of d_{z} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{z}) [#mum]",
        nBins_,
        lowedge,
        highedge);

    a_dzPhiMADBiasTrend = MADBiasTrendsDir.make<TH1F>(
        "MADs_dz_phi",
        "Median absolute deviation of d_{z} vs #phi sector;#varphi (sector) [degrees];MAD(d_{z}) [#mum]",
        nBins_,
        lowedge,
        highedge);

    a_dxyEtaMedianBiasTrend =
        MedianBiasTrendsDir.make<TH1F>("medians_dxy_eta",
                                       "Median of d_{xy} vs #eta sector;#eta (sector);#mu_{1/2}(d_{xy}) [#mum]",
                                       nBins_,
                                       lowedge,
                                       highedge);

    a_dxyEtaMADBiasTrend = MADBiasTrendsDir.make<TH1F>(
        "MADs_dxy_eta",
        "Median absolute deviation of d_{xy} vs #eta sector;#eta (sector);MAD(d_{xy}) [#mum]",
        nBins_,
        lowedge,
        highedge);

    a_dzEtaMedianBiasTrend =
        MedianBiasTrendsDir.make<TH1F>("medians_dz_eta",
                                       "Median of d_{z} vs #eta sector;#eta (sector);#mu_{1/2}(d_{z}) [#mum]",
                                       nBins_,
                                       lowedge,
                                       highedge);

    a_dzEtaMADBiasTrend =
        MADBiasTrendsDir.make<TH1F>("MADs_dz_eta",
                                    "Median absolute deviation of d_{z} vs #eta sector;#eta (sector);MAD(d_{z}) [#mum]",
                                    nBins_,
                                    lowedge,
                                    highedge);

    n_dxyPhiMedianBiasTrend = MedianBiasTrendsDir.make<TH1F>(
        "norm_medians_dxy_phi",
        "Median of d_{xy}/#sigma_{d_{xy}} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{xy}/#sigma_{d_{xy}})",
        nBins_,
        lowedge,
        highedge);

    n_dxyPhiMADBiasTrend = MADBiasTrendsDir.make<TH1F>("norm_MADs_dxy_phi",
                                                       "Median absolute deviation of d_{xy}/#sigma_{d_{xy}} vs #phi "
                                                       "sector;#varphi (sector) [degrees]; MAD(d_{xy}/#sigma_{d_{xy}})",
                                                       nBins_,
                                                       lowedge,
                                                       highedge);

    n_dzPhiMedianBiasTrend = MedianBiasTrendsDir.make<TH1F>(
        "norm_medians_dz_phi",
        "Median of d_{z}/#sigma_{d_{z}} vs #phi sector;#varphi (sector) [degrees];#mu_{1/2}(d_{z}/#sigma_{d_{z}})",
        nBins_,
        lowedge,
        highedge);

    n_dzPhiMADBiasTrend = MADBiasTrendsDir.make<TH1F>("norm_MADs_dz_phi",
                                                      "Median absolute deviation of d_{z}/#sigma_{d_{z}} vs #phi "
                                                      "sector;#varphi (sector) [degrees];MAD(d_{z}/#sigma_{d_{z}})",
                                                      nBins_,
                                                      lowedge,
                                                      highedge);

    n_dxyEtaMedianBiasTrend = MedianBiasTrendsDir.make<TH1F>(
        "norm_medians_dxy_eta",
        "Median of d_{xy}/#sigma_{d_{xy}} vs #eta sector;#eta (sector);#mu_{1/2}(d_{xy}/#sigma_{d_{z}})",
        nBins_,
        lowedge,
        highedge);

    n_dxyEtaMADBiasTrend = MADBiasTrendsDir.make<TH1F>(
        "norm_MADs_dxy_eta",
        "Median absolute deviation of d_{xy}/#sigma_{d_{xy}} vs #eta sector;#eta (sector);MAD(d_{xy}/#sigma_{d_{z}})",
        nBins_,
        lowedge,
        highedge);

    n_dzEtaMedianBiasTrend = MedianBiasTrendsDir.make<TH1F>(
        "norm_medians_dz_eta",
        "Median of d_{z}/#sigma_{d_{z}} vs #eta sector;#eta (sector);#mu_{1/2}(d_{z}/#sigma_{d_{z}})",
        nBins_,
        lowedge,
        highedge);

    n_dzEtaMADBiasTrend = MADBiasTrendsDir.make<TH1F>(
        "norm_MADs_dz_eta",
        "Median absolute deviation of d_{z}/#sigma_{d_{z}} vs #eta sector;#eta (sector);MAD(d_{z}/#sigma_{d_{z}})",
        nBins_,
        lowedge,
        highedge);
  }
}
// ------------ method called once each job just after ending the event loop  ------------
void PrimaryVertexValidation::endJob() {
  // shring the histograms to fit
  h_probeL1Ladder_->GetXaxis()->SetRangeUser(-1.5, nLadders_ + 0.5);
  h_probeL1Module_->GetXaxis()->SetRangeUser(-1.5, nModZ_ + 0.5);
  h2_probeLayer1Map_->GetXaxis()->SetRangeUser(0.5, nModZ_ + 0.5);
  h2_probeLayer1Map_->GetYaxis()->SetRangeUser(0.5, nLadders_ + 0.5);
  h2_probePassingLayer1Map_->GetXaxis()->SetRangeUser(0.5, nModZ_ + 0.5);
  h2_probePassingLayer1Map_->GetYaxis()->SetRangeUser(0.5, nLadders_ + 0.5);

  TFileDirectory RunFeatures = fs->mkdir("RunFeatures");
  h_runStartTimes = RunFeatures.make<TH1I>(
      "runStartTimes", "run start times", runNumbersTimesLog_.size(), 0, runNumbersTimesLog_.size());
  h_runEndTimes =
      RunFeatures.make<TH1I>("runEndTimes", "run end times", runNumbersTimesLog_.size(), 0, runNumbersTimesLog_.size());

  unsigned int count = 1;
  for (const auto& run : runNumbersTimesLog_) {
    // strip down the microseconds
    h_runStartTimes->SetBinContent(count, run.second.first / 10e6);
    h_runStartTimes->GetXaxis()->SetBinLabel(count, (std::to_string(run.first)).c_str());

    h_runEndTimes->SetBinContent(count, run.second.second / 10e6);
    h_runEndTimes->GetXaxis()->SetBinLabel(count, (std::to_string(run.first)).c_str());

    count++;
  }

  edm::LogInfo("PrimaryVertexValidation") << "######################################\n"
                                          << "# PrimaryVertexValidation::endJob()\n"
                                          << "# Number of analyzed events: " << Nevt_ << "\n"
                                          << "######################################";

  // means and widhts vs ladder and module number

  a_dxymodZMeanTrend = MeanTrendsDir.make<TH1F>(
      "means_dxy_modZ", "#LT d_{xy} #GT vs modZ;module number (Z);#LT d_{xy} #GT [#mum]", nModZ_, 0., nModZ_);

  a_dxymodZWidthTrend = WidthTrendsDir.make<TH1F>(
      "widths_dxy_modZ", "#sigma_{d_{xy}} vs modZ;module number (Z);#sigma_{d_{xy}} [#mum]", nModZ_, 0., nModZ_);

  a_dzmodZMeanTrend = MeanTrendsDir.make<TH1F>(
      "means_dz_modZ", "#LT d_{z} #GT vs modZ;module number (Z);#LT d_{z} #GT [#mum]", nModZ_, 0., nModZ_);

  a_dzmodZWidthTrend = WidthTrendsDir.make<TH1F>(
      "widths_dz_modZ", "#sigma_{d_{z}} vs modZ;module number (Z);#sigma_{d_{z}} [#mum]", nModZ_, 0., nModZ_);

  a_dxyladderMeanTrend = MeanTrendsDir.make<TH1F>("means_dxy_ladder",
                                                  "#LT d_{xy} #GT vs ladder;ladder number (#phi);#LT d_{xy} #GT [#mum]",
                                                  nLadders_,
                                                  0.,
                                                  nLadders_);

  a_dxyladderWidthTrend =
      WidthTrendsDir.make<TH1F>("widths_dxy_ladder",
                                "#sigma_{d_{xy}} vs ladder;ladder number (#phi);#sigma_{d_{xy}} [#mum]",
                                nLadders_,
                                0.,
                                nLadders_);

  a_dzladderMeanTrend = MeanTrendsDir.make<TH1F>(
      "means_dz_ladder", "#LT d_{z} #GT vs ladder;ladder number (#phi);#LT d_{z} #GT [#mum]", nLadders_, 0., nLadders_);

  a_dzladderWidthTrend =
      WidthTrendsDir.make<TH1F>("widths_dz_ladder",
                                "#sigma_{d_{z}} vs ladder;ladder number (#phi);#sigma_{d_{z}} [#mum]",
                                nLadders_,
                                0.,
                                nLadders_);

  n_dxymodZMeanTrend = MeanTrendsDir.make<TH1F>(
      "norm_means_dxy_modZ",
      "#LT d_{xy}/#sigma_{d_{xy}} #GT vs modZ;module number (Z);#LT d_{xy}/#sigma_{d_{xy}} #GT",
      nModZ_,
      0.,
      nModZ_);

  n_dxymodZWidthTrend = WidthTrendsDir.make<TH1F>(
      "norm_widths_dxy_modZ",
      "width(d_{xy}/#sigma_{d_{xy}}) vs modZ;module number (Z); width(d_{xy}/#sigma_{d_{xy}})",
      nModZ_,
      0.,
      nModZ_);

  n_dzmodZMeanTrend =
      MeanTrendsDir.make<TH1F>("norm_means_dz_modZ",
                               "#LT d_{z}/#sigma_{d_{z}} #GT vs modZ;module number (Z);#LT d_{z}/#sigma_{d_{z}} #GT",
                               nModZ_,
                               0.,
                               nModZ_);

  n_dzmodZWidthTrend =
      WidthTrendsDir.make<TH1F>("norm_widths_dz_modZ",
                                "width(d_{z}/#sigma_{d_{z}}) vs pT;module number (Z);width(d_{z}/#sigma_{d_{z}})",
                                nModZ_,
                                0.,
                                nModZ_);

  n_dxyladderMeanTrend = MeanTrendsDir.make<TH1F>(
      "norm_means_dxy_ladder",
      "#LT d_{xy}/#sigma_{d_{xy}} #GT vs ladder;ladder number (#phi);#LT d_{xy}/#sigma_{d_{z}} #GT",
      nLadders_,
      0.,
      nLadders_);

  n_dxyladderWidthTrend = WidthTrendsDir.make<TH1F>(
      "norm_widths_dxy_ladder",
      "width(d_{xy}/#sigma_{d_{xy}}) vs ladder;ladder number (#phi);width(d_{xy}/#sigma_{d_{z}})",
      nLadders_,
      0.,
      nLadders_);

  n_dzladderMeanTrend = MeanTrendsDir.make<TH1F>(
      "norm_means_dz_ladder",
      "#LT d_{z}/#sigma_{d_{z}} #GT vs ladder;ladder number (#phi);#LT d_{z}/#sigma_{d_{z}} #GT",
      nLadders_,
      0.,
      nLadders_);

  n_dzladderWidthTrend = WidthTrendsDir.make<TH1F>(
      "norm_widths_dz_ladder",
      "width(d_{z}/#sigma_{d_{z}}) vs ladder;ladder number (#phi);width(d_{z}/#sigma_{d_{z}})",
      nLadders_,
      0.,
      nLadders_);

  // 2D maps of residuals in bins of L1 modules

  a_dxyL1MeanMap = Mean2DMapsDir.make<TH2F>("means_dxy_l1map",
                                            "#LT d_{xy} #GT map;module number [z];ladder number [#varphi]",
                                            nModZ_,
                                            0.,
                                            nModZ_,
                                            nLadders_,
                                            0.,
                                            nLadders_);

  a_dzL1MeanMap = Mean2DMapsDir.make<TH2F>("means_dz_l1map",
                                           "#LT d_{z} #GT map;module number [z];ladder number [#varphi]",
                                           nModZ_,
                                           0.,
                                           nModZ_,
                                           nLadders_,
                                           0.,
                                           nLadders_);

  n_dxyL1MeanMap =
      Mean2DMapsDir.make<TH2F>("norm_means_dxy_l1map",
                               "#LT d_{xy}/#sigma_{d_{xy}} #GT map;module number [z];ladder number [#varphi]",
                               nModZ_,
                               0.,
                               nModZ_,
                               nLadders_,
                               0.,
                               nLadders_);

  n_dzL1MeanMap = Mean2DMapsDir.make<TH2F>("norm_means_dz_l1map",
                                           "#LT d_{z}/#sigma_{d_{z}} #GT map;module number [z];ladder number [#varphi]",
                                           nModZ_,
                                           0.,
                                           nModZ_,
                                           nLadders_,
                                           0.,
                                           nLadders_);

  a_dxyL1WidthMap = Width2DMapsDir.make<TH2F>("widths_dxy_l1map",
                                              "#sigma_{d_{xy}} map;module number [z];ladder number [#varphi]",
                                              nModZ_,
                                              0.,
                                              nModZ_,
                                              nLadders_,
                                              0.,
                                              nLadders_);

  a_dzL1WidthMap = Width2DMapsDir.make<TH2F>("widths_dz_l1map",
                                             "#sigma_{d_{z}} map;module number [z];ladder number [#varphi]",
                                             nModZ_,
                                             0.,
                                             nModZ_,
                                             nLadders_,
                                             0.,
                                             nLadders_);

  n_dxyL1WidthMap =
      Width2DMapsDir.make<TH2F>("norm_widths_dxy_l1map",
                                "width(d_{xy}/#sigma_{d_{xy}}) map;module number [z];ladder number [#varphi]",
                                nModZ_,
                                0.,
                                nModZ_,
                                nLadders_,
                                0.,
                                nLadders_);

  n_dzL1WidthMap =
      Width2DMapsDir.make<TH2F>("norm_widths_dz_l1map",
                                "width(d_{z}/#sigma_{d_{z}}) map;module number [z];ladder number [#varphi]",
                                nModZ_,
                                0.,
                                nModZ_,
                                nLadders_,
                                0.,
                                nLadders_);

  if (useTracksFromRecoVtx_) {
    fillTrendPlotByIndex(a_dxyPhiMeanBiasTrend, a_dxyPhiBiasResiduals, PVValHelper::MEAN, PVValHelper::phi);
    fillTrendPlotByIndex(a_dxyPhiWidthBiasTrend, a_dxyPhiBiasResiduals, PVValHelper::WIDTH, PVValHelper::phi);
    fillTrendPlotByIndex(a_dzPhiMeanBiasTrend, a_dzPhiBiasResiduals, PVValHelper::MEAN, PVValHelper::phi);
    fillTrendPlotByIndex(a_dzPhiWidthBiasTrend, a_dzPhiBiasResiduals, PVValHelper::WIDTH, PVValHelper::phi);

    fillTrendPlotByIndex(a_dxyEtaMeanBiasTrend, a_dxyEtaBiasResiduals, PVValHelper::MEAN, PVValHelper::eta);
    fillTrendPlotByIndex(a_dxyEtaWidthBiasTrend, a_dxyEtaBiasResiduals, PVValHelper::WIDTH, PVValHelper::eta);
    fillTrendPlotByIndex(a_dzEtaMeanBiasTrend, a_dzEtaBiasResiduals, PVValHelper::MEAN, PVValHelper::eta);
    fillTrendPlotByIndex(a_dzEtaWidthBiasTrend, a_dzEtaBiasResiduals, PVValHelper::WIDTH, PVValHelper::eta);

    fillTrendPlotByIndex(n_dxyPhiMeanBiasTrend, n_dxyPhiBiasResiduals, PVValHelper::MEAN, PVValHelper::phi);
    fillTrendPlotByIndex(n_dxyPhiWidthBiasTrend, n_dxyPhiBiasResiduals, PVValHelper::WIDTH, PVValHelper::phi);
    fillTrendPlotByIndex(n_dzPhiMeanBiasTrend, n_dzPhiBiasResiduals, PVValHelper::MEAN, PVValHelper::phi);
    fillTrendPlotByIndex(n_dzPhiWidthBiasTrend, n_dzPhiBiasResiduals, PVValHelper::WIDTH, PVValHelper::phi);

    fillTrendPlotByIndex(n_dxyEtaMeanBiasTrend, n_dxyEtaBiasResiduals, PVValHelper::MEAN, PVValHelper::eta);
    fillTrendPlotByIndex(n_dxyEtaWidthBiasTrend, n_dxyEtaBiasResiduals, PVValHelper::WIDTH, PVValHelper::eta);
    fillTrendPlotByIndex(n_dzEtaMeanBiasTrend, n_dzEtaBiasResiduals, PVValHelper::MEAN, PVValHelper::eta);
    fillTrendPlotByIndex(n_dzEtaWidthBiasTrend, n_dzEtaBiasResiduals, PVValHelper::WIDTH, PVValHelper::eta);

    // medians and MADs

    fillTrendPlotByIndex(a_dxyPhiMedianBiasTrend, a_dxyPhiBiasResiduals, PVValHelper::MEDIAN, PVValHelper::phi);
    fillTrendPlotByIndex(a_dxyPhiMADBiasTrend, a_dxyPhiBiasResiduals, PVValHelper::MAD, PVValHelper::phi);
    fillTrendPlotByIndex(a_dzPhiMedianBiasTrend, a_dzPhiBiasResiduals, PVValHelper::MEDIAN, PVValHelper::phi);
    fillTrendPlotByIndex(a_dzPhiMADBiasTrend, a_dzPhiBiasResiduals, PVValHelper::MAD, PVValHelper::phi);

    fillTrendPlotByIndex(a_dxyEtaMedianBiasTrend, a_dxyEtaBiasResiduals, PVValHelper::MEDIAN, PVValHelper::eta);
    fillTrendPlotByIndex(a_dxyEtaMADBiasTrend, a_dxyEtaBiasResiduals, PVValHelper::MAD, PVValHelper::eta);
    fillTrendPlotByIndex(a_dzEtaMedianBiasTrend, a_dzEtaBiasResiduals, PVValHelper::MEDIAN, PVValHelper::eta);
    fillTrendPlotByIndex(a_dzEtaMADBiasTrend, a_dzEtaBiasResiduals, PVValHelper::MAD, PVValHelper::eta);

    fillTrendPlotByIndex(n_dxyPhiMedianBiasTrend, n_dxyPhiBiasResiduals, PVValHelper::MEDIAN, PVValHelper::phi);
    fillTrendPlotByIndex(n_dxyPhiMADBiasTrend, n_dxyPhiBiasResiduals, PVValHelper::MAD, PVValHelper::phi);
    fillTrendPlotByIndex(n_dzPhiMedianBiasTrend, n_dzPhiBiasResiduals, PVValHelper::MEDIAN, PVValHelper::phi);
    fillTrendPlotByIndex(n_dzPhiMADBiasTrend, n_dzPhiBiasResiduals, PVValHelper::MAD, PVValHelper::phi);

    fillTrendPlotByIndex(n_dxyEtaMedianBiasTrend, n_dxyEtaBiasResiduals, PVValHelper::MEDIAN, PVValHelper::eta);
    fillTrendPlotByIndex(n_dxyEtaMADBiasTrend, n_dxyEtaBiasResiduals, PVValHelper::MAD, PVValHelper::eta);
    fillTrendPlotByIndex(n_dzEtaMedianBiasTrend, n_dzEtaBiasResiduals, PVValHelper::MEDIAN, PVValHelper::eta);
    fillTrendPlotByIndex(n_dzEtaMADBiasTrend, n_dzEtaBiasResiduals, PVValHelper::MAD, PVValHelper::eta);

    // 2d Maps

    fillMap(a_dxyMeanBiasMap, a_dxyBiasResidualsMap, PVValHelper::MEAN, nBins_, nBins_);
    fillMap(a_dxyWidthBiasMap, a_dxyBiasResidualsMap, PVValHelper::WIDTH, nBins_, nBins_);
    fillMap(a_dzMeanBiasMap, a_dzBiasResidualsMap, PVValHelper::MEAN, nBins_, nBins_);
    fillMap(a_dzWidthBiasMap, a_dzBiasResidualsMap, PVValHelper::WIDTH, nBins_, nBins_);

    fillMap(n_dxyMeanBiasMap, n_dxyBiasResidualsMap, PVValHelper::MEAN, nBins_, nBins_);
    fillMap(n_dxyWidthBiasMap, n_dxyBiasResidualsMap, PVValHelper::WIDTH, nBins_, nBins_);
    fillMap(n_dzMeanBiasMap, n_dzBiasResidualsMap, PVValHelper::MEAN, nBins_, nBins_);
    fillMap(n_dzWidthBiasMap, n_dzBiasResidualsMap, PVValHelper::WIDTH, nBins_, nBins_);
  }

  // do profiles

  fillTrendPlotByIndex(a_dxyPhiMeanTrend, a_dxyPhiResiduals, PVValHelper::MEAN, PVValHelper::phi);
  fillTrendPlotByIndex(a_dxyPhiWidthTrend, a_dxyPhiResiduals, PVValHelper::WIDTH, PVValHelper::phi);
  fillTrendPlotByIndex(a_dzPhiMeanTrend, a_dzPhiResiduals, PVValHelper::MEAN, PVValHelper::phi);
  fillTrendPlotByIndex(a_dzPhiWidthTrend, a_dzPhiResiduals, PVValHelper::WIDTH, PVValHelper::phi);

  fillTrendPlotByIndex(a_dxyEtaMeanTrend, a_dxyEtaResiduals, PVValHelper::MEAN, PVValHelper::eta);
  fillTrendPlotByIndex(a_dxyEtaWidthTrend, a_dxyEtaResiduals, PVValHelper::WIDTH, PVValHelper::eta);
  fillTrendPlotByIndex(a_dzEtaMeanTrend, a_dzEtaResiduals, PVValHelper::MEAN, PVValHelper::eta);
  fillTrendPlotByIndex(a_dzEtaWidthTrend, a_dzEtaResiduals, PVValHelper::WIDTH, PVValHelper::eta);

  fillTrendPlotByIndex(n_dxyPhiMeanTrend, n_dxyPhiResiduals, PVValHelper::MEAN, PVValHelper::phi);
  fillTrendPlotByIndex(n_dxyPhiWidthTrend, n_dxyPhiResiduals, PVValHelper::WIDTH, PVValHelper::phi);
  fillTrendPlotByIndex(n_dzPhiMeanTrend, n_dzPhiResiduals, PVValHelper::MEAN, PVValHelper::phi);
  fillTrendPlotByIndex(n_dzPhiWidthTrend, n_dzPhiResiduals, PVValHelper::WIDTH);

  fillTrendPlotByIndex(n_dxyEtaMeanTrend, n_dxyEtaResiduals, PVValHelper::MEAN, PVValHelper::eta);
  fillTrendPlotByIndex(n_dxyEtaWidthTrend, n_dxyEtaResiduals, PVValHelper::WIDTH, PVValHelper::eta);
  fillTrendPlotByIndex(n_dzEtaMeanTrend, n_dzEtaResiduals, PVValHelper::MEAN, PVValHelper::eta);
  fillTrendPlotByIndex(n_dzEtaWidthTrend, n_dzEtaResiduals, PVValHelper::WIDTH, PVValHelper::eta);

  // vs transverse momentum

  fillTrendPlotByIndex(a_dxypTMeanTrend, h_dxy_pT_, PVValHelper::MEAN);
  fillTrendPlotByIndex(a_dxypTWidthTrend, h_dxy_pT_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(a_dzpTMeanTrend, h_dz_pT_, PVValHelper::MEAN);
  fillTrendPlotByIndex(a_dzpTWidthTrend, h_dz_pT_, PVValHelper::WIDTH);

  fillTrendPlotByIndex(a_dxypTCentralMeanTrend, h_dxy_Central_pT_, PVValHelper::MEAN);
  fillTrendPlotByIndex(a_dxypTCentralWidthTrend, h_dxy_Central_pT_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(a_dzpTCentralMeanTrend, h_dz_Central_pT_, PVValHelper::MEAN);
  fillTrendPlotByIndex(a_dzpTCentralWidthTrend, h_dz_Central_pT_, PVValHelper::WIDTH);

  fillTrendPlotByIndex(n_dxypTMeanTrend, h_norm_dxy_pT_, PVValHelper::MEAN);
  fillTrendPlotByIndex(n_dxypTWidthTrend, h_norm_dxy_pT_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(n_dzpTMeanTrend, h_norm_dz_pT_, PVValHelper::MEAN);
  fillTrendPlotByIndex(n_dzpTWidthTrend, h_norm_dz_pT_, PVValHelper::WIDTH);

  fillTrendPlotByIndex(n_dxypTCentralMeanTrend, h_norm_dxy_Central_pT_, PVValHelper::MEAN);
  fillTrendPlotByIndex(n_dxypTCentralWidthTrend, h_norm_dxy_Central_pT_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(n_dzpTCentralMeanTrend, h_norm_dz_Central_pT_, PVValHelper::MEAN);
  fillTrendPlotByIndex(n_dzpTCentralWidthTrend, h_norm_dz_Central_pT_, PVValHelper::WIDTH);

  // vs ladder and module number

  fillTrendPlotByIndex(a_dxymodZMeanTrend, h_dxy_modZ_, PVValHelper::MEAN);
  fillTrendPlotByIndex(a_dxymodZWidthTrend, h_dxy_modZ_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(a_dzmodZMeanTrend, h_dz_modZ_, PVValHelper::MEAN);
  fillTrendPlotByIndex(a_dzmodZWidthTrend, h_dz_modZ_, PVValHelper::WIDTH);

  fillTrendPlotByIndex(a_dxyladderMeanTrend, h_dxy_ladder_, PVValHelper::MEAN);
  fillTrendPlotByIndex(a_dxyladderWidthTrend, h_dxy_ladder_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(a_dzladderMeanTrend, h_dz_ladder_, PVValHelper::MEAN);
  fillTrendPlotByIndex(a_dzladderWidthTrend, h_dz_ladder_, PVValHelper::WIDTH);

  fillTrendPlotByIndex(n_dxymodZMeanTrend, h_norm_dxy_modZ_, PVValHelper::MEAN);
  fillTrendPlotByIndex(n_dxymodZWidthTrend, h_norm_dxy_modZ_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(n_dzmodZMeanTrend, h_norm_dz_modZ_, PVValHelper::MEAN);
  fillTrendPlotByIndex(n_dzmodZWidthTrend, h_norm_dz_modZ_, PVValHelper::WIDTH);

  fillTrendPlotByIndex(n_dxyladderMeanTrend, h_norm_dxy_ladder_, PVValHelper::MEAN);
  fillTrendPlotByIndex(n_dxyladderWidthTrend, h_norm_dxy_ladder_, PVValHelper::WIDTH);
  fillTrendPlotByIndex(n_dzladderMeanTrend, h_norm_dz_ladder_, PVValHelper::MEAN);
  fillTrendPlotByIndex(n_dzladderWidthTrend, h_norm_dz_ladder_, PVValHelper::WIDTH);

  // medians and MADs

  fillTrendPlotByIndex(a_dxyPhiMedianTrend, a_dxyPhiResiduals, PVValHelper::MEDIAN, PVValHelper::phi);
  fillTrendPlotByIndex(a_dxyPhiMADTrend, a_dxyPhiResiduals, PVValHelper::MAD, PVValHelper::phi);

  fillTrendPlotByIndex(a_dzPhiMedianTrend, a_dzPhiResiduals, PVValHelper::MEDIAN, PVValHelper::phi);
  fillTrendPlotByIndex(a_dzPhiMADTrend, a_dzPhiResiduals, PVValHelper::MAD, PVValHelper::phi);

  fillTrendPlotByIndex(a_dxyEtaMedianTrend, a_dxyEtaResiduals, PVValHelper::MEDIAN, PVValHelper::eta);
  fillTrendPlotByIndex(a_dxyEtaMADTrend, a_dxyEtaResiduals, PVValHelper::MAD, PVValHelper::eta);
  fillTrendPlotByIndex(a_dzEtaMedianTrend, a_dzEtaResiduals, PVValHelper::MEDIAN, PVValHelper::eta);
  fillTrendPlotByIndex(a_dzEtaMADTrend, a_dzEtaResiduals, PVValHelper::MAD, PVValHelper::eta);

  fillTrendPlotByIndex(n_dxyPhiMedianTrend, n_dxyPhiResiduals, PVValHelper::MEDIAN, PVValHelper::phi);
  fillTrendPlotByIndex(n_dxyPhiMADTrend, n_dxyPhiResiduals, PVValHelper::MAD, PVValHelper::phi);
  fillTrendPlotByIndex(n_dzPhiMedianTrend, n_dzPhiResiduals, PVValHelper::MEDIAN, PVValHelper::phi);
  fillTrendPlotByIndex(n_dzPhiMADTrend, n_dzPhiResiduals, PVValHelper::MAD, PVValHelper::phi);

  fillTrendPlotByIndex(n_dxyEtaMedianTrend, n_dxyEtaResiduals, PVValHelper::MEDIAN, PVValHelper::eta);
  fillTrendPlotByIndex(n_dxyEtaMADTrend, n_dxyEtaResiduals, PVValHelper::MAD, PVValHelper::eta);
  fillTrendPlotByIndex(n_dzEtaMedianTrend, n_dzEtaResiduals, PVValHelper::MEDIAN, PVValHelper::eta);
  fillTrendPlotByIndex(n_dzEtaMADTrend, n_dzEtaResiduals, PVValHelper::MAD, PVValHelper::eta);

  // 2D Maps

  fillMap(a_dxyMeanMap, a_dxyResidualsMap, PVValHelper::MEAN, nBins_, nBins_);
  fillMap(a_dxyWidthMap, a_dxyResidualsMap, PVValHelper::WIDTH, nBins_, nBins_);
  fillMap(a_dzMeanMap, a_dzResidualsMap, PVValHelper::MEAN, nBins_, nBins_);
  fillMap(a_dzWidthMap, a_dzResidualsMap, PVValHelper::WIDTH, nBins_, nBins_);

  fillMap(n_dxyMeanMap, n_dxyResidualsMap, PVValHelper::MEAN, nBins_, nBins_);
  fillMap(n_dxyWidthMap, n_dxyResidualsMap, PVValHelper::WIDTH, nBins_, nBins_);
  fillMap(n_dzMeanMap, n_dzResidualsMap, PVValHelper::MEAN, nBins_, nBins_);
  fillMap(n_dzWidthMap, n_dzResidualsMap, PVValHelper::WIDTH, nBins_, nBins_);

  // 2D Maps of residuals in bins of L1 modules

  fillMap(a_dxyL1MeanMap, a_dxyL1ResidualsMap, PVValHelper::MEAN, nModZ_, nLadders_);
  fillMap(a_dxyL1WidthMap, a_dxyL1ResidualsMap, PVValHelper::WIDTH, nModZ_, nLadders_);
  fillMap(a_dzL1MeanMap, a_dzL1ResidualsMap, PVValHelper::MEAN, nModZ_, nLadders_);
  fillMap(a_dzL1WidthMap, a_dzL1ResidualsMap, PVValHelper::WIDTH, nModZ_, nLadders_);

  fillMap(n_dxyL1MeanMap, n_dxyL1ResidualsMap, PVValHelper::MEAN, nModZ_, nLadders_);
  fillMap(n_dxyL1WidthMap, n_dxyL1ResidualsMap, PVValHelper::WIDTH, nModZ_, nLadders_);
  fillMap(n_dzL1MeanMap, n_dzL1ResidualsMap, PVValHelper::MEAN, nModZ_, nLadders_);
  fillMap(n_dzL1WidthMap, n_dzL1ResidualsMap, PVValHelper::WIDTH, nModZ_, nLadders_);
}

//*************************************************************
std::pair<long long, long long> PrimaryVertexValidation::getRunTime(const edm::EventSetup& iSetup) const
//*************************************************************
{
  edm::ESHandle<RunInfo> runInfo = iSetup.getHandle(runInfoToken_);
  if (debug_) {
    edm::LogInfo("PrimaryVertexValidation")
        << runInfo.product()->m_start_time_str << " " << runInfo.product()->m_stop_time_str << std::endl;
  }
  return std::make_pair(runInfo.product()->m_start_time_ll, runInfo.product()->m_stop_time_ll);
}

//*************************************************************
bool PrimaryVertexValidation::isBFieldConsistentWithMode(const edm::EventSetup& iSetup) const
//*************************************************************
{
  edm::ESHandle<RunInfo> runInfo = iSetup.getHandle(runInfoToken_);
  double average_current = runInfo.product()->m_avg_current;
  bool isOn = (average_current > 2000.);
  bool is0T = (ptOfProbe_ == 0.);

  return ((isOn && !is0T) || (!isOn && is0T));
}

//*************************************************************
void PrimaryVertexValidation::SetVarToZero()
//*************************************************************
{
  nTracks_ = 0;
  nClus_ = 0;
  nOfflineVertices_ = 0;
  RunNumber_ = 0;
  LuminosityBlockNumber_ = 0;
  xOfflineVertex_ = -999.;
  yOfflineVertex_ = -999.;
  zOfflineVertex_ = -999.;
  xErrOfflineVertex_ = 0.;
  yErrOfflineVertex_ = 0.;
  zErrOfflineVertex_ = 0.;
  BSx0_ = -999.;
  BSy0_ = -999.;
  BSz0_ = -999.;
  Beamsigmaz_ = -999.;
  Beamdxdz_ = -999.;
  BeamWidthX_ = -999.;
  BeamWidthY_ = -999.;
  wxy2_ = -999.;

  for (int i = 0; i < nMaxtracks_; ++i) {
    pt_[i] = 0;
    p_[i] = 0;
    nhits_[i] = 0;
    nhits1D_[i] = 0;
    nhits2D_[i] = 0;
    nhitsBPIX_[i] = 0;
    nhitsFPIX_[i] = 0;
    nhitsTIB_[i] = 0;
    nhitsTID_[i] = 0;
    nhitsTOB_[i] = 0;
    nhitsTEC_[i] = 0;
    isHighPurity_[i] = 0;
    eta_[i] = 0;
    theta_[i] = 0;
    phi_[i] = 0;
    chi2_[i] = 0;
    chi2ndof_[i] = 0;
    charge_[i] = 0;
    qoverp_[i] = 0;
    dz_[i] = 0;
    dxy_[i] = 0;
    dzBs_[i] = 0;
    dxyBs_[i] = 0;
    xPCA_[i] = 0;
    yPCA_[i] = 0;
    zPCA_[i] = 0;
    xUnbiasedVertex_[i] = 0;
    yUnbiasedVertex_[i] = 0;
    zUnbiasedVertex_[i] = 0;
    chi2normUnbiasedVertex_[i] = 0;
    chi2UnbiasedVertex_[i] = 0;
    chi2ProbUnbiasedVertex_[i] = 0;
    DOFUnbiasedVertex_[i] = 0;
    sumOfWeightsUnbiasedVertex_[i] = 0;
    tracksUsedForVertexing_[i] = 0;
    dxyFromMyVertex_[i] = 0;
    dzFromMyVertex_[i] = 0;
    d3DFromMyVertex_[i] = 0;
    dxyErrorFromMyVertex_[i] = 0;
    dzErrorFromMyVertex_[i] = 0;
    d3DErrorFromMyVertex_[i] = 0;
    IPTsigFromMyVertex_[i] = 0;
    IPLsigFromMyVertex_[i] = 0;
    IP3DsigFromMyVertex_[i] = 0;
    hasRecVertex_[i] = 0;
    isGoodTrack_[i] = 0;
  }
}

//*************************************************************
void PrimaryVertexValidation::fillTrendPlot(TH1F* trendPlot,
                                            TH1F* residualsPlot[100],
                                            PVValHelper::estimator fitPar_,
                                            const std::string& var_)
//*************************************************************
{
  for (int i = 0; i < nBins_; ++i) {
    char phibincenter[129];
    auto phiBins = theDetails_.trendbins[PVValHelper::phi];
    sprintf(phibincenter, "%.f", (phiBins[i] + phiBins[i + 1]) / 2.);

    char etabincenter[129];
    auto etaBins = theDetails_.trendbins[PVValHelper::eta];
    sprintf(etabincenter, "%.1f", (etaBins[i] + etaBins[i + 1]) / 2.);

    switch (fitPar_) {
      case PVValHelper::MEAN: {
        float mean_ = PVValHelper::fitResiduals(residualsPlot[i]).first.value();
        float meanErr_ = PVValHelper::fitResiduals(residualsPlot[i]).first.error();
        trendPlot->SetBinContent(i + 1, mean_);
        trendPlot->SetBinError(i + 1, meanErr_);
        break;
      }
      case PVValHelper::WIDTH: {
        float width_ = PVValHelper::fitResiduals(residualsPlot[i]).second.value();
        float widthErr_ = PVValHelper::fitResiduals(residualsPlot[i]).second.error();
        trendPlot->SetBinContent(i + 1, width_);
        trendPlot->SetBinError(i + 1, widthErr_);
        break;
      }
      case PVValHelper::MEDIAN: {
        float median_ = PVValHelper::getMedian(residualsPlot[i]).value();
        float medianErr_ = PVValHelper::getMedian(residualsPlot[i]).error();
        trendPlot->SetBinContent(i + 1, median_);
        trendPlot->SetBinError(i + 1, medianErr_);
        break;
      }
      case PVValHelper::MAD: {
        float mad_ = PVValHelper::getMAD(residualsPlot[i]).value();
        float madErr_ = PVValHelper::getMAD(residualsPlot[i]).error();
        trendPlot->SetBinContent(i + 1, mad_);
        trendPlot->SetBinError(i + 1, madErr_);
        break;
      }
      default:
        edm::LogWarning("PrimaryVertexValidation")
            << "fillTrendPlot() " << fitPar_ << " unknown estimator!" << std::endl;
        break;
    }

    if (var_.find("eta") != std::string::npos) {
      trendPlot->GetXaxis()->SetBinLabel(i + 1, etabincenter);
    } else if (var_.find("phi") != std::string::npos) {
      trendPlot->GetXaxis()->SetBinLabel(i + 1, phibincenter);
    } else {
      edm::LogWarning("PrimaryVertexValidation")
          << "fillTrendPlot() " << var_ << " unknown track parameter!" << std::endl;
    }
  }
}

//*************************************************************
void PrimaryVertexValidation::fillTrendPlotByIndex(TH1F* trendPlot,
                                                   std::vector<TH1F*>& h,
                                                   PVValHelper::estimator fitPar_,
                                                   PVValHelper::plotVariable plotVar)
//*************************************************************
{
  for (auto iterator = h.begin(); iterator != h.end(); iterator++) {
    unsigned int bin = std::distance(h.begin(), iterator) + 1;
    std::pair<Measurement1D, Measurement1D> myFit = PVValHelper::fitResiduals((*iterator));

    switch (fitPar_) {
      case PVValHelper::MEAN: {
        float mean_ = myFit.first.value();
        float meanErr_ = myFit.first.error();
        trendPlot->SetBinContent(bin, mean_);
        trendPlot->SetBinError(bin, meanErr_);
        break;
      }
      case PVValHelper::WIDTH: {
        float width_ = myFit.second.value();
        float widthErr_ = myFit.second.error();
        trendPlot->SetBinContent(bin, width_);
        trendPlot->SetBinError(bin, widthErr_);
        break;
      }
      case PVValHelper::MEDIAN: {
        float median_ = PVValHelper::getMedian(*iterator).value();
        float medianErr_ = PVValHelper::getMedian(*iterator).error();
        trendPlot->SetBinContent(bin, median_);
        trendPlot->SetBinError(bin, medianErr_);
        break;
      }
      case PVValHelper::MAD: {
        float mad_ = PVValHelper::getMAD(*iterator).value();
        float madErr_ = PVValHelper::getMAD(*iterator).error();
        trendPlot->SetBinContent(bin, mad_);
        trendPlot->SetBinError(bin, madErr_);
        break;
      }
      default:
        edm::LogWarning("PrimaryVertexValidation")
            << "fillTrendPlotByIndex() " << fitPar_ << " unknown estimator!" << std::endl;
        break;
    }

    char bincenter[129];
    if (plotVar == PVValHelper::eta) {
      auto etaBins = theDetails_.trendbins[PVValHelper::eta];
      sprintf(bincenter, "%.1f", (etaBins[bin - 1] + etaBins[bin]) / 2.);
      trendPlot->GetXaxis()->SetBinLabel(bin, bincenter);
    } else if (plotVar == PVValHelper::phi) {
      auto phiBins = theDetails_.trendbins[PVValHelper::phi];
      sprintf(bincenter, "%.f", (phiBins[bin - 1] + phiBins[bin]) / 2.);
      trendPlot->GetXaxis()->SetBinLabel(bin, bincenter);
    } else {
      /// FIXME DO SOMETHING HERE
      //edm::LogWarning("PrimaryVertexValidation")<<"fillTrendPlotByIndex() "<< plotVar <<" unknown track parameter!"<<std::endl;
    }
  }
}

//*************************************************************
void PrimaryVertexValidation::fillMap(TH2F* trendMap,
                                      TH1F* residualsMapPlot[100][100],
                                      PVValHelper::estimator fitPar_,
                                      const int nXBins_,
                                      const int nYBins_)
//*************************************************************
{
  for (int i = 0; i < nYBins_; ++i) {
    char phibincenter[129];
    auto phiBins = theDetails_.trendbins[PVValHelper::phi];
    sprintf(phibincenter, "%.f", (phiBins[i] + phiBins[i + 1]) / 2.);

    if (nXBins_ == nYBins_) {
      trendMap->GetYaxis()->SetBinLabel(i + 1, phibincenter);
    }

    for (int j = 0; j < nXBins_; ++j) {
      char etabincenter[129];
      auto etaBins = theDetails_.trendbins[PVValHelper::eta];
      sprintf(etabincenter, "%.1f", (etaBins[j] + etaBins[j + 1]) / 2.);

      if (i == 0) {
        if (nXBins_ == nYBins_) {
          trendMap->GetXaxis()->SetBinLabel(j + 1, etabincenter);
        }
      }

      switch (fitPar_) {
        case PVValHelper::MEAN: {
          float mean_ = PVValHelper::fitResiduals(residualsMapPlot[i][j]).first.value();
          float meanErr_ = PVValHelper::fitResiduals(residualsMapPlot[i][j]).first.error();
          trendMap->SetBinContent(j + 1, i + 1, mean_);
          trendMap->SetBinError(j + 1, i + 1, meanErr_);
          break;
        }
        case PVValHelper::WIDTH: {
          float width_ = PVValHelper::fitResiduals(residualsMapPlot[i][j]).second.value();
          float widthErr_ = PVValHelper::fitResiduals(residualsMapPlot[i][j]).second.error();
          trendMap->SetBinContent(j + 1, i + 1, width_);
          trendMap->SetBinError(j + 1, i + 1, widthErr_);
          break;
        }
        case PVValHelper::MEDIAN: {
          float median_ = PVValHelper::getMedian(residualsMapPlot[i][j]).value();
          float medianErr_ = PVValHelper::getMedian(residualsMapPlot[i][j]).error();
          trendMap->SetBinContent(j + 1, i + 1, median_);
          trendMap->SetBinError(j + 1, i + 1, medianErr_);
          break;
        }
        case PVValHelper::MAD: {
          float mad_ = PVValHelper::getMAD(residualsMapPlot[i][j]).value();
          float madErr_ = PVValHelper::getMAD(residualsMapPlot[i][j]).error();
          trendMap->SetBinContent(j + 1, i + 1, mad_);
          trendMap->SetBinError(j + 1, i + 1, madErr_);
          break;
        }
        default:
          edm::LogWarning("PrimaryVertexValidation:") << " fillMap() " << fitPar_ << " unknown estimator!" << std::endl;
      }
    }  // closes loop on eta bins
  }    // cloeses loop on phi bins
}

//*************************************************************
bool PrimaryVertexValidation::vtxSort(const reco::Vertex& a, const reco::Vertex& b)
//*************************************************************
{
  if (a.tracksSize() != b.tracksSize())
    return a.tracksSize() > b.tracksSize() ? true : false;
  else
    return a.chi2() < b.chi2() ? true : false;
}

//*************************************************************
bool PrimaryVertexValidation::passesTrackCuts(const reco::Track& track,
                                              const reco::Vertex& vertex,
                                              const std::string& qualityString_,
                                              double dxyErrMax_,
                                              double dzErrMax_,
                                              double ptErrMax_)
//*************************************************************
{
  math::XYZPoint vtxPoint(0.0, 0.0, 0.0);
  double vzErr = 0.0, vxErr = 0.0, vyErr = 0.0;
  vtxPoint = vertex.position();
  vzErr = vertex.zError();
  vxErr = vertex.xError();
  vyErr = vertex.yError();

  double dxy = 0.0, dz = 0.0, dxysigma = 0.0, dzsigma = 0.0;
  dxy = track.dxy(vtxPoint);
  dz = track.dz(vtxPoint);
  dxysigma = sqrt(track.d0Error() * track.d0Error() + vxErr * vyErr);
  dzsigma = sqrt(track.dzError() * track.dzError() + vzErr * vzErr);

  if (track.quality(reco::TrackBase::qualityByName(qualityString_)) != 1)
    return false;
  if (std::abs(dxy / dxysigma) > dxyErrMax_)
    return false;
  if (std::abs(dz / dzsigma) > dzErrMax_)
    return false;
  if (track.ptError() / track.pt() > ptErrMax_)
    return false;

  return true;
}

//*************************************************************
std::map<std::string, TH1*> PrimaryVertexValidation::bookVertexHistograms(const TFileDirectory& dir)
//*************************************************************
{
  TH1F::SetDefaultSumw2(kTRUE);

  std::map<std::string, TH1*> h;

  // histograms of track quality (Data and MC)
  std::string types[] = {"all", "sel"};
  for (const auto& type : types) {
    h["pseudorapidity_" + type] =
        dir.make<TH1F>(("rapidity_" + type).c_str(), "track pseudorapidity; track #eta; tracks", 100, -3., 3.);
    h["z0_" + type] = dir.make<TH1F>(("z0_" + type).c_str(), "track z_{0};track z_{0} (cm);tracks", 80, -40., 40.);
    h["phi_" + type] = dir.make<TH1F>(("phi_" + type).c_str(), "track #phi; track #phi;tracks", 80, -M_PI, M_PI);
    h["eta_" + type] = dir.make<TH1F>(("eta_" + type).c_str(), "track #eta; track #eta;tracks", 80, -4., 4.);
    h["pt_" + type] = dir.make<TH1F>(("pt_" + type).c_str(), "track p_{T}; track p_{T} [GeV];tracks", 100, 0., 20.);
    h["p_" + type] = dir.make<TH1F>(("p_" + type).c_str(), "track p; track p [GeV];tracks", 100, 0., 20.);
    h["found_" + type] =
        dir.make<TH1F>(("found_" + type).c_str(), "n. found hits;n^{found}_{hits};tracks", 30, 0., 30.);
    h["lost_" + type] = dir.make<TH1F>(("lost_" + type).c_str(), "n. lost hits;n^{lost}_{hits};tracks", 20, 0., 20.);
    h["nchi2_" + type] =
        dir.make<TH1F>(("nchi2_" + type).c_str(), "normalized track #chi^{2};track #chi^{2}/ndf;tracks", 100, 0., 20.);
    h["rstart_" + type] = dir.make<TH1F>(
        ("rstart_" + type).c_str(), "track start radius; track innermost radius r (cm);tracks", 100, 0., 20.);
    h["expectedInner_" + type] = dir.make<TH1F>(
        ("expectedInner_" + type).c_str(), "n. expected inner hits;n^{expected}_{inner};tracks", 10, 0., 10.);
    h["expectedOuter_" + type] = dir.make<TH1F>(
        ("expectedOuter_" + type).c_str(), "n. expected outer hits;n^{expected}_{outer};tracks ", 10, 0., 10.);
    h["logtresxy_" + type] =
        dir.make<TH1F>(("logtresxy_" + type).c_str(),
                       "log10(track r-#phi resolution/#mum);log10(track r-#phi resolution/#mum);tracks",
                       100,
                       0.,
                       5.);
    h["logtresz_" + type] = dir.make<TH1F>(("logtresz_" + type).c_str(),
                                           "log10(track z resolution/#mum);log10(track z resolution/#mum);tracks",
                                           100,
                                           0.,
                                           5.);
    h["tpullxy_" + type] =
        dir.make<TH1F>(("tpullxy_" + type).c_str(), "track r-#phi pull;pull_{r-#phi};tracks", 100, -10., 10.);
    h["tpullz_" + type] =
        dir.make<TH1F>(("tpullz_" + type).c_str(), "track r-z pull;pull_{r-z};tracks", 100, -50., 50.);
    h["tlogDCAxy_" + type] = dir.make<TH1F>(
        ("tlogDCAxy_" + type).c_str(), "track log_{10}(DCA_{r-#phi});track log_{10}(DCA_{r-#phi});tracks", 200, -5., 3.);
    h["tlogDCAz_" + type] = dir.make<TH1F>(
        ("tlogDCAz_" + type).c_str(), "track log_{10}(DCA_{r-z});track log_{10}(DCA_{r-z});tracks", 200, -5., 5.);
    h["lvseta_" + type] = dir.make<TH2F>(
        ("lvseta_" + type).c_str(), "cluster length vs #eta;track #eta;cluster length", 60, -3., 3., 20, 0., 20);
    h["lvstanlambda_" + type] = dir.make<TH2F>(("lvstanlambda_" + type).c_str(),
                                               "cluster length vs tan #lambda; tan#lambda;cluster length",
                                               60,
                                               -6.,
                                               6.,
                                               20,
                                               0.,
                                               20);
    h["restrkz_" + type] =
        dir.make<TH1F>(("restrkz_" + type).c_str(), "z-residuals (track vs vertex);res_{z} (cm);tracks", 200, -5., 5.);
    h["restrkzvsphi_" + type] = dir.make<TH2F>(("restrkzvsphi_" + type).c_str(),
                                               "z-residuals (track - vertex) vs track #phi;track #phi;res_{z} (cm)",
                                               12,
                                               -M_PI,
                                               M_PI,
                                               100,
                                               -0.5,
                                               0.5);
    h["restrkzvseta_" + type] = dir.make<TH2F>(("restrkzvseta_" + type).c_str(),
                                               "z-residuals (track - vertex) vs track #eta;track #eta;res_{z} (cm)",
                                               12,
                                               -3.,
                                               3.,
                                               200,
                                               -0.5,
                                               0.5);
    h["pulltrkzvsphi_" + type] =
        dir.make<TH2F>(("pulltrkzvsphi_" + type).c_str(),
                       "normalized z-residuals (track - vertex) vs track #phi;track #phi;res_{z}/#sigma_{res_{z}}",
                       12,
                       -M_PI,
                       M_PI,
                       100,
                       -5.,
                       5.);
    h["pulltrkzvseta_" + type] =
        dir.make<TH2F>(("pulltrkzvseta_" + type).c_str(),
                       "normalized z-residuals (track - vertex) vs track #eta;track #eta;res_{z}/#sigma_{res_{z}}",
                       12,
                       -3.,
                       3.,
                       100,
                       -5.,
                       5.);
    h["pulltrkz_" + type] = dir.make<TH1F>(("pulltrkz_" + type).c_str(),
                                           "normalized z-residuals (track vs vertex);res_{z}/#sigma_{res_{z}};tracks",
                                           100,
                                           -5.,
                                           5.);
    h["sigmatrkz0_" + type] = dir.make<TH1F>(
        ("sigmatrkz0_" + type).c_str(), "z-resolution (excluding beam);#sigma^{trk}_{z_{0}} (cm);tracks", 100, 0., 5.);
    h["sigmatrkz_" + type] = dir.make<TH1F>(
        ("sigmatrkz_" + type).c_str(), "z-resolution (including beam);#sigma^{trk}_{z} (cm);tracks", 100, 0., 5.);
    h["nbarrelhits_" + type] = dir.make<TH1F>(
        ("nbarrelhits_" + type).c_str(), "number of pixel barrel hits;n. hits Barrel Pixel;tracks", 10, 0., 10.);
    h["nbarrelLayers_" + type] = dir.make<TH1F>(
        ("nbarrelLayers_" + type).c_str(), "number of pixel barrel layers;n. layers Barrel Pixel;tracks", 10, 0., 10.);
    h["nPxLayers_" + type] = dir.make<TH1F>(
        ("nPxLayers_" + type).c_str(), "number of pixel layers (barrel+endcap);n. Pixel layers;tracks", 10, 0., 10.);
    h["nSiLayers_" + type] =
        dir.make<TH1F>(("nSiLayers_" + type).c_str(), "number of Tracker layers;n. Tracker layers;tracks", 20, 0., 20.);
    h["trackAlgo_" + type] =
        dir.make<TH1F>(("trackAlgo_" + type).c_str(), "track algorithm;track algo;tracks", 30, 0., 30.);
    h["trackQuality_" + type] =
        dir.make<TH1F>(("trackQuality_" + type).c_str(), "track quality;track quality;tracks", 7, -1., 6.);
  }

  return h;
}

//*************************************************************
// Generic booker function
//*************************************************************
std::vector<TH1F*> PrimaryVertexValidation::bookResidualsHistogram(const TFileDirectory& dir,
                                                                   unsigned int theNOfBins,
                                                                   PVValHelper::residualType resType,
                                                                   PVValHelper::plotVariable varType,
                                                                   bool isNormalized) {
  TH1F::SetDefaultSumw2(kTRUE);

  auto hash = std::make_pair(resType, varType);

  double down = theDetails_.range[hash].first;
  double up = theDetails_.range[hash].second;

  if (isNormalized) {
    up = up / 100.;
    down = down / 100.;
  }

  std::vector<TH1F*> h;
  h.reserve(theNOfBins);

  if (theNOfBins == 0) {
    edm::LogError("PrimaryVertexValidation")
        << "bookResidualsHistogram() The number of bins cannot be identically 0" << std::endl;
    assert(false);
  }

  std::string s_resType = std::get<0>(PVValHelper::getTypeString(resType));
  std::string s_varType = std::get<0>(PVValHelper::getVarString(varType));

  std::string t_resType = std::get<1>(PVValHelper::getTypeString(resType));
  std::string t_varType = std::get<1>(PVValHelper::getVarString(varType));
  std::string units = std::get<2>(PVValHelper::getTypeString(resType));

  for (unsigned int i = 0; i < theNOfBins; i++) {
    TString title = (varType == PVValHelper::phi || varType == PVValHelper::eta)
                        ? Form("%s vs %s - bin %i (%f < %s < %f);%s %s;tracks",
                               t_resType.c_str(),
                               t_varType.c_str(),
                               i,
                               theDetails_.trendbins[varType][i],
                               t_varType.c_str(),
                               theDetails_.trendbins[varType][i + 1],
                               t_resType.c_str(),
                               units.c_str())
                        : Form("%s vs %s - bin %i;%s %s;tracks",
                               t_resType.c_str(),
                               t_varType.c_str(),
                               i,
                               t_resType.c_str(),
                               units.c_str());

    TH1F* htemp = dir.make<TH1F>(
        Form("histo_%s_%s_plot%i", s_resType.c_str(), s_varType.c_str(), i),
        //Form("%s vs %s - bin %i;%s %s;tracks",t_resType.c_str(),t_varType.c_str(),i,t_resType.c_str(),units.c_str()),
        title.Data(),
        theDetails_.histobins,
        down,
        up);
    h.push_back(htemp);
  }

  return h;
}

//*************************************************************
void PrimaryVertexValidation::fillTrackHistos(std::map<std::string, TH1*>& h,
                                              const std::string& ttype,
                                              const reco::TransientTrack* tt,
                                              const reco::Vertex& v,
                                              const reco::BeamSpot& beamSpot,
                                              double fBfield_)
//*************************************************************
{
  using namespace reco;

  PVValHelper::fill(h, "pseudorapidity_" + ttype, tt->track().eta());
  PVValHelper::fill(h, "z0_" + ttype, tt->track().vz());
  PVValHelper::fill(h, "phi_" + ttype, tt->track().phi());
  PVValHelper::fill(h, "eta_" + ttype, tt->track().eta());
  PVValHelper::fill(h, "pt_" + ttype, tt->track().pt());
  PVValHelper::fill(h, "p_" + ttype, tt->track().p());
  PVValHelper::fill(h, "found_" + ttype, tt->track().found());
  PVValHelper::fill(h, "lost_" + ttype, tt->track().lost());
  PVValHelper::fill(h, "nchi2_" + ttype, tt->track().normalizedChi2());
  PVValHelper::fill(h, "rstart_" + ttype, (tt->track().innerPosition()).Rho());

  double d0Error = tt->track().d0Error();
  double d0 = tt->track().dxy(beamSpot.position());
  double dz = tt->track().dz(beamSpot.position());
  if (d0Error > 0) {
    PVValHelper::fill(h, "logtresxy_" + ttype, log(d0Error / 0.0001) / log(10.));
    PVValHelper::fill(h, "tpullxy_" + ttype, d0 / d0Error);
    PVValHelper::fill(h, "tlogDCAxy_" + ttype, log(std::abs(d0 / d0Error)));
  }
  //double z0=tt->track().vz();
  double dzError = tt->track().dzError();
  if (dzError > 0) {
    PVValHelper::fill(h, "logtresz_" + ttype, log(dzError / 0.0001) / log(10.));
    PVValHelper::fill(h, "tpullz_" + ttype, dz / dzError);
    PVValHelper::fill(h, "tlogDCAz_" + ttype, log(std::abs(dz / dzError)));
  }

  //
  double wxy2_ = pow(beamSpot.BeamWidthX(), 2) + pow(beamSpot.BeamWidthY(), 2);

  PVValHelper::fill(
      h, "sigmatrkz_" + ttype, sqrt(pow(tt->track().dzError(), 2) + wxy2_ / pow(tan(tt->track().theta()), 2)));
  PVValHelper::fill(h, "sigmatrkz0_" + ttype, tt->track().dzError());

  // track vs vertex
  if (v.isValid()) {  // && (v.ndof()<10.)) {
    // emulate clusterizer input
    //const TransientTrack & tt = theB_->build(&t); wrong !!!!
    //reco::TransientTrack tt = theB_->build(&t);
    //ttt->track().setBeamSpot(beamSpot); // need the setBeamSpot !
    double z = (tt->stateAtBeamLine().trackStateAtPCA()).position().z();
    double tantheta = tan((tt->stateAtBeamLine().trackStateAtPCA()).momentum().theta());
    double dz2 = pow(tt->track().dzError(), 2) + wxy2_ / pow(tantheta, 2);

    PVValHelper::fill(h, "restrkz_" + ttype, z - v.position().z());
    PVValHelper::fill(h, "restrkzvsphi_" + ttype, tt->track().phi(), z - v.position().z());
    PVValHelper::fill(h, "restrkzvseta_" + ttype, tt->track().eta(), z - v.position().z());
    PVValHelper::fill(h, "pulltrkzvsphi_" + ttype, tt->track().phi(), (z - v.position().z()) / sqrt(dz2));
    PVValHelper::fill(h, "pulltrkzvseta_" + ttype, tt->track().eta(), (z - v.position().z()) / sqrt(dz2));

    PVValHelper::fill(h, "pulltrkz_" + ttype, (z - v.position().z()) / sqrt(dz2));

    double x1 = tt->track().vx() - beamSpot.x0();
    double y1 = tt->track().vy() - beamSpot.y0();

    double kappa = -0.002998 * fBfield_ * tt->track().qoverp() / cos(tt->track().theta());
    double D0 = x1 * sin(tt->track().phi()) - y1 * cos(tt->track().phi()) - 0.5 * kappa * (x1 * x1 + y1 * y1);
    double q = sqrt(1. - 2. * kappa * D0);
    double s0 = (x1 * cos(tt->track().phi()) + y1 * sin(tt->track().phi())) / q;
    // double s1;
    if (std::abs(kappa * s0) > 0.001) {
      //s1=asin(kappa*s0)/kappa;
    } else {
      //double ks02=(kappa*s0)*(kappa*s0);
      //s1=s0*(1.+ks02/6.+3./40.*ks02*ks02+5./112.*pow(ks02,3));
    }
    // sp.ddcap=-2.*D0/(1.+q);
    //double zdcap=tt->track().vz()-s1/tan(tt->track().theta());
  }
  //

  // collect some info on hits and clusters
  PVValHelper::fill(h, "nbarrelLayers_" + ttype, tt->track().hitPattern().pixelBarrelLayersWithMeasurement());
  PVValHelper::fill(h, "nPxLayers_" + ttype, tt->track().hitPattern().pixelLayersWithMeasurement());
  PVValHelper::fill(h, "nSiLayers_" + ttype, tt->track().hitPattern().trackerLayersWithMeasurement());
  PVValHelper::fill(
      h, "expectedInner_" + ttype, tt->track().hitPattern().numberOfLostHits(HitPattern::MISSING_INNER_HITS));
  PVValHelper::fill(
      h, "expectedOuter_" + ttype, tt->track().hitPattern().numberOfLostHits(HitPattern::MISSING_OUTER_HITS));
  PVValHelper::fill(h, "trackAlgo_" + ttype, tt->track().algo());
  PVValHelper::fill(h, "trackQuality_" + ttype, tt->track().qualityMask());

  //
  int longesthit = 0, nbarrel = 0;
  for (auto const& hit : tt->track().recHits()) {
    if (hit->isValid() && hit->geographicalId().det() == DetId::Tracker) {
      bool barrel = DetId(hit->geographicalId()).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
      //bool endcap = DetId::DetId(hit->geographicalId()).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
      if (barrel) {
        const SiPixelRecHit* pixhit = dynamic_cast<const SiPixelRecHit*>(&(*hit));
        edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> const& clust = (*pixhit).cluster();
        if (clust.isNonnull()) {
          nbarrel++;
          if (clust->sizeY() - longesthit > 0)
            longesthit = clust->sizeY();
          if (clust->sizeY() > 20.) {
            PVValHelper::fill(h, "lvseta_" + ttype, tt->track().eta(), 19.9);
            PVValHelper::fill(h, "lvstanlambda_" + ttype, tan(tt->track().lambda()), 19.9);
          } else {
            PVValHelper::fill(h, "lvseta_" + ttype, tt->track().eta(), float(clust->sizeY()));
            PVValHelper::fill(h, "lvstanlambda_" + ttype, tan(tt->track().lambda()), float(clust->sizeY()));
          }
        }
      }
    }
  }
  PVValHelper::fill(h, "nbarrelhits_" + ttype, float(nbarrel));
  //-------------------------------------------------------------------
}

void PrimaryVertexValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Validates alignment payloads by evaluating unbiased track paramter resisuals to vertices");

  // PV Validation specific

  desc.addUntracked<int>("compressionSettings", -1);
  desc.add<bool>("storeNtuple", false);
  desc.add<bool>("isLightNtuple", true);
  desc.add<bool>("useTracksFromRecoVtx", false);
  desc.addUntracked<double>("vertexZMax", 99);
  desc.addUntracked<double>("intLumi", 0.);
  desc.add<bool>("askFirstLayerHit", false);
  desc.addUntracked<bool>("doBPix", true);
  desc.addUntracked<bool>("doFPix", true);
  desc.addUntracked<double>("probePt", 0.);
  desc.addUntracked<double>("probeP", 0.);
  desc.addUntracked<double>("probeEta", 2.4);
  desc.addUntracked<double>("probeNHits", 0.);
  desc.addUntracked<int>("numberOfBins", 24);
  desc.addUntracked<double>("minPt", 1.);
  desc.addUntracked<double>("maxPt", 20.);
  desc.add<bool>("Debug", false);
  desc.addUntracked<bool>("runControl", false);
  desc.addUntracked<bool>("forceBeamSpot", false);

  std::vector<unsigned int> defaultRuns;
  defaultRuns.push_back(0);
  desc.addUntracked<std::vector<unsigned int>>("runControlNumber", defaultRuns);

  // event sources

  desc.add<edm::InputTag>("TrackCollectionTag", edm::InputTag("ALCARECOTkAlMinBias"));
  desc.add<edm::InputTag>("VertexCollectionTag", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("BeamSpotTag", edm::InputTag("offlineBeamSpot"));

  // track filtering

  edm::ParameterSetDescription psd0;
  psd0.add<double>("maxNormalizedChi2", 5.0);
  psd0.add<double>("minPt", 0.0);
  psd0.add<std::string>("algorithm", "filter");
  psd0.add<double>("maxEta", 5.0);
  psd0.add<double>("maxD0Significance", 5.0);
  psd0.add<double>("maxD0Error", 1.0);
  psd0.add<double>("maxDzError", 1.0);
  psd0.add<std::string>("trackQuality", "any");
  psd0.add<int>("minPixelLayersWithHits", 2);
  psd0.add<int>("minSiliconLayersWithHits", 5);
  psd0.add<int>("numTracksThreshold", 0);  // HI only
  desc.add<edm::ParameterSetDescription>("TkFilterParameters", psd0);

  // PV Clusterization
  {
    edm::ParameterSetDescription psd0;
    {
      edm::ParameterSetDescription psd1;
      psd1.addUntracked<bool>("verbose", false);
      psd1.addUntracked<double>("zdumpcenter", 0.);
      psd1.addUntracked<double>("zdumpwidth", 20.);
      psd1.addUntracked<bool>("use_vdt", false);  // obsolete, appears in HLT configs
      psd1.add<double>("d0CutOff", 3.0);
      psd1.add<double>("Tmin", 2.0);
      psd1.add<double>("delta_lowT", 0.001);
      psd1.add<double>("zmerge", 0.01);
      psd1.add<double>("dzCutOff", 3.0);
      psd1.add<double>("Tpurge", 2.0);
      psd1.add<int>("convergence_mode", 0);
      psd1.add<double>("delta_highT", 0.01);
      psd1.add<double>("Tstop", 0.5);
      psd1.add<double>("coolingFactor", 0.6);
      psd1.add<double>("vertexSize", 0.006);
      psd1.add<double>("uniquetrkweight", 0.8);
      psd1.add<double>("zrange", 4.0);
      psd1.add<double>("tmerge", 0.01);           // 4D only
      psd1.add<double>("dtCutOff", 4.);           // 4D only
      psd1.add<double>("t0Max", 1.0);             // 4D only
      psd1.add<double>("vertexSizeTime", 0.008);  // 4D only
      psd0.add<edm::ParameterSetDescription>("TkDAClusParameters", psd1);
      edm::ParameterSetDescription psd2;
      psd2.add<double>("zSeparation", 1.0);
      psd0.add<edm::ParameterSetDescription>("TkGapClusParameters", psd2);
    }
    psd0.add<std::string>("algorithm", "DA_vect");
    desc.add<edm::ParameterSetDescription>("TkClusParameters", psd0);
  }

  descriptions.add("primaryVertexValidation", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PrimaryVertexValidation);
