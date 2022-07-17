// Package:    SiOuterTracker
// Class:      SiOuterTracker
//
// Author: Emily MacDonald (emily.kaelyn.macdonald@cern.ch)

// system include files
#include <memory>
#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>

// user include files
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"

#include "DQMOffline/L1Trigger/interface/L1TPhase2OuterTrackerTkMET.h"

// constructors and destructor
L1TPhase2OuterTrackerTkMET::L1TPhase2OuterTrackerTkMET(const edm::ParameterSet& iConfig)
    : conf_(iConfig), m_topoToken(esConsumes()) {
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  ttTrackToken_ =
      consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > >(conf_.getParameter<edm::InputTag>("TTTracksTag"));
  pvToken = consumes<l1t::TkPrimaryVertexCollection>(conf_.getParameter<edm::InputTag>("L1VertexInputTag"));

  maxZ0 = conf_.getParameter<double>("maxZ0");
  DeltaZ = conf_.getParameter<double>("DeltaZ");
  chi2dofMax = conf_.getParameter<double>("chi2dofMax");
  bendchi2Max = conf_.getParameter<double>("bendchi2Max");
  minPt = conf_.getParameter<double>("minPt");
  nStubsmin = iConfig.getParameter<int>("nStubsmin");
  nStubsPSmin = iConfig.getParameter<int>("nStubsPSmin");
  maxPt = conf_.getParameter<double>("maxPt");
  maxEta = conf_.getParameter<double>("maxEta");
  HighPtTracks = iConfig.getParameter<int>("HighPtTracks");
}

L1TPhase2OuterTrackerTkMET::~L1TPhase2OuterTrackerTkMET() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// member functions

// ------------ method called for each event  ------------
void L1TPhase2OuterTrackerTkMET::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // L1 Primaries
  edm::Handle<l1t::TkPrimaryVertexCollection> L1VertexHandle;
  iEvent.getByToken(pvToken, L1VertexHandle);

  edm::Handle<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > > TTTrackHandle;
  iEvent.getByToken(ttTrackToken_, TTTrackHandle);

  // for PS stubs
  // Tracker Topology
  const TrackerTopology* const tTopo = &iSetup.getData(m_topoToken);

  // Adding protection
  if (!TTTrackHandle.isValid()) {
    edm::LogWarning("L1TPhase2OuterTrackerTkMET") << "cant find tracks" << std::endl;
    return;
  }
  if (!L1VertexHandle.isValid()) {
    edm::LogWarning("L1TPhase2OuterTrackerTkMET") << "cant find vertex" << std::endl;
    return;
  }
  float sumPx = 0;
  float sumPy = 0;
  float etTot = 0;
  double sumPx_PU = 0;
  double sumPy_PU = 0;
  double etTot_PU = 0;
  int nTracks_counter = 0;

  float zVTX = L1VertexHandle->begin()->zvertex();
  unsigned int tkCnt = 0;
  for (const auto& trackIter : *TTTrackHandle) {
    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > tempTrackPtr(TTTrackHandle, tkCnt++);  /// Make the pointer
    float pt = tempTrackPtr->momentum().perp();
    float phi = tempTrackPtr->momentum().phi();
    float eta = tempTrackPtr->momentum().eta();
    std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> > >
        theStubs = trackIter.getStubRefs();
    int nstubs = (int)theStubs.size();

    float chi2dof = tempTrackPtr->chi2Red();
    float bendchi2 = tempTrackPtr->stubPtConsistency();
    float z0 = tempTrackPtr->z0();

    if (pt < minPt || fabs(z0) > maxZ0 || fabs(eta) > maxEta || chi2dof > chi2dofMax || bendchi2 > bendchi2Max)
      continue;
    if (maxPt > 0 && pt > maxPt) {
      if (HighPtTracks == 0)
        continue;  // ignore these very high PT tracks: truncate
      if (HighPtTracks == 1)
        pt = maxPt;  // saturate
    }

    int nPS = 0.;  // number of stubs in PS modules
    // loop over the stubs
    for (unsigned int istub = 0; istub < (unsigned int)theStubs.size(); istub++) {
      DetId detId(theStubs.at(istub)->getDetId());
      if (detId.det() == DetId::Detector::Tracker) {
        if ((detId.subdetId() == StripSubdetector::TOB && tTopo->tobLayer(detId) <= 3) ||
            (detId.subdetId() == StripSubdetector::TID && tTopo->tidRing(detId) <= 9))
          nPS++;
      }
    }

    if (nstubs < nStubsmin || nPS < nStubsPSmin)
      continue;

    // construct deltaZ cut to be based on track eta
    if (fabs(eta) >= 0 && fabs(eta) < 0.7)
      DeltaZ = 0.4;
    else if (fabs(eta) >= 0.7 && fabs(eta) < 1.0)
      DeltaZ = 0.6;
    else if (fabs(eta) >= 1.0 && fabs(eta) < 1.2)
      DeltaZ = 0.76;
    else if (fabs(eta) >= 1.2 && fabs(eta) < 1.6)
      DeltaZ = 1.0;
    else if (fabs(eta) >= 1.6 && fabs(eta) < 2.0)
      DeltaZ = 1.7;
    else if (fabs(eta) >= 2.0 && fabs(eta) <= 2.4)
      DeltaZ = 2.2;

    if (fabs(z0 - zVTX) <= DeltaZ) {
      nTracks_counter++;
      Track_Pt->Fill(pt);
      Track_NStubs->Fill(nstubs);
      Track_NPSstubs->Fill(nPS);
      Track_Eta->Fill(eta);
      Track_VtxZ->Fill(z0);
      Track_Chi2Dof->Fill(chi2dof);
      Track_BendChi2->Fill(bendchi2);

      sumPx += pt * cos(phi);
      sumPy += pt * sin(phi);
      etTot += pt;
    } else {  // PU sums
      sumPx_PU += pt * cos(phi);
      sumPy_PU += pt * sin(phi);
      etTot_PU += pt;
    }
  }  // end loop over tracks

  Track_N->Fill(nTracks_counter);
  float et = sqrt(sumPx * sumPx + sumPy * sumPy);
  double etmiss_PU = sqrt(sumPx_PU * sumPx_PU + sumPy_PU * sumPy_PU);

  math::XYZTLorentzVector missingEt(-sumPx, -sumPy, 0, et);

  TkMET_QualityCuts->Fill(missingEt.Pt());
  TkMET_PU->Fill(etmiss_PU);

}  // end of method

// ------------ method called once each job just before starting event loop  ------------
//Creating all histograms for DQM file output
void L1TPhase2OuterTrackerTkMET::bookHistograms(DQMStore::IBooker& iBooker,
                                                edm::Run const& run,
                                                edm::EventSetup const& es) {
  std::string HistoName;
  iBooker.setCurrentFolder(topFolderName_ + "/TkMET_Tracks/");

  // Num of L1Tracks in tkMET selection
  HistoName = "Track_N";
  edm::ParameterSet psTrack_N = conf_.getParameter<edm::ParameterSet>("TH1_NTracks");
  Track_N = iBooker.book1D(HistoName,
                           HistoName,
                           psTrack_N.getParameter<int32_t>("Nbinsx"),
                           psTrack_N.getParameter<double>("xmin"),
                           psTrack_N.getParameter<double>("xmax"));
  Track_N->setAxisTitle("# L1 Tracks", 1);
  Track_N->setAxisTitle("# Events", 2);

  //Pt of the tracks
  edm::ParameterSet psTrack_Pt = conf_.getParameter<edm::ParameterSet>("TH1_Track_Pt");
  HistoName = "Track_Pt";
  Track_Pt = iBooker.book1D(HistoName,
                            HistoName,
                            psTrack_Pt.getParameter<int32_t>("Nbinsx"),
                            psTrack_Pt.getParameter<double>("xmin"),
                            psTrack_Pt.getParameter<double>("xmax"));
  Track_Pt->setAxisTitle("p_{T} [GeV]", 1);
  Track_Pt->setAxisTitle("# L1 Tracks", 2);

  //Eta
  edm::ParameterSet psTrack_Eta = conf_.getParameter<edm::ParameterSet>("TH1_Track_Eta");
  HistoName = "Track_Eta";
  Track_Eta = iBooker.book1D(HistoName,
                             HistoName,
                             psTrack_Eta.getParameter<int32_t>("Nbinsx"),
                             psTrack_Eta.getParameter<double>("xmin"),
                             psTrack_Eta.getParameter<double>("xmax"));
  Track_Eta->setAxisTitle("#eta", 1);
  Track_Eta->setAxisTitle("# L1 Tracks", 2);

  //VtxZ
  edm::ParameterSet psTrack_VtxZ = conf_.getParameter<edm::ParameterSet>("TH1_Track_VtxZ");
  HistoName = "Track_VtxZ";
  Track_VtxZ = iBooker.book1D(HistoName,
                              HistoName,
                              psTrack_VtxZ.getParameter<int32_t>("Nbinsx"),
                              psTrack_VtxZ.getParameter<double>("xmin"),
                              psTrack_VtxZ.getParameter<double>("xmax"));
  Track_VtxZ->setAxisTitle("L1 Track vertex position z [cm]", 1);
  Track_VtxZ->setAxisTitle("# L1 Tracks", 2);

  //chi2dof
  edm::ParameterSet psTrack_Chi2Dof = conf_.getParameter<edm::ParameterSet>("TH1_Track_Chi2Dof");
  HistoName = "Track_Chi2Dof";
  Track_Chi2Dof = iBooker.book1D(HistoName,
                                 HistoName,
                                 psTrack_Chi2Dof.getParameter<int32_t>("Nbinsx"),
                                 psTrack_Chi2Dof.getParameter<double>("xmin"),
                                 psTrack_Chi2Dof.getParameter<double>("xmax"));
  Track_Chi2Dof->setAxisTitle("L1 Track #chi^{2}/D.O.F.", 1);
  Track_Chi2Dof->setAxisTitle("# L1 Tracks", 2);

  //bend chi2
  edm::ParameterSet psTrack_BendChi2 = conf_.getParameter<edm::ParameterSet>("TH1_Track_BendChi2");
  HistoName = "Track_BendChi2";
  Track_BendChi2 = iBooker.book1D(HistoName,
                                  HistoName,
                                  psTrack_BendChi2.getParameter<int32_t>("Nbinsx"),
                                  psTrack_BendChi2.getParameter<double>("xmin"),
                                  psTrack_BendChi2.getParameter<double>("xmax"));
  Track_BendChi2->setAxisTitle("L1 Track Bend #chi^{2}", 1);
  Track_BendChi2->setAxisTitle("# L1 Tracks", 2);

  //nstubs
  edm::ParameterSet psTrack_NStubs = conf_.getParameter<edm::ParameterSet>("TH1_Track_NStubs");
  HistoName = "Track_NStubs";
  Track_NStubs = iBooker.book1D(HistoName,
                                HistoName,
                                psTrack_NStubs.getParameter<int32_t>("Nbinsx"),
                                psTrack_NStubs.getParameter<double>("xmin"),
                                psTrack_NStubs.getParameter<double>("xmax"));
  Track_NStubs->setAxisTitle("# L1 Stubs", 1);
  Track_NStubs->setAxisTitle("# L1 Tracks", 2);

  //nPSstubs
  edm::ParameterSet psTrack_NPSstubs = conf_.getParameter<edm::ParameterSet>("TH1_Track_NPSstubs");
  HistoName = "Track_NPSstubs";
  Track_NPSstubs = iBooker.book1D(HistoName,
                                  HistoName,
                                  psTrack_NPSstubs.getParameter<int32_t>("Nbinsx"),
                                  psTrack_NPSstubs.getParameter<double>("xmin"),
                                  psTrack_NPSstubs.getParameter<double>("xmax"));
  Track_NPSstubs->setAxisTitle("# PS Stubs", 1);
  Track_NPSstubs->setAxisTitle("# L1 Tracks", 2);

  iBooker.setCurrentFolder(topFolderName_ + "/TkMET");
  //loose tkMET
  edm::ParameterSet psTrack_TkMET = conf_.getParameter<edm::ParameterSet>("TH1_Track_TkMET");
  HistoName = "TkMET_QualityCuts";
  TkMET_QualityCuts = iBooker.book1D(HistoName,
                                     HistoName,
                                     psTrack_TkMET.getParameter<int32_t>("Nbinsx"),
                                     psTrack_TkMET.getParameter<double>("xmin"),
                                     psTrack_TkMET.getParameter<double>("xmax"));
  TkMET_QualityCuts->setAxisTitle("L1 Track MET [GeV]", 1);
  TkMET_QualityCuts->setAxisTitle("# Events", 2);

  //tkMET -- PU only
  HistoName = "TkMET_PU";
  TkMET_PU = iBooker.book1D(HistoName,
                            HistoName,
                            psTrack_TkMET.getParameter<int32_t>("Nbinsx"),
                            psTrack_TkMET.getParameter<double>("xmin"),
                            psTrack_TkMET.getParameter<double>("xmax"));
  TkMET_PU->setAxisTitle("L1 Track MET (PU only) [GeV]", 1);
  TkMET_PU->setAxisTitle("# Events", 2);
}  //end of method

DEFINE_FWK_MODULE(L1TPhase2OuterTrackerTkMET);
