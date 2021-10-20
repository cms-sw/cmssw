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
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "DataFormats/L1TCorrelator/interface/TkJet.h"
#include "DataFormats/L1TCorrelator/interface/TkJetFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkHTMiss.h"
#include "DataFormats/L1TCorrelator/interface/TkHTMissFwd.h"

#include "DQMOffline/L1Trigger/interface/L1TPhase2OTTrackJetsandMHT.h"


// constructors and destructor
L1TPhase2OTTrackJetsandMHT::L1TPhase2OTTrackJetsandMHT(const edm::ParameterSet& iConfig)
:conf_(iConfig)
{
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  jetToken      = consumes< l1t::TkJetCollection > (conf_.getParameter<edm::InputTag>("L1TkJetInputTag"));
  jet_minPt  = conf_.getParameter<double>("jet_minPt");
  jet_maxEta = conf_.getParameter<double>("jet_maxEta");
  minNtracksHighPt = conf_.getParameter<int>("jet_minNtracksHighPt");
  minNtracksLowPt = conf_.getParameter<int>("jet_minNtracksLowPt");
}

L1TPhase2OTTrackJetsandMHT::~L1TPhase2OTTrackJetsandMHT()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// member functions

// ------------ method called for each event  ------------
void L1TPhase2OTTrackJetsandMHT::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle< l1t::TkJetCollection > L1TkJetsHandle;
  iEvent.getByToken(jetToken, L1TkJetsHandle);
  std::vector< l1t::TkJet >::const_iterator jetIter;

  if ( ! L1TkJetsHandle.isValid() ) {
    edm::LogWarning("DataNotFound")<< "\nWarning: L1TkJetParticleCollection not found in the event. Exit"<< std::endl;
    return;
  }
  float sumPx = 0;
  float sumPy = 0;
  float tmp_HT = 0;

  // loop over jets
  for (jetIter = L1TkJetsHandle->begin(); jetIter != L1TkJetsHandle->end(); ++jetIter) {

    float px = jetIter->px();
    float py = jetIter->py();
    float et = jetIter->et();
    float tmp_jet_pt  = jetIter->pt();
    float tmp_jet_eta = jetIter->eta();
    float tmp_jet_phi = jetIter->phi();
    float tmp_jet_vtx = jetIter->jetVtx();
    float tmp_jet_ntracks = jetIter->ntracks();
    float tmp_jet_nTighttracks = jetIter->nTighttracks();
    float tmp_jet_nDisplacedtracks = jetIter->nDisptracks();
    float tmp_jet_nTightDisptracks = jetIter->nTightDisptracks();

    allJets_pt->Fill(tmp_jet_pt);
    allJets_eta->Fill(tmp_jet_eta);
    allJets_phi->Fill(tmp_jet_phi);
    allJets_vtx->Fill(tmp_jet_vtx);
    allJets_nTracks->Fill(tmp_jet_ntracks);
    allJets_nTightTracks->Fill(tmp_jet_nTighttracks);
    allJets_nDisplacedTracks->Fill(tmp_jet_nDisplacedtracks);
    allJets_nTightDispTracks->Fill(tmp_jet_nTightDisptracks);

    if (tmp_jet_pt < jet_minPt) continue;
    if (fabs(tmp_jet_eta) > jet_maxEta) continue;
    if(tmp_jet_ntracks<minNtracksLowPt && et>50)continue;
    if(tmp_jet_ntracks<minNtracksHighPt && et>100)continue;

    HTJets_pt->Fill(tmp_jet_pt);
    HTJets_eta->Fill(tmp_jet_eta);
    HTJets_phi->Fill(tmp_jet_phi);
    HTJets_vtx->Fill(tmp_jet_vtx);
    HTJets_nTracks->Fill(tmp_jet_ntracks);
    HTJets_nTightTracks->Fill(tmp_jet_nTighttracks);
    HTJets_nDisplacedTracks->Fill(tmp_jet_nDisplacedtracks);
    HTJets_nTightDispTracks->Fill(tmp_jet_nTightDisptracks);

    sumPx += px;
    sumPy += py;
    tmp_HT += tmp_jet_pt;
  } // end jet loop

  // float tmp_MHT = sqrt(sumPx*sumPx + sumPy*sumPy);
  // define missing HT
  float et = sqrt(sumPx*sumPx + sumPy*sumPy);
  math::XYZTLorentzVector missingEt(-sumPx, -sumPy, 0, et);
  // edm::RefProd<l1t::TkJetCollection> jetCollRef(L1TkJetsHandle);
  // edm::Ref< l1t::TkPrimaryVertexCollection > L1VtxRef; 	// null reference
  // l1t::TkHTMiss tkMHT(missingEt, tmp_HT, jetCollRef, L1VtxRef);
  double MHT = missingEt.Pt();
  double HT = tmp_HT;
  h_MHT->Fill(MHT);
  h_HT->Fill(HT);

} // end of method

// ------------ method called once each job just before starting event loop  ------------
//Creating all histograms for DQM file output
void L1TPhase2OTTrackJetsandMHT::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const & run, edm::EventSetup const & es) {
  std::string HistoName;

  iBooker.setCurrentFolder(topFolderName_+"/TrackJets/AllJets");

  // Num of tracks in track jet
  HistoName = "NTracks";
  edm::ParameterSet psTrack_N =  conf_.getParameter<edm::ParameterSet>("TH1_NTracks");
  allJets_nTracks = iBooker.book1D(HistoName, HistoName, psTrack_N.getParameter<int32_t>("Nbinsx"), psTrack_N.getParameter<double>("xmin"), psTrack_N.getParameter<double>("xmax"));
  allJets_nTracks->setAxisTitle("# Tracks per jet", 1);
  allJets_nTracks->setAxisTitle("# Events", 2);

  // Num of tight tracks in track jet
  HistoName = "NTightTracks";
  allJets_nTightTracks = iBooker.book1D(HistoName, HistoName, psTrack_N.getParameter<int32_t>("Nbinsx"), psTrack_N.getParameter<double>("xmin"), psTrack_N.getParameter<double>("xmax"));
  allJets_nTightTracks->setAxisTitle("# Tight tracks per jet", 1);
  allJets_nTightTracks->setAxisTitle("# Events", 2);

  // Num of displaced tracks in track jet
  HistoName = "NDisplacedTracks";
  allJets_nDisplacedTracks = iBooker.book1D(HistoName, HistoName, psTrack_N.getParameter<int32_t>("Nbinsx"), psTrack_N.getParameter<double>("xmin"), psTrack_N.getParameter<double>("xmax"));
  allJets_nDisplacedTracks->setAxisTitle("# Displaced tracks per jet", 1);
  allJets_nDisplacedTracks->setAxisTitle("# Events", 2);

  // Num of tight displaced tracks in track jet
  HistoName = "NTightDispTracks";
  allJets_nTightDispTracks = iBooker.book1D(HistoName, HistoName, psTrack_N.getParameter<int32_t>("Nbinsx"), psTrack_N.getParameter<double>("xmin"), psTrack_N.getParameter<double>("xmax"));
  allJets_nTightDispTracks->setAxisTitle("# Tight displaced tracks per jet", 1);
  allJets_nTightDispTracks->setAxisTitle("# Events", 2);

  //Pt of jets
  edm::ParameterSet psJet_Pt =  conf_.getParameter<edm::ParameterSet>("TH1_Jet_Pt");
  HistoName = "Jet_Pt";
  allJets_pt = iBooker.book1D(HistoName, HistoName, psJet_Pt.getParameter<int32_t>("Nbinsx"), psJet_Pt.getParameter<double>("xmin"), psJet_Pt.getParameter<double>("xmax"));
  allJets_pt->setAxisTitle("p_{T} [GeV]", 1);
  allJets_pt->setAxisTitle("# Track Jets", 2);

  //Eta
  edm::ParameterSet psJet_Eta =  conf_.getParameter<edm::ParameterSet>("TH1_Jet_Eta");
  HistoName = "Track_Eta";
  allJets_eta = iBooker.book1D(HistoName, HistoName, psJet_Eta.getParameter<int32_t>("Nbinsx"), psJet_Eta.getParameter<double>("xmin"), psJet_Eta.getParameter<double>("xmax"));
  allJets_eta->setAxisTitle("#eta", 1);
  allJets_eta->setAxisTitle("# Track Jets", 2);

  //Phi
  edm::ParameterSet psJet_Phi =  conf_.getParameter<edm::ParameterSet>("TH1_Jet_Phi");
  HistoName = "Track_Phi";
  allJets_phi = iBooker.book1D(HistoName, HistoName, psJet_Phi.getParameter<int32_t>("Nbinsx"), psJet_Phi.getParameter<double>("xmin"), psJet_Phi.getParameter<double>("xmax"));
  allJets_phi->setAxisTitle("#phi", 1);
  allJets_phi->setAxisTitle("# Track Jets", 2);

  //Vtx
  edm::ParameterSet psJet_Vtx =  conf_.getParameter<edm::ParameterSet>("TH1_Jet_Vtx");
  HistoName = "Jet_Vtx";
  allJets_vtx = iBooker.book1D(HistoName, HistoName, psJet_Vtx.getParameter<int32_t>("Nbinsx"), psJet_Vtx.getParameter<double>("xmin"), psJet_Vtx.getParameter<double>("xmax"));
  allJets_vtx->setAxisTitle("Track Jet vertex position z [cm]", 1);
  allJets_vtx->setAxisTitle("# Track Jets", 2);



  iBooker.setCurrentFolder(topFolderName_+"/TrackJets/HTJets");

  // Num of tracks in track jet
  HistoName = "NTracks";
  HTJets_nTracks = iBooker.book1D(HistoName, HistoName, psTrack_N.getParameter<int32_t>("Nbinsx"), psTrack_N.getParameter<double>("xmin"), psTrack_N.getParameter<double>("xmax"));
  HTJets_nTracks->setAxisTitle("# Tracks per jet", 1);
  HTJets_nTracks->setAxisTitle("# Events", 2);

  // Num of tight tracks in track jet
  HistoName = "NTightTracks";
  HTJets_nTightTracks = iBooker.book1D(HistoName, HistoName, psTrack_N.getParameter<int32_t>("Nbinsx"), psTrack_N.getParameter<double>("xmin"), psTrack_N.getParameter<double>("xmax"));
  HTJets_nTightTracks->setAxisTitle("# Tight tracks per jet", 1);
  HTJets_nTightTracks->setAxisTitle("# Events", 2);

  // Num of displaced tracks in track jet
  HistoName = "NDisplacedTracks";
  HTJets_nDisplacedTracks = iBooker.book1D(HistoName, HistoName, psTrack_N.getParameter<int32_t>("Nbinsx"), psTrack_N.getParameter<double>("xmin"), psTrack_N.getParameter<double>("xmax"));
  HTJets_nDisplacedTracks->setAxisTitle("# Displaced tracks per jet", 1);
  HTJets_nDisplacedTracks->setAxisTitle("# Events", 2);

  // Num of tight displaced tracks in track jet
  HistoName = "NTightDispTracks";
  HTJets_nTightDispTracks = iBooker.book1D(HistoName, HistoName, psTrack_N.getParameter<int32_t>("Nbinsx"), psTrack_N.getParameter<double>("xmin"), psTrack_N.getParameter<double>("xmax"));
  HTJets_nTightDispTracks->setAxisTitle("# Tight displaced tracks per jet", 1);
  HTJets_nTightDispTracks->setAxisTitle("# Events", 2);

  //Pt of jets
  HistoName = "Jet_Pt";
  HTJets_pt = iBooker.book1D(HistoName, HistoName, psJet_Pt.getParameter<int32_t>("Nbinsx"), psJet_Pt.getParameter<double>("xmin"), psJet_Pt.getParameter<double>("xmax"));
  HTJets_pt->setAxisTitle("p_{T} [GeV]", 1);
  HTJets_pt->setAxisTitle("# Track Jets", 2);

  //Eta
  HistoName = "Track_Eta";
  HTJets_eta = iBooker.book1D(HistoName, HistoName, psJet_Eta.getParameter<int32_t>("Nbinsx"), psJet_Eta.getParameter<double>("xmin"), psJet_Eta.getParameter<double>("xmax"));
  HTJets_eta->setAxisTitle("#eta", 1);
  HTJets_eta->setAxisTitle("# Track Jets", 2);

  //Phi
  HistoName = "Track_Phi";
  HTJets_phi = iBooker.book1D(HistoName, HistoName, psJet_Phi.getParameter<int32_t>("Nbinsx"), psJet_Phi.getParameter<double>("xmin"), psJet_Phi.getParameter<double>("xmax"));
  HTJets_phi->setAxisTitle("#phi", 1);
  HTJets_phi->setAxisTitle("# Track Jets", 2);

  //Vtx
  HistoName = "Jet_Vtx";
  HTJets_vtx = iBooker.book1D(HistoName, HistoName, psJet_Vtx.getParameter<int32_t>("Nbinsx"), psJet_Vtx.getParameter<double>("xmin"), psJet_Vtx.getParameter<double>("xmax"));
  HTJets_vtx->setAxisTitle("Track Jet vertex position z [cm]", 1);
  HTJets_vtx->setAxisTitle("# Track Jets", 2);

  iBooker.setCurrentFolder(topFolderName_+"/TrackJets");

  //MHT
  HistoName = "MHT";
  edm::ParameterSet psMHT =  conf_.getParameter<edm::ParameterSet>("TH1_MHT");
  h_MHT = iBooker.book1D(HistoName, HistoName, psMHT.getParameter<int32_t>("Nbinsx"), psMHT.getParameter<double>("xmin"), psMHT.getParameter<double>("xmax"));
  h_MHT->setAxisTitle("Track Jet MHT [GeV]", 1);
  h_MHT->setAxisTitle("#Events", 2);

  //HT
  HistoName = "HT";
  edm::ParameterSet psHT =  conf_.getParameter<edm::ParameterSet>("TH1_HT");
  h_HT = iBooker.book1D(HistoName, HistoName, psHT.getParameter<int32_t>("Nbinsx"), psHT.getParameter<double>("xmin"), psHT.getParameter<double>("xmax"));
  h_HT->setAxisTitle("Track Jet HT [GeV]", 1);
  h_HT->setAxisTitle("#Events", 2);


} //end of method

DEFINE_FWK_MODULE(L1TPhase2OTTrackJetsandMHT);
