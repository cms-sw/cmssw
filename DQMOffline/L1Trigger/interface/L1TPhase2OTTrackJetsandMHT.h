#ifndef L1TPhase2_OTTrackJetsandMHT_h
#define L1TPhase2_OTTrackJetsandMHT_h

#include <vector>
#include <memory>
#include <string>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
// #include "DataFormats/L1TVertex/interface/Vertex.h"
#include "DataFormats/L1TCorrelator/interface/TkPrimaryVertex.h"

#include "DataFormats/L1TCorrelator/interface/TkJet.h"
#include "DataFormats/L1TCorrelator/interface/TkJetFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkHTMiss.h"
#include "DataFormats/L1TCorrelator/interface/TkHTMissFwd.h"



class DQMStore;

class L1TPhase2OTTrackJetsandMHT : public DQMEDAnalyzer {

public:
  explicit L1TPhase2OTTrackJetsandMHT(const edm::ParameterSet&);
  ~L1TPhase2OTTrackJetsandMHT() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  //All track jets
  MonitorElement* allJets_pt = nullptr; // pt of jet
  MonitorElement* allJets_eta = nullptr; // eta of jet
  MonitorElement* allJets_phi = nullptr; // phi of jet
  MonitorElement* allJets_vtx = nullptr; // vtx of jet
  MonitorElement* allJets_nTracks = nullptr; // num of tracks that went into jet
  MonitorElement* allJets_nTightTracks = nullptr; // num of tight tracks that went into jet
  MonitorElement* allJets_nDisplacedTracks = nullptr; // num of displaced tracks that went into jet
  MonitorElement* allJets_nTightDispTracks = nullptr; // num of tight displaced tracks that went into jet

  //          m_2ltrkjet_vz->push_back(jetIter->jetVtx());
          // m_2ltrkjet_ntracks->push_back(jetIter->ntracks());
          // m_2ltrkjet_phi->push_back(jetIter->phi());
          // m_2ltrkjet_eta->push_back(jetIter->eta());
          // m_2ltrkjet_pt->push_back(jetIter->pt());
          // m_2ltrkjet_p->push_back(jetIter->p());
          // m_2ltrkjet_nDisplaced->push_back(jetIter->nDisptracks());
          // m_2ltrkjet_nTight->push_back(jetIter->nTighttracks());
          // m_2ltrkjet_nTightDisplaced->push_back(jetIter->nTightDisptracks());


  //Jets used in HT and MHT (pT, eta, and NTracks cuts)
  MonitorElement* HTJets_pt = nullptr; // pt of jet
  MonitorElement* HTJets_eta = nullptr; // eta of jet
  MonitorElement* HTJets_phi = nullptr; // phi of jet
  MonitorElement* HTJets_vtx = nullptr; // vtx of jet
  MonitorElement* HTJets_nTracks = nullptr; // num of tracks that went into jet
  MonitorElement* HTJets_nTightTracks = nullptr; // num of tight tracks that went into jet
  MonitorElement* HTJets_nDisplacedTracks = nullptr; // num of displaced tracks that went into jet
  MonitorElement* HTJets_nTightDispTracks = nullptr; // num of tight displaced tracks that went into jet

  //MHT
  MonitorElement* h_MHT = nullptr; // MHT of event from track jets

  //HT
  MonitorElement* h_HT = nullptr; // HT of event from track jets


 private:
  edm::ParameterSet conf_;

  float jet_minPt;                // [GeV]
  float jet_maxEta;               // [rad]
  //const edm::EDGetTokenT< L1TkPrimaryVertexCollection > pvToken;
  edm::EDGetTokenT< l1t::TkJetCollection > jetToken;

  //const edm::EDGetTokenT< std::vector< l1t::L1TkJetParticle > > jetToken;


  unsigned int minNtracksHighPt;
  unsigned int minNtracksLowPt;


  std::string topFolderName_;
};
#endif
