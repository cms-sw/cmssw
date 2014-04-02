#ifndef DQMExample_Step1_H
#define DQMExample_Step1_H

//Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//event
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//Candidate handling
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

// Electron
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

// PFMET
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"

// Vertex utilities
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// CaloJets
#include "DataFormats/JetReco/interface/CaloJet.h"

// Conversions
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

// Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "FWCore/Common/interface/TriggerNames.h"

 
class DQMExample_Step1: public DQMEDAnalyzer{

public:

  DQMExample_Step1(const edm::ParameterSet& ps);
  virtual ~DQMExample_Step1();
  
protected:

  void beginJob();
  void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup);
  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup) ;
  void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup);
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);
  void endJob();

private:
  //histos booking function
  void bookHistos(DQMStore::IBooker &);

  //other functions
  bool MediumEle(const edm::Event & iEvent, const edm::EventSetup & iESetup, const reco::GsfElectron & electron);
  double Distance(const reco::Candidate & c1, const reco::Candidate & c2 );
  double DistancePhi(const reco::Candidate & c1, const reco::Candidate & c2 );
  double calcDeltaPhi(double phi1, double phi2);

  //private variables
  math::XYZPoint PVPoint_;

  //variables from config file
  edm::EDGetTokenT<reco::GsfElectronCollection> theElectronCollection_;
  edm::EDGetTokenT<reco::ConversionCollection> theConversionCollection_;
  edm::EDGetTokenT<reco::CaloJetCollection> theCaloJetCollection_;
  edm::EDGetTokenT<reco::PFMETCollection> thePfMETCollection_;
  edm::EDGetTokenT<reco::VertexCollection> thePVCollection_;
  edm::EDGetTokenT<reco::BeamSpot> theBSCollection_;
  edm::EDGetTokenT<trigger::TriggerEvent> triggerEvent_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResults_;
  edm::InputTag triggerFilter_;
  std::string triggerPath_;

  double ptThrL1_;
  double ptThrL2_;
  double ptThrJet_;
  double ptThrMet_;

  int nElectrons;
  int nBJets;

  // Histograms
  MonitorElement* h_vertex_number;

  MonitorElement* h_pfMet;

  MonitorElement* h_eMultiplicity;
  MonitorElement* h_ePt_leading;
  MonitorElement* h_eEta_leading;
  MonitorElement* h_ePhi_leading;
  MonitorElement* h_ePt_leading_matched;
  MonitorElement* h_eEta_leading_matched;
  MonitorElement* h_ePhi_leading_matched;

  MonitorElement* h_eMultiplicity_HLT;
  MonitorElement* h_ePt_leading_HLT;
  MonitorElement* h_eEta_leading_HLT;
  MonitorElement* h_ePhi_leading_HLT;
  MonitorElement* h_ePt_leading_HLT_matched;
  MonitorElement* h_eEta_leading_HLT_matched;
  MonitorElement* h_ePhi_leading_HLT_matched;

  MonitorElement* h_jMultiplicity;
  MonitorElement* h_jPt_leading;
  MonitorElement* h_jEta_leading;
  MonitorElement* h_jPhi_leading;
  
  MonitorElement* h_ePt_diff;
};


#endif
