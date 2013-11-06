#ifndef B2GDQM_H
#define B2GDQM_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

// Trigger stuff
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/Common/interface/Handle.h" 
#include "FWCore/Framework/interface/DataKeyTags.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include <DataFormats/EgammaCandidates/interface/GsfElectron.h>

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DQMServices/Core/interface/MonitorElement.h"

// ParticleFlow
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

// EGamma
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

// Muon
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/MuonReco/interface/MuonIsolation.h" 

// Tau
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"

// Jets
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"


// Photon
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

// MET
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"

//
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DQMStore;
 
class B2GDQM: public edm::EDAnalyzer{

public:

  B2GDQM(const edm::ParameterSet& ps);
  virtual ~B2GDQM();
  
protected:

  virtual void beginJob();
  virtual void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& eSetup);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;
  virtual void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
  virtual void endRun(edm::Run const& run, edm::EventSetup const& eSetup);
  virtual void endJob();
  
  //Diagnostic
  //virtual void analyzeMultiJetsTrigger(edm::Event const& e);
  
  

  virtual void analyzeEventInterpretation(edm::Event const& e, edm::EventSetup const& eSetup);
  

private:

  void bookHistos(DQMStore * bei );
  
  int nLumiSecs_;
  int nEvents_, irun, ievt;
  
  DQMStore* bei_;  
  HLTConfigProvider hltConfigProvider_;
  bool isValidHltConfig_;

  // Variables from config file
  edm::InputTag theTriggerResultsCollection;
  edm::EDGetTokenT<edm::TriggerResults> triggerToken_;
 
  edm::Handle<edm::TriggerResults> triggerResults_;

  

 
  std::vector<edm::InputTag> jetLabels_;
  std::vector< edm::EDGetTokenT< edm::View<reco::Jet> > > jetTokens_;
  edm::InputTag PFMETLabel_;
  edm::EDGetTokenT< std::vector<reco::PFMET> > PFMETToken_;

  
  ///////////////////////////
  // Parameters 
  ///////////////////////////

  std::vector<double> jetPtMins_;
 
  std::string PFJetCorService_;
  ///////////////////////////
  // Histograms
  ///////////////////////////
  std::vector<MonitorElement*> pfJet_pt;
  std::vector<MonitorElement*> pfJet_y;
  std::vector<MonitorElement*> pfJet_phi;
  std::vector<MonitorElement*> pfJet_m;
  std::vector<MonitorElement*> pfJet_chef;
  std::vector<MonitorElement*> pfJet_nhef;
  std::vector<MonitorElement*> pfJet_cemf;
  std::vector<MonitorElement*> pfJet_nemf;
  std::vector<MonitorElement*> boostedJet_subjetPt;
  std::vector<MonitorElement*> boostedJet_subjetY;
  std::vector<MonitorElement*> boostedJet_subjetPhi;
  std::vector<MonitorElement*> boostedJet_subjetM;
  std::vector<MonitorElement*> boostedJet_subjetN;
  std::vector<MonitorElement*> boostedJet_massDrop;
  std::vector<MonitorElement*> boostedJet_minMass;
  MonitorElement* pfMet_pt;
  MonitorElement* pfMet_phi;
  

 
};


#endif
