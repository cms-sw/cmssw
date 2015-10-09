#ifndef SUSY_HLT_VBF_Mu_H
#define SUSY_HLT_VBF_Mu_H

//event
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// MET
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"


//Muon
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

// Jets
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

// Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"




class SUSY_HLT_VBF_Mu: public DQMEDAnalyzer{
    
public:
    SUSY_HLT_VBF_Mu(const edm::ParameterSet& ps);
    virtual ~SUSY_HLT_VBF_Mu();
    
protected:
    void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
    void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
    void analyze(edm::Event const& e, edm::EventSetup const& eSetup);
    void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup) ;
    void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup);
    void endRun(edm::Run const& run, edm::EventSetup const& eSetup);
    
private:
    //histos booking function
    void bookHistos(DQMStore::IBooker &);
    
    //variables from config file
    edm::EDGetTokenT<reco::MuonCollection> theMuonCollection_;
    edm::EDGetTokenT<reco::PFJetCollection> thePfJetCollection_;
    edm::EDGetTokenT<reco::CaloJetCollection> theCaloJetCollection_;
    edm::EDGetTokenT<edm::TriggerResults> triggerResults_;
    edm::EDGetTokenT<trigger::TriggerEvent> theTrigSummary_;
    edm::EDGetTokenT<reco::PFMETCollection> thePfMETCollection_;    
    edm::EDGetTokenT<reco::CaloMETCollection> theCaloMETCollection_;    
    HLTConfigProvider fHltConfig;
    
    std::string HLTProcess_;
    std::string triggerPath_;
    edm::InputTag triggerMetFilter_;
    edm::InputTag triggerDiJetFilter_;
    edm::InputTag triggerHTFilter_;
    edm::InputTag triggerMuFilter_;
    edm::InputTag triggerCaloMETFilter_;
    double ptThrJet_;
    double etaThrJet_;
    double ptThrJetTrig_;
    double etaThrJetTrig_;
    double metCut_;
    double deltaetaVBFJets;
    double dijet ;
    double dijetOff ;
    double pfmetOnlinethreshold;
    double muonOnlinethreshold;
    double htOnlinethreshold;
    double mjjOnlinethreshold;
    // Histograms
    MonitorElement* h_triggerMuPt;
    MonitorElement* h_triggerMuEta;
    MonitorElement* h_triggerMuPhi;
    MonitorElement* h_triggerCaloMet;
    MonitorElement* h_triggerMet;
    MonitorElement* h_triggerMetPhi;
    MonitorElement* h_Met;
    MonitorElement* h_ht; 
    MonitorElement* h_DiJetMass;
    MonitorElement* h_den_muonpt;
    MonitorElement* h_num_muonpt;
    MonitorElement* h_den_muoneta;
    MonitorElement* h_num_muoneta;
    MonitorElement* h_den_mjj;
    MonitorElement* h_num_mjj;
    MonitorElement* h_den_ht;
    MonitorElement* h_num_ht;
    MonitorElement* h_den_met;
    MonitorElement* h_num_met;

};

#endif
