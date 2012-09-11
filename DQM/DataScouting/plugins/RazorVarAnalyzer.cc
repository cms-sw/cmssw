#include "DQM/DataScouting/plugins/RazorVarAnalyzer.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include <TRegexp.h>
#include <TString.h>

#include <cmath>

//------------------------------------------------------------------------------
// A simple constructor which takes as inoput only the name of the PF jet collection
RazorVarAnalyzer::RazorVarAnalyzer( const edm::ParameterSet & conf ):
  ScoutingAnalyzerBase(conf),
  m_eleCollectionTag(conf.getUntrackedParameter<edm::InputTag>("eleCollectionName",edm::InputTag("hltPixelMatchElectronsActivity"))),
  m_jetCollectionTag(conf.getUntrackedParameter<edm::InputTag>("jetCollectionName",edm::InputTag("hltCaloJetIDPassed"))),
  m_muCollectionTag(conf.getUntrackedParameter<edm::InputTag>("muCollectionName",edm::InputTag("hltL3MuonCandidates"))),
  m_metCollectionTag(conf.getUntrackedParameter<edm::InputTag>("metCollectionName",edm::InputTag("hltMet"))),
  m_metCleanCollectionTag(conf.getUntrackedParameter<edm::InputTag>("metCleanCollectionName",edm::InputTag("hltMetClean"))),
  m_triggerCollectionTag(conf.getUntrackedParameter<edm::InputTag>("triggerCollectionName",edm::InputTag("TriggerResults","","HLT"))),
  m_razorVarCollectionTag(conf.getUntrackedParameter<edm::InputTag>("razorVarCollectionName")){
}

//------------------------------------------------------------------------------
// Nothing to destroy: the DQM service thinks about everything
RazorVarAnalyzer::~RazorVarAnalyzer(){}

bool RazorVarAnalyzer::triggerRegexp(const edm::TriggerResults& triggerResults, const edm::TriggerNames& triggerNames, const std::string& trigger) const{

  TString triggerName(trigger.c_str());
  TRegexp regexp(triggerName);

  bool result = false;

  // Store which triggers passed and failed.
  edm::TriggerNames::Strings const& names = triggerNames.triggerNames();
  for(edm::TriggerNames::Strings::const_iterator it = names.begin(); it != names.end(); ++it) {

    //match the name to the regexp
    TString name(it->c_str());
    if(name.Index(regexp) < 0){
      continue;
    }

    const unsigned int i = triggerNames.triggerIndex(*it);
    if(i >= triggerNames.size())
      continue;
    
    //more than one trigger can match, so we keep going until we have one that fired
    const bool triggered = triggerResults.wasrun(i) && triggerResults.accept(i);
    if(triggered){
      result = triggered;
      break;
    }  
  }
  return result;
}

//------------------------------------------------------------------------------
// Usual analyze method
void RazorVarAnalyzer::analyze( const edm::Event & iEvent, const edm::EventSetup & c ){

  //check for the noise filter
  edm::Handle<reco::CaloMETCollection> calomet_handle;
  iEvent.getByLabel(m_metCollectionTag,calomet_handle);
  edm::Handle<reco::CaloMETCollection> calomet_clean_handle;
  iEvent.getByLabel(m_metCleanCollectionTag,calomet_clean_handle);

  bool isNoiseEvent = false;
  if( calomet_handle.isValid() && calomet_clean_handle.isValid() ){
    isNoiseEvent = fabs(calomet_handle->front().pt() - calomet_clean_handle->front().pt()) > 0.1;
    std::cout << "isNoiseEvent: " << isNoiseEvent << std::endl;
  }


  //look at the trigger
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByLabel(m_triggerCollectionTag, triggerResults);

  const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);
  /*
   * Current trigger menu
   *
   * DST_HT250_v1
   * DST_L1HTT_Or_L1MultiJet_v1
   * DST_Mu5_HT250_v1
   * DST_Ele8_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_HT250_v1
  */
  const bool htTrigger = triggerRegexp(*triggerResults, triggerNames, "^DST_HT[0-9]+_v[0-9]+$");
  const bool muTrigger = triggerRegexp(*triggerResults, triggerNames, "^DST_Mu[0-9]+_HT[0-9]+_v[0-9]+$");
  const bool eleTrigger = triggerRegexp(*triggerResults, triggerNames, "^DST_Ele[0-9]+_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_HT[0-9]+_v[0-9]+$");

  const bool l1Trigger = triggerRegexp(*triggerResults, triggerNames, "^DST_L1HTT_Or_L1MultiJet_v[0-9]+$");
  const bool hltTrigger = htTrigger || muTrigger || eleTrigger;
  
  //count the number of jets with a minimal selection
  edm::Handle<reco::CaloJetCollection> calojets_handle;
  iEvent.getByLabel(m_jetCollectionTag,calojets_handle);
  
  unsigned int njets = 0;
  for(reco::CaloJetCollection::const_iterator it = calojets_handle->begin(); it != calojets_handle->end(); ++it){
    if(it->pt() >= 30. && fabs(it->eta()) <= 3.0){
      njets++;
    }
  }

  //count the number of muons
  edm::Handle<std::vector<reco::RecoChargedCandidate> > muon_handle;
  iEvent.getByLabel(m_muCollectionTag,muon_handle);

  unsigned int nmu_loose = 0;
  unsigned int nmu_tight = 0;
  if(muon_handle.isValid()){
    for(std::vector<reco::RecoChargedCandidate>::const_iterator it = muon_handle->begin(); it != muon_handle->end(); ++it){
      if(it->pt() >= 15 && fabs(it->eta()) <= 2.1) nmu_tight++;
      if(it->pt() >= 10 && fabs(it->eta()) <= 2.4) nmu_loose++;
    }
  }
  
  //count the number of electrons
  edm::Handle<reco::ElectronCollection> ele_handle;
  iEvent.getByLabel(m_eleCollectionTag,ele_handle);
  
  unsigned int nele_loose = 0;
  unsigned int nele_tight = 0;
  if(ele_handle.isValid()){
    for(reco::ElectronCollection::const_iterator it = ele_handle->begin(); it != ele_handle->end(); ++it){
      if(it->pt() >= 20 && fabs(it->eta()) <= 2.5) nele_tight++;
      if(it->pt() >= 10 && fabs(it->eta()) <= 2.5) nele_loose++;
    }
  }
  
  //now get the box number: {'MuEle':0,'MuMu':1,'EleEle':2,'Mu':3,'Ele':4,'Had':5}
  unsigned int box_num = 5;
  if(nmu_tight > 0 && nele_tight > 0){
    box_num = 0;
  }else if(nmu_tight > 0 && nmu_loose > 1){
    box_num = 1;
  }else if(nele_tight > 0 && nele_loose > 1){
    box_num = 2;
  }else if(nmu_tight > 0){
    box_num = 3;
  }else if(nele_tight > 0){
    box_num = 4;
  }

  edm::Handle<std::vector<double> > razorvar_handle;
  iEvent.getByLabel(m_razorVarCollectionTag,razorvar_handle);

  if(razorvar_handle->size() > 1){
    const double MR = razorvar_handle->at(0);
    const double R = razorvar_handle->at(1);
    
    //we only fill histograms if the noise filter is not fired
    if(isNoiseEvent){
      m_rsqMRFullyIncNoise->Fill(MR,R*R);
      return;
    }

    //fill the level 1 histogram
    if(l1Trigger){
      m_rsqMRFullyIncLevel1->Fill(MR,R*R);
      if(box_num == 3) m_rsqMRMuLevel1->Fill(MR,R*R);
      if(box_num == 4) m_rsqMREleLevel1->Fill(MR,R*R);
      if(box_num == 5) m_rsqMRHadLevel1->Fill(MR,R*R);
    }

    m_rsqMRFullyInc->Fill(MR,R*R);
    if(njets >= 4) m_rsqMRInc4J->Fill(MR,R*R);
    if(njets >= 6) m_rsqMRInc6J->Fill(MR,R*R);
    if(njets >= 8) m_rsqMRInc8J->Fill(MR,R*R);
    if(njets >= 10) m_rsqMRInc10J->Fill(MR,R*R);
    if(njets >= 12) m_rsqMRInc12J->Fill(MR,R*R);
    if(njets >= 14) m_rsqMRInc14J->Fill(MR,R*R);
    
    //now fill the boxes
    if(box_num == 0) m_rsqMREleMu->Fill(MR,R*R);
    if(box_num == 1) m_rsqMRMuMu->Fill(MR,R*R);
    if(box_num == 2) m_rsqMREleEle->Fill(MR,R*R);
    if(box_num == 3) m_rsqMRMu->Fill(MR,R*R);
    if(box_num == 4) m_rsqMREle->Fill(MR,R*R);
    if(box_num == 5) m_rsqMRHad->Fill(MR,R*R);

    //and the versions with the trigger requirement
    if(box_num == 3 && muTrigger) m_rsqMRMuHLT->Fill(MR,R*R);
    if(box_num == 4 && eleTrigger) m_rsqMREleHLT->Fill(MR,R*R);
    if(box_num == 5 && htTrigger) m_rsqMRHadHLT->Fill(MR,R*R);

    //finally the multijet boxes - think ttbar
    //muon boxes: muons are not in jets
    if( box_num == 3 && njets >= 4) m_rsqMRMuMJ->Fill(MR,R*R);
    //ele boxes: electrons are in jets
    else if( box_num == 4 && njets >= 5) m_rsqMREleMJ->Fill(MR,R*R);
    //fill the Had box
    else if( box_num == 5 && njets >= 6) m_rsqMRHadMJ->Fill(MR,R*R);
  }
  
}

void RazorVarAnalyzer::endRun( edm::Run const &, edm::EventSetup const & ){
}

//------------------------------------------------------------------------------
// Function to book the Monitoring Elements.
void RazorVarAnalyzer::bookMEs(){
  
  //the full inclusive histograms
  m_rsqMRFullyInc = bookH2withSumw2("rsqMRFullyInc",
				    "M_{R} vs R^{2} (All Events)",
				    400,0.,4000.,
				    50,0.,1.,
				    "M_{R} [GeV]",
				    "R^{2}");
  m_rsqMRInc4J = bookH2withSumw2("rsqMRInc4J",
				    "M_{R} vs R^{2} (>= 4j)",
				    400,0.,4000.,
				    50,0.,1.,
				    "M_{R} [GeV]",
				    "R^{2}");
  m_rsqMRInc6J = bookH2withSumw2("rsqMRInc6J",
				    "M_{R} vs R^{2} (>= 6j)",
				    400,0.,4000.,
				    50,0.,1.,
				    "M_{R} [GeV]",
				    "R^{2}");
  m_rsqMRInc8J = bookH2withSumw2("rsqMRInc8J",
				    "M_{R} vs R^{2} (>= 8j)",
				    400,0.,4000.,
				    50,0.,1.,
				    "M_{R} [GeV]",
				    "R^{2}");
  m_rsqMRInc10J = bookH2withSumw2("rsqMRInc10J",
				    "M_{R} vs R^{2} (>= 10j)",
				    400,0.,4000.,
				    50,0.,1.,
				    "M_{R} [GeV]",
				    "R^{2}");
  m_rsqMRInc12J = bookH2withSumw2("rsqMRInc12J",
				    "M_{R} vs R^{2} (>= 12j)",
				    400,0.,4000.,
				    50,0.,1.,
				    "M_{R} [GeV]",
				    "R^{2}");
  m_rsqMRInc14J = bookH2withSumw2("rsqMRInc14J",
				    "M_{R} vs R^{2} (>= 14j)",
				    400,0.,4000.,
				    50,0.,1.,
				    "M_{R} [GeV]",
				    "R^{2}");

  //the by box histograms
  m_rsqMREleMu = bookH2withSumw2("rsqMREleMu",
				 "M_{R} vs R^{2} (EleMu box)",
				 400,0.,4000.,
				 50,0.,1.,
				 "M_{R} [GeV]",
				 "R^{2}");
  m_rsqMRMuMu = bookH2withSumw2("rsqMRMuMu",
				"M_{R} vs R^{2} (MuMu box)",
				400,0.,4000.,
				50,0.,1.,
				"M_{R} [GeV]",
				"R^{2}");
  m_rsqMREleEle = bookH2withSumw2("rsqMREleEle",
				  "M_{R} vs R^{2} (EleEle box)",
				  400,0.,4000.,
				  50,0.,1.,
				  "M_{R} [GeV]",
				  "R^{2}");
  m_rsqMRMu = bookH2withSumw2("rsqMRMu",
			      "M_{R} vs R^{2} (Mu box)",
			      400,0.,4000.,
			      50,0.,1.,
			      "M_{R} [GeV]",
			      "R^{2}");
  m_rsqMREle = bookH2withSumw2("rsqMREle",
			       "M_{R} vs R^{2} (Ele box)",
			       400,0.,4000.,
			       50,0.,1.,
			       "M_{R} [GeV]",
			       "R^{2}");
  m_rsqMRHad = bookH2withSumw2("rsqMRHad",
				  "M_{R} vs R^{2} (Had box)",
				  400,0.,4000.,
				  50,0.,1.,
				  "M_{R} [GeV]",
				  "R^{2}");

  //the by box histograms
  m_rsqMRMuMJ = bookH2withSumw2("rsqMRMuMJ",
			      "M_{R} vs R^{2} (Mu box >= 4j)",
			      400,0.,4000.,
			      50,0.,1.,
			      "M_{R} [GeV]",
			      "R^{2}");
  m_rsqMREleMJ = bookH2withSumw2("rsqMREleMJ",
			       "M_{R} vs R^{2} (Ele box >= 5j)",
			       400,0.,4000.,
			       50,0.,1.,
			       "M_{R} [GeV]",
			       "R^{2}");
  m_rsqMRHadMJ = bookH2withSumw2("rsqMRHadMJ",
				 "M_{R} vs R^{2} (Had box >= 6j)",
				 400,0.,4000.,
				 50,0.,1.,
				 "M_{R} [GeV]",
				 "R^{2}");

  //calibration histograms
  m_rsqMRFullyIncNoise = bookH2withSumw2("rsqMRFullyIncNoise",
				    "M_{R} vs R^{2} (Noise Events)",
				    400,0.,4000.,
				    50,0.,1.,
				    "M_{R} [GeV]",
				    "R^{2}");
  m_rsqMRFullyIncLevel1 = bookH2withSumw2("rsqMRFullyIncLeve1",
				    "M_{R} vs R^{2} (Level 1 Events)",
				    400,0.,4000.,
				    50,0.,1.,
				    "M_{R} [GeV]",
				    "R^{2}");

  //trigger histograms
  m_rsqMRMuLevel1 = bookH2withSumw2("rsqMRMuLevel1",
			      "M_{R} vs R^{2} (Mu box - Level 1)",
			      400,0.,4000.,
			      50,0.,1.,
			      "M_{R} [GeV]",
			      "R^{2}");
  m_rsqMREleLevel1 = bookH2withSumw2("rsqMREleLevel1",
			       "M_{R} vs R^{2} (Ele box - Level 1)",
			       400,0.,4000.,
			       50,0.,1.,
			       "M_{R} [GeV]",
			       "R^{2}");
  m_rsqMRHadLevel1 = bookH2withSumw2("rsqMRHadLevel1",
				  "M_{R} vs R^{2} (Had box- Level 1)",
				  400,0.,4000.,
				  50,0.,1.,
				  "M_{R} [GeV]",
				  "R^{2}");

  m_rsqMRMuHLT = bookH2withSumw2("rsqMRMuHLT",
			      "M_{R} vs R^{2} (Mu box - Trigger)",
			      400,0.,4000.,
			      50,0.,1.,
			      "M_{R} [GeV]",
			      "R^{2}");
  m_rsqMREleHLT = bookH2withSumw2("rsqMREleHLT",
			       "M_{R} vs R^{2} (Ele box - Trigger)",
			       400,0.,4000.,
			       50,0.,1.,
			       "M_{R} [GeV]",
			       "R^{2}");
  m_rsqMRHadHLT = bookH2withSumw2("rsqMRHadHLT",
				  "M_{R} vs R^{2} (Had box - Trigger)",
				  400,0.,4000.,
				  50,0.,1.,
				  "M_{R} [GeV]",
				  "R^{2}");
  




}

