#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "HLTriggerOffline/SUSYBSM/interface/SUSY_HLT_VBF_Mu.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <assert.h>
#include <cstdlib>
#include "TLorentzVector.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <regex.h>
#include <stdio.h>

SUSY_HLT_VBF_Mu::SUSY_HLT_VBF_Mu(const edm::ParameterSet& ps)
{
  edm::LogInfo("SUSY_HLT_VBF_Mu") << "Constructor SUSY_HLT_VBF_Mu::SUSY_HLT_VBF_Mu " << std::endl;
  // Get parameters from configuration file
  theTrigSummary_ = consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("trigSummary"));
  theMuonCollection_ = consumes<reco::MuonCollection>(ps.getParameter<edm::InputTag>("MuonCollection"));
  thePfJetCollection_ = consumes<reco::PFJetCollection>(ps.getParameter<edm::InputTag>("pfJetCollection"));
  thePfMETCollection_ = consumes<reco::PFMETCollection>(ps.getParameter<edm::InputTag>("pfMETCollection"));
  theCaloJetCollection_ = consumes<reco::CaloJetCollection>(ps.getParameter<edm::InputTag>("caloJetCollection"));
  theCaloMETCollection_ = consumes<reco::CaloMETCollection>(ps.getParameter<edm::InputTag>("caloMETCollection"));
  triggerResults_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
  HLTProcess_ = ps.getParameter<std::string>("HLTProcess");
  triggerPath_ = ps.getParameter<std::string>("TriggerPath");
  triggerMuFilter_ = ps.getParameter<edm::InputTag>("TriggerFilterMuon");
  triggerHTFilter_  = ps.getParameter<edm::InputTag>("TriggerFilterHT");
  triggerMetFilter_ = ps.getParameter<edm::InputTag>("TriggerFilterMET");
  triggerDiJetFilter_ = ps.getParameter<edm::InputTag>("TriggerFilterMJJ");
  triggerCaloMETFilter_ = ps.getParameter<edm::InputTag>("TriggerFilterCaloMET");
  ptThrJetTrig_ = ps.getUntrackedParameter<double>("PtThrJetTrig");
  etaThrJetTrig_ = ps.getUntrackedParameter<double>("EtaThrJetTrig");
  ptThrJet_ = ps.getUntrackedParameter<double>("PtThrJet");
  etaThrJet_ = ps.getUntrackedParameter<double>("EtaThrJet");
  deltaetaVBFJets = ps.getUntrackedParameter<double>("DeltaEtaVBFJets");
  pfmetOnlinethreshold  = ps.getUntrackedParameter<double>("PFMetCutOnline");
  muonOnlinethreshold  = ps.getUntrackedParameter<double>("MuonCutOnline");
  htOnlinethreshold = ps.getUntrackedParameter<double>("HTCutOnline");
  mjjOnlinethreshold = ps.getUntrackedParameter<double>("MJJCutOnline");
  
}

SUSY_HLT_VBF_Mu::~SUSY_HLT_VBF_Mu()
{
  edm::LogInfo("SUSY_HLT_VBF_Mu") << "Destructor SUSY_HLT_VBF_Mu::~SUSY_HLT_VBF_Mu " << std::endl;
}

void SUSY_HLT_VBF_Mu::dqmBeginRun(edm::Run const &run, edm::EventSetup const &e)
{
  
  bool changed;
  
  if (!fHltConfig.init(run, e, HLTProcess_, changed)) {
    edm::LogError("SUSY_HLT_VBF_Mu") << "Initialization of HLTConfigProvider failed!!";
    return;
  }
  
  bool pathFound = false;
  const std::vector<std::string> allTrigNames = fHltConfig.triggerNames();
  for(size_t j = 0; j <allTrigNames.size(); ++j) {
    if(allTrigNames[j].find(triggerPath_) != std::string::npos) {
      pathFound = true;
    }
  }
  
  if(!pathFound) {
    LogDebug("SUSY_HLT_VBF_Mu") << "Path not found" << "\n";
    return;
  }

  edm::LogInfo("SUSY_HLT_VBF_Mu") << "SUSY_HLT_VBF_Mu::beginRun" << std::endl;
}

void SUSY_HLT_VBF_Mu::bookHistograms(DQMStore::IBooker & ibooker_, edm::Run const &, edm::EventSetup const &)
{
  edm::LogInfo("SUSY_HLT_VBF_Mu") << "SUSY_HLT_VBF_Mu::bookHistograms" << std::endl;
  //book at beginRun
  bookHistos(ibooker_);
}

void SUSY_HLT_VBF_Mu::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
				       edm::EventSetup const& context)
{
  edm::LogInfo("SUSY_HLT_VBF_Mu") << "SUSY_HLT_VBF_Mu::beginLuminosityBlock" << std::endl;
}



void SUSY_HLT_VBF_Mu::analyze(edm::Event const& e, edm::EventSetup const& eSetup){


  edm::LogInfo("SUSY_HLT_VBF_Mu") << "SUSY_HLT_VBF_Mu::analyze" << std::endl;
  
  //-------------------------------
  //--- Jets
  //-------------------------------
  
  edm::Handle<reco::PFJetCollection> pfJetCollection;
  e.getByToken (thePfJetCollection_,pfJetCollection);
  if ( !pfJetCollection.isValid() ){
    edm::LogError ("SUSY_HLT_VBF_Mu") << "invalid collection: PFJets" << "\n";
    return;
  }

  edm::Handle<reco::CaloJetCollection> caloJetCollection;
  e.getByToken (theCaloJetCollection_,caloJetCollection);
  if ( !caloJetCollection.isValid() ){
    edm::LogError ("SUSY_HLT_VBF_Mu") << "invalid collection: CaloJets" << "\n";
    return;
  }
    
  //-------------------------------
  //--- Muon
  //-------------------------------
  edm::Handle<reco::MuonCollection> MuonCollection;
  e.getByToken (theMuonCollection_, MuonCollection);
  if ( !MuonCollection.isValid() ){
    edm::LogError ("SUSY_HLT_VBF_Mu") << "invalid collection: Muons " << "\n";
    return;
  }
  
   
  //-------------------------------
  //--- MET
  //-------------------------------
  
  edm::Handle<reco::CaloMETCollection> caloMETCollection;
  e.getByToken(theCaloMETCollection_, caloMETCollection);                                                      
  if ( !caloMETCollection.isValid() ){
    edm::LogError ("SUSY_HLT_VBF_Mu") << "invalid collection: CaloMET" << "\n";
    return;
  }


  edm::Handle<reco::PFMETCollection> pfMETCollection;
  e.getByToken(thePfMETCollection_, pfMETCollection);
  if ( !pfMETCollection.isValid() ){
    edm::LogError ("SUSY_HLT_VBF_Mu") << "invalid collection: PFMET" << "\n";
    return;
  }
  //
  
  //-------------------------------
  //--- Trigger
  //-------------------------------
  edm::Handle<edm::TriggerResults> hltresults;
  e.getByToken(triggerResults_,hltresults);
  if(!hltresults.isValid()){
    edm::LogError ("SUSY_HLT_VBF_Mu") << "invalid collection: TriggerResults" << "\n";
    return;
  }
  edm::Handle<trigger::TriggerEvent> triggerSummary;
  e.getByToken(theTrigSummary_, triggerSummary);
  if(!triggerSummary.isValid()) {
    edm::LogError ("SUSY_HLT_VBF_Mu") << "invalid collection: TriggerSummary" << "\n";
    return;
  }
  
  
  
  //get online objects
  
  std::vector<float> ptMuon, etaMuon, phiMuon;
  size_t filterIndexMu = triggerSummary->filterIndex( triggerMuFilter_ );
  trigger::TriggerObjectCollection triggerObjects = triggerSummary->getObjects();
  if( !(filterIndexMu >= triggerSummary->sizeFilters()) ){
    const trigger::Keys& keys = triggerSummary->filterKeys( filterIndexMu );
    for( size_t j = 0; j < keys.size(); ++j ){
      trigger::TriggerObject foundObject = triggerObjects[keys[j]];
      if(fabs(foundObject.id()) == 13){ //It's a muon
	
	bool same= false;
	for(unsigned int x=0;x<ptMuon.size();x++){
	  if(fabs(ptMuon[x] - foundObject.pt()) < 0.01)
	    same = true;
	}
	
	if(!same){
	  h_triggerMuPt->Fill(foundObject.pt());
	  h_triggerMuEta->Fill(foundObject.eta());
	  h_triggerMuPhi->Fill(foundObject.phi());
	  ptMuon.push_back(foundObject.pt());
	  etaMuon.push_back(foundObject.eta());
	  phiMuon.push_back(foundObject.phi());
	}
      }
    }
  }
  

    
  //get online objects                                                                                                                                       
  
  size_t filterIndexMet = triggerSummary->filterIndex( triggerMetFilter_ );
  size_t dijetFilterIndex = triggerSummary->filterIndex( triggerDiJetFilter_ );
  
  if( !(filterIndexMet >= triggerSummary->sizeFilters()) ){
    const trigger::Keys& keys = triggerSummary->filterKeys( filterIndexMet );
    for( size_t j = 0; j < keys.size(); ++j ){
      trigger::TriggerObject foundObject = triggerObjects[keys[j]];
      h_triggerMet->Fill(foundObject.pt());
      h_triggerMetPhi->Fill(foundObject.phi());
    }
  }
  
  dijet = -1; 
  
  std::vector<float> ptJet,  etaJet, phiJet;
  if( !(dijetFilterIndex >= triggerSummary->sizeFilters()) ){
    const trigger::Keys& KEYS(triggerSummary->filterKeys(dijetFilterIndex));                                                     
    const  size_t nK(KEYS.size());
    const trigger::TriggerObjectCollection& TOC(triggerSummary->getObjects());
    
    for (size_t i=0; i < nK ; i++ ) {
      const trigger::TriggerObject& TO1(TOC[KEYS[i]]);
      if( TO1.pt() > ptThrJetTrig_ && fabs(TO1.eta()) < etaThrJetTrig_ ){
	
	// for dijet part
	for (size_t j=i; j < nK; j++) {
	  if( i < j ){
	    const trigger::TriggerObject& TO2(TOC[KEYS[j]]);
	    if( TO2.pt() > ptThrJetTrig_ && fabs(TO2.eta()) < etaThrJetTrig_  ){
	      double tmpdeta = fabs( TO1.eta() - TO2.eta() );
	      double tmpopposite = TO1.eta() * TO2.eta() ;
	      if( tmpdeta > deltaetaVBFJets && tmpopposite < 0){
		TLorentzVector j1 ( TO1.px(),  TO1.py(),  TO1.pz(),  TO1.energy());
		TLorentzVector j2 ( TO2.px(),  TO2.py(),  TO2.pz(),  TO2.energy());
		double tmpmass = ( j1 + j2 ).M();
		if( dijet < tmpmass ) {
           dijet = tmpmass ;
         }
	      }
	    }
	  }
	}
      }
    }
   h_DiJetMass->Fill(dijet);
  }
 
 
  size_t filterIndexCaloMET = triggerSummary->filterIndex( triggerCaloMETFilter_ );
  if( filterIndexCaloMET  < triggerSummary->sizeFilters() ) {
    const trigger::Keys& keys = triggerSummary->filterKeys( filterIndexCaloMET );
     if(keys.size() ) {
      float met_h = triggerObjects[ keys[0] ].pt();
      h_triggerCaloMet->Fill(met_h);
     } 
    }



  
  size_t filterIndexHt = triggerSummary->filterIndex( triggerHTFilter_ );
  if( filterIndexHt < triggerSummary->sizeFilters() ) {
    const trigger::Keys& keys = triggerSummary->filterKeys( filterIndexHt );
    if( keys.size() ) {
      float ht = triggerObjects[ keys[0] ].pt();
      h_ht->Fill( ht );
    }
  }
  
  
  
    bool hasFired = false;
    
    const edm::TriggerNames& trigNames = e.triggerNames(*hltresults);
    unsigned int numTriggers = trigNames.size();
    
    for( unsigned int hltIndex=0; hltIndex<numTriggers; ++hltIndex ){
      if (trigNames.triggerName(hltIndex).find(triggerPath_) != std::string::npos && hltresults->wasrun(hltIndex) && hltresults->accept(hltIndex)) hasFired = true;
    }
    
    
    
    
    //Matching the muon
    int indexOfMatchedMuon = -1;
    int offlineCounter = 0;
    for(reco::MuonCollection::const_iterator muon = MuonCollection->begin(); muon != MuonCollection->end()  ; ++muon) {
      for(size_t off_i = 0; off_i < ptMuon.size(); ++off_i) {
    if(reco::deltaR(muon->eta(),muon->phi(),etaMuon[off_i],phiMuon[off_i]) < 0.5) {
	  indexOfMatchedMuon = offlineCounter;
	  break;
	}
      }
      offlineCounter++;
    }
    
    
    
    float pfHT = 0.0;
    
    for (reco::PFJetCollection::const_iterator i_pfjet = pfJetCollection->begin(); i_pfjet != pfJetCollection->end(); ++i_pfjet){
      if (i_pfjet->pt() < ptThrJet_) continue;
      if (fabs(i_pfjet->eta()) > etaThrJet_) continue;
      pfHT += i_pfjet->pt();
    }

    // 
    
    dijetOff =-1;
    
    size_t jetCol = pfJetCollection->size();
    
    for (size_t i=0; i < jetCol ; i++ ) {
      
      if( pfJetCollection->at(i).pt() > ptThrJetTrig_ && fabs(pfJetCollection->at(i).eta()) < etaThrJetTrig_ ){
	
	for (size_t j=i; j <jetCol ; j++) {
	  
	  if( i < j ){
	    
	    if( pfJetCollection->at(j).pt() > ptThrJetTrig_ && fabs(pfJetCollection->at(j).eta()) < etaThrJetTrig_  ){
	      
	      double tmpdetaOff = fabs( pfJetCollection->at(i).eta() - pfJetCollection->at(j).eta() );
	      double tmpoppositeOff = pfJetCollection->at(i).eta() * pfJetCollection->at(j).eta() ;
	      if( tmpdetaOff > deltaetaVBFJets && tmpoppositeOff < 0){
		TLorentzVector j1Off ( pfJetCollection->at(i).px(),  pfJetCollection->at(i).py(),  pfJetCollection->at(i).pz(),  pfJetCollection->at(i).energy());
		TLorentzVector j2Off ( pfJetCollection->at(j).px(),  pfJetCollection->at(j).py(),  pfJetCollection->at(j).pz(),  pfJetCollection->at(j).energy());
		double tmpmassOff = ( j1Off + j2Off ).M();
		if( dijetOff < tmpmassOff ) dijetOff = tmpmassOff ;
	
	      }
	    }
	  }
	}
      }
    }
    

//  For trigger turn on curves

  // for muon
    if(indexOfMatchedMuon > -1 && (dijetOff > mjjOnlinethreshold) && (pfMETCollection->begin()->et() > pfmetOnlinethreshold) && (pfHT > htOnlinethreshold)) {
      h_den_muonpt->Fill(MuonCollection->at(indexOfMatchedMuon).pt());
      h_den_muoneta->Fill(MuonCollection->at(indexOfMatchedMuon).eta());
      if(hasFired) {
	h_num_muonpt->Fill(MuonCollection->at(indexOfMatchedMuon).pt());
	h_num_muoneta->Fill(MuonCollection->at(indexOfMatchedMuon).eta());
      }
    }
    
    
    // For MJJ
    if(indexOfMatchedMuon > -1 && (MuonCollection->at(indexOfMatchedMuon).pt() > muonOnlinethreshold) && (pfMETCollection->begin()->et() > pfmetOnlinethreshold) && (pfHT > htOnlinethreshold)) {
      h_den_mjj->Fill(dijetOff);
      if(hasFired) {
	h_num_mjj->Fill(dijetOff);
      }
    }
    
    // For HT
    //
    if(indexOfMatchedMuon > -1 && (MuonCollection->at(indexOfMatchedMuon).pt() > muonOnlinethreshold) && (pfMETCollection->begin()->et() > pfmetOnlinethreshold) && (dijetOff > mjjOnlinethreshold)) {
      
      h_den_ht->Fill(pfHT);
      if(hasFired) {
	h_num_ht->Fill(pfHT);
      }
    } 
    
    if(indexOfMatchedMuon > -1 && (dijetOff > mjjOnlinethreshold) && (pfHT > htOnlinethreshold) && (MuonCollection->at(indexOfMatchedMuon).pt() > muonOnlinethreshold)) {
      
      h_den_met->Fill(pfMETCollection->begin()->et());
      if(hasFired) {
	h_num_met->Fill(pfMETCollection->begin()->et());
      }
    }
    
}

void SUSY_HLT_VBF_Mu::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup)
{
    edm::LogInfo("SUSY_HLT_VBF_Mu") << "SUSY_HLT_VBF_Mu::endLuminosityBlock" << std::endl;
}


void SUSY_HLT_VBF_Mu::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
    edm::LogInfo("SUSY_HLT_VBF_Mu") << "SUSY_HLT_VBF_Mu::endRun" << std::endl;
}

void SUSY_HLT_VBF_Mu::bookHistos(DQMStore::IBooker & ibooker_)
{
    ibooker_.cd();
    ibooker_.setCurrentFolder("HLT/SUSYBSM/SUSY_HLT_VBF_Mu");
    
    //offline quantities
    
    //online quantities
    h_triggerMuPt = ibooker_.book1D("triggerMuPt", "Trigger Muon Pt; GeV", 50, 0.0, 500.0);
    h_triggerMuEta = ibooker_.book1D("triggerMuEta", "Trigger Muon Eta", 20, -3.0, 3.0);
    h_triggerMuPhi = ibooker_.book1D("triggerMuPhi", "Trigger Muon Phi", 20, -3.5, 3.5);

    h_triggerCaloMet = ibooker_.book1D("h_triggerCaloMet", "Trigger Calo MET; GeV", 20, 0.0, 500.0);

    h_ht = ibooker_.book1D("h_ht", "h_ht",30, 0.0, 1500.0 ); 

    h_triggerMet = ibooker_.book1D("triggerMet", "Trigger MET; GeV", 20, 0.0, 500.0);
    h_triggerMetPhi = ibooker_.book1D("triggerMetPhi", "Trigger MET Phi", 20, -3.5, 3.5);

    h_DiJetMass  = ibooker_.book1D("h_DiJetMass" , "h_DiJetMass", 500,0.0,5000.0);    
        
//num and den hists to be divided in harvesting step to make turn on curves

    h_den_muonpt = ibooker_.book1D("h_den_muonpt","h_den_muonpt", 50, 0.0, 500.0);
    h_num_muonpt = ibooker_.book1D("h_num_muonpt","h_num_muonpt", 50, 0.0, 500.0);

    
    h_den_muoneta = ibooker_.book1D("h_den_muoneta","h_den_muoneta", 20, -3.0, 3.0);
    h_num_muoneta = ibooker_.book1D("h_num_muoneta","h_num_muoneta", 20, -3.0, 3.0);
    
    h_den_mjj = ibooker_.book1D("h_den_mjj","h_den_mjj",500,0.0,5000.0);
    h_num_mjj = ibooker_.book1D("h_num_mjj","h_num_mjj",500,0.0,5000.0);
    
    h_den_met = ibooker_.book1D("h_den_met","h_den_met",20,0.0,500.0);
    h_num_met = ibooker_.book1D("h_num_met","h_num_met",20,0.0,500.0);
    
    h_den_ht = ibooker_.book1D("h_den_ht","h_den_ht",30,0.0,1500.0);
    h_num_ht = ibooker_.book1D("h_num_ht","h_num_ht",30,0.0,1500.0);


    ibooker_.cd();
}

//define this as a plug-in
DEFINE_FWK_MODULE(SUSY_HLT_VBF_Mu);
