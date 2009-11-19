#include "DQM/Physics/plugins/DiLeptonsChecker.h"

DiLeptonsChecker::DiLeptonsChecker(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  dqmStore_ = edm::Service<DQMStore>().operator->();
  outputFileName_ = iConfig.getParameter<std::string>("outputFileName");
  
  labelTriggerResults_ = iConfig.getParameter<edm::InputTag>("labelTriggerResults");
  labelBeamSpot_       = iConfig.getParameter<edm::InputTag>( "labelBeamSpot"     );
  labelMuons_          = iConfig.getParameter<edm::InputTag>( "labelMuons"        );
  labelElectrons_      = iConfig.getParameter<edm::InputTag>( "labelElectrons"    );
  labelJets_           = iConfig.getParameter<edm::InputTag>( "labelJets"         );
  labelMETs_           = iConfig.getParameter<edm::InputTag>( "labelMETs"         );
  verbose_             = iConfig.getParameter<bool>         ( "verbose"           );
  
  electronIDLabel_= iConfig.getParameter<std::string>("labelElectronID") ;

  lookAtDiElectronsChannel_          = iConfig.getParameter<bool>("lookAtDiElectronsChannel");
  lookAtDiMuonsChannel_              = iConfig.getParameter<bool>("lookAtDiMuonsChannel");
  lookAtElectronMuonChannel_         = iConfig.getParameter<bool>("lookAtElectronMuonChannel");
  
  useJES_                            = iConfig.getParameter<bool>("useJES");
  jetCorrector_                      = iConfig.getParameter<std::string>   ("jetCorrector");
  
  
  
  //*****************************************
  //FIXME set configurable
  //*****************************************
  if(lookAtDiElectronsChannel_)  relativePath_ = std::string("DiElectronsChecker");
  if(lookAtDiMuonsChannel_)      relativePath_ = std::string("DiMuonsChecker");
  if(lookAtElectronMuonChannel_) relativePath_ = std::string("ElectronMuonDiLeptonsChecker");

  //Common tools: modules
  //plots before selection
  std::string label("_beforeSelection");
  jetChecker        = new JetChecker(iConfig,relativePath_,label);
  metChecker        = new MetChecker(iConfig,relativePath_,label);
  muonChecker       = new MuonChecker(iConfig,relativePath_,label);
  electronChecker   = new ElectronChecker(iConfig,relativePath_,label);
  kinematicsChecker = new KinematicsChecker(iConfig,relativePath_,label);
  
  //after trigger
  std::string labelTrigger("_afterTirggerSelection");
  jetCheckerTrigger        = new JetChecker(iConfig,relativePath_,labelTrigger);
  metCheckerTrigger        = new MetChecker(iConfig,relativePath_,labelTrigger);
  muonCheckerTrigger       = new MuonChecker(iConfig,relativePath_,labelTrigger);
  electronCheckerTrigger   = new ElectronChecker(iConfig,relativePath_,labelTrigger);
  kinematicsCheckerTrigger = new KinematicsChecker(iConfig,relativePath_,labelTrigger);
  
  std::string labelNonIsoLept("_afterNonIsoLeptSelection");
  jetCheckerNonIsoLept        = new JetChecker(iConfig,relativePath_,labelNonIsoLept);
  metCheckerNonIsoLept        = new MetChecker(iConfig,relativePath_,labelNonIsoLept);
  muonCheckerNonIsoLept       = new MuonChecker(iConfig,relativePath_,labelNonIsoLept);
  electronCheckerNonIsoLept   = new ElectronChecker(iConfig,relativePath_,labelNonIsoLept);
  kinematicsCheckerNonIsoLept = new KinematicsChecker(iConfig,relativePath_,labelNonIsoLept);
  
  std::string labelIsoLept("_afterIsoLeptSelection");
  jetCheckerIsoLept        = new JetChecker(iConfig,relativePath_,labelIsoLept);
  metCheckerIsoLept        = new MetChecker(iConfig,relativePath_,labelIsoLept);
  muonCheckerIsoLept       = new MuonChecker(iConfig,relativePath_,labelIsoLept);
  electronCheckerIsoLept   = new ElectronChecker(iConfig,relativePath_,labelIsoLept);
  kinematicsCheckerIsoLept = new KinematicsChecker(iConfig,relativePath_,labelIsoLept);
  
  std::string labelLeptPair("_afterLeptPairSelection");
  jetCheckerLeptPair        = new JetChecker(iConfig,relativePath_,labelLeptPair);
  metCheckerLeptPair        = new MetChecker(iConfig,relativePath_,labelLeptPair);
  muonCheckerLeptPair       = new MuonChecker(iConfig,relativePath_,labelLeptPair);
  electronCheckerLeptPair   = new ElectronChecker(iConfig,relativePath_,labelLeptPair);
  kinematicsCheckerLeptPair = new KinematicsChecker(iConfig,relativePath_,labelLeptPair);
  
  std::string labelInvM("_afterInvMSelection");
  jetCheckerInvM        = new JetChecker(iConfig,relativePath_,labelInvM);
  metCheckerInvM        = new MetChecker(iConfig,relativePath_,labelInvM);
  muonCheckerInvM       = new MuonChecker(iConfig,relativePath_,labelInvM);
  electronCheckerInvM   = new ElectronChecker(iConfig,relativePath_,labelInvM);
  kinematicsCheckerInvM = new KinematicsChecker(iConfig,relativePath_,labelInvM);
  
  std::string labelJet("_afterJetSelection");
  jetCheckerJet        = new JetChecker(iConfig,relativePath_,labelJet);
  metCheckerJet        = new MetChecker(iConfig,relativePath_,labelJet);
  muonCheckerJet       = new MuonChecker(iConfig,relativePath_,labelJet);
  electronCheckerJet   = new ElectronChecker(iConfig,relativePath_,labelJet);
  kinematicsCheckerJet = new KinematicsChecker(iConfig,relativePath_,labelJet);
  
  std::string labelMet("_afterMetSelection");
  jetCheckerMet        = new JetChecker(iConfig,relativePath_,labelMet);
  metCheckerMet        = new MetChecker(iConfig,relativePath_,labelMet);
  muonCheckerMet       = new MuonChecker(iConfig,relativePath_,labelMet);
  electronCheckerMet   = new ElectronChecker(iConfig,relativePath_,labelMet);
  kinematicsCheckerMet = new KinematicsChecker(iConfig,relativePath_,labelMet);

  std::string labelBtag("_afterBtagSelection");
  jetCheckerBtag        = new JetChecker(iConfig,relativePath_,labelBtag);
  metCheckerBtag        = new MetChecker(iConfig,relativePath_,labelBtag);
  muonCheckerBtag       = new MuonChecker(iConfig,relativePath_,labelBtag);
  electronCheckerBtag   = new ElectronChecker(iConfig,relativePath_,labelBtag);
  kinematicsCheckerBtag = new KinematicsChecker(iConfig,relativePath_,labelBtag);

  std::string labelDBtag("_afterDoubleBtagSelection");
  jetCheckerDBtag        = new JetChecker(iConfig,relativePath_,labelDBtag);
  metCheckerDBtag        = new MetChecker(iConfig,relativePath_,labelDBtag);
  muonCheckerDBtag       = new MuonChecker(iConfig,relativePath_,labelDBtag);
  electronCheckerDBtag   = new ElectronChecker(iConfig,relativePath_,labelDBtag);
  kinematicsCheckerDBtag = new KinematicsChecker(iConfig,relativePath_,labelDBtag);
  
  //Can declare many time the same module and use them for different selection
  //other plots ??

  //Configuration
  nofJets_                = iConfig.getParameter<int>   ( "NofJets"     );
  ptThrJets_              = iConfig.getParameter<double>( "PtThrJets"   );
  etaThrJets_             = iConfig.getParameter<double>( "EtaThrJets"  );
  eHThrJets_              = iConfig.getParameter<double>( "EHThrJets"   );
  ptThrMuons_             = iConfig.getParameter<double>( "PtThrMuons"  );
  etaThrMuons_            = iConfig.getParameter<double>( "EtaThrMuons" );
  muonRelIso_             = iConfig.getParameter<double>( "MuonRelIso"  );
  muonRelIsoCalo_         = iConfig.getParameter<double>( "MuonRelIsoCalo"  );
  muonRelIsoTrk_          = iConfig.getParameter<double>( "MuonRelIsoTrk"  );
  ptThrElectrons_         = iConfig.getParameter<double>( "PtThrElectrons"  );
  etaThrElectrons_        = iConfig.getParameter<double>( "EtaThrElectrons" );
  electronRelIsoTrk_      = iConfig.getParameter<double>( "ElectronRelIsoTrk"  );
  electronRelIsoCalo_     = iConfig.getParameter<double>( "ElectronRelIsoCalo"  );
  electronRelIso_         = iConfig.getParameter<double>( "ElectronRelIso"  );
  muonVetoEM_             = iConfig.getParameter<double>( "MuonVetoEM"  );
  muonVetoHad_            = iConfig.getParameter<double>( "MuonVetoHad" );
  muonD0Cut_              = iConfig.getParameter<double>( "MuonD0Cut"   );
  electronD0Cut_          = iConfig.getParameter<double>( "ElectronD0Cut"   );
  metCut_                 = iConfig.getParameter<double>( "metCut"      );
  chi2Cut_                = iConfig.getParameter<int>   ( "Chi2Cut"     );
  nofValidHits_           = iConfig.getParameter<int>   ( "NofValidHits");
  triggerPath_            = iConfig.getParameter<std::vector<std::string> >( "triggerPath" );
  luminosity_             = iConfig.getParameter<int>   ( "Luminosity"  );
  xsection_               = iConfig.getParameter<double>( "Xsection"    );
  deltaREMCut_            = iConfig.getParameter<double>( "deltaREMCut"    );
}


DiLeptonsChecker::~DiLeptonsChecker()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
   //delete dqmStore_;
   //delete jetChecker;
   //delete metChecker;
   //delete muonChecker;
   //delete kinematicsChecker;
}

double DiLeptonsChecker::ComputeNbEvent(MonitorElement* h, int bin){
	if(h->getBinContent(1)>0)return(h->getBinContent(bin+1)*xsection_*luminosity_/(h->getBinContent(1)));
	else return (0);
}

double DiLeptonsChecker::ComputeNbEventError(MonitorElement* h, int bin){
	if(h->getBinContent(1)>0)return(   (pow((double)(h->getBinContent(bin+1)), 0.5))*xsection_*luminosity_/(h->getBinContent(1))   );
	else return (0);
}

void
DiLeptonsChecker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //using namespace edm;
  
  int Ch = -1;
  if     (lookAtDiElectronsChannel_ ) Ch =1 ;
  else if(lookAtElectronMuonChannel_) Ch =2 ;
  else if(lookAtDiMuonsChannel_     ) Ch =4 ;

  //Here you handle the collection you want to access
  edm::Handle<edm::View<reco::GsfElectron> >  electronsHandle; 
  iEvent.getByLabel(labelElectrons_,electronsHandle);
  edm::View<reco::GsfElectron> electrons = *electronsHandle;
  
  //Read eID results
  edm::Handle<edm::ValueMap<float> >  eIDValueMap; 
  //Robust-Loose 
  iEvent.getByLabel( electronIDLabel_ , eIDValueMap ); 
  //const edm::ValueMap<float> & eIDmap = * eIDValueMap ;

  edm::Handle<edm::View<reco::Muon> >  muonsHandle; 
  iEvent.getByLabel(labelMuons_,muonsHandle);
  edm::View<reco::Muon> muons = *muonsHandle;
  
  edm::Handle<edm::View<reco::CaloJet> >  jetsHandle; 
  iEvent.getByLabel(labelJets_,jetsHandle);
  edm::View<reco::CaloJet> jets = *jetsHandle;
  
  edm::Handle<edm::View<reco::CaloMET> >  metsHandle; 
  iEvent.getByLabel(labelMETs_,metsHandle);
  edm::View<reco::CaloMET> mets = *metsHandle;
  std::vector<reco::CaloMET> vmets;
  for(unsigned int i=0;i<mets.size();i++) vmets.push_back(mets.at(i));
  
  edm::Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByLabel(labelBeamSpot_, beamSpotHandle);
  reco::BeamSpot beamSpot = * beamSpotHandle;
  
  edm::Handle<edm::TriggerResults> trigResults;
  iEvent.getByLabel(labelTriggerResults_,trigResults);
  
  ////////////////////////////////////////
  //Check if branches are available
  ////////////////////////////////////////
  if (!jetsHandle.isValid())      throw cms::Exception("ProductNotFound") <<"Jet collection not found"      <<std::endl;
  if (!electronsHandle.isValid()) throw cms::Exception("ProductNotFound") <<"Electron collection not found" <<std::endl;
  if (!muonsHandle.isValid())     throw cms::Exception("ProductNotFound") <<"Muon collection not found"     <<std::endl;
  if (!metsHandle.isValid())      throw cms::Exception("ProductNotFound") <<"MET collection not found"      <<std::endl;
  if (!beamSpotHandle.isValid())  throw cms::Exception("ProductNotFound") <<"BeamSpot not found"            <<std::endl;
  if (!trigResults.isValid())     throw cms::Exception("ProductNotFound") <<"Trigger results not found"     <<std::endl;
  ////////////////////////////////////////
  
  
  Selection* selection = new Selection();
  selection->Set(beamSpot, jets, muons, electrons, mets); 
  selection->SetConfiguration(ptThrJets_, etaThrJets_, eHThrJets_, ptThrMuons_, etaThrMuons_, muonRelIsoCalo_, muonRelIsoTrk_, muonVetoEM_, muonVetoHad_, ptThrElectrons_, etaThrElectrons_,  electronRelIsoCalo_,  electronRelIsoTrk_);
  selection->SetMuonConfig( muonD0Cut_, chi2Cut_, nofValidHits_);
  selection->SetElectronConfig( electronD0Cut_);
  
  const edm::ValueMap<float> & eIDmap = * eIDValueMap ;
  selection->SeteID( electrons, eIDmap );
  //selection->Setdxy( electrons );
  //selection->SetMuonAll(muons);
  
  //bool selected = selection->isSelected(nofJets_, string("muon"), 1);
  //bool selected = selection->isSelected(nofJets_, 1, 1);
  //means at least NofJets jets and at least 1 muon
  
  //No selection
  metChecker->analyze(vmets);
  jetChecker->analyze(selection->GetSelectedJets(), useJES_, iEvent, iSetup);
  if(!lookAtDiElectronsChannel_) muonChecker->analyze(selection->GetMuons());
  if(!lookAtDiMuonsChannel_)     electronChecker->analyze(selection->GetElectrons());
  kinematicsChecker->analyze(selection->GetSelectedJets(), vmets, selection->GetMuons(), selection->GetElectrons());
  
  histocontainer_["Selection"]->Fill(0);
  
  //trigger selection
  bool triggered = false;
  edm::TriggerNames triggerNames_;
  triggerNames_.init(*trigResults);
  for(unsigned int i=0; i<triggerNames_.triggerNames().size();i++){
    for(unsigned int j=0; j<triggerPath_.size(); j++){
      if(triggerNames_.triggerNames()[i] == triggerPath_[j]) {
	if(trigResults->accept(i)){
	  triggered = true;
	  break;
	}
      }
    } 
  }
  if(triggered) { 
    selection->SelectMuonsDiLeptSimpleSel();
    selection->SelectElectronsDiLeptSimpleeID();
    
    //Trigger selection
    if( (lookAtDiElectronsChannel_   && selection->GetElectrons().size() >=2) ||
	(lookAtDiMuonsChannel_       && selection->GetMuons().size()     >=2) ||
	(lookAtElectronMuonChannel_  && selection->GetMuons().size()     > 0 && selection->GetElectrons().size() >0 )
	){
      histocontainer_["Selection"]->Fill(1);   
      metCheckerTrigger->analyze(vmets);
      jetCheckerTrigger->analyze(selection->GetJets(), useJES_, iEvent, iSetup);
      if(!lookAtDiElectronsChannel_) muonCheckerTrigger->analyze(selection->GetMuons());
      if(!lookAtDiMuonsChannel_)     electronCheckerTrigger->analyze(selection->GetElectrons());
      kinematicsCheckerTrigger->analyze(selection->GetSelectedJets(), vmets, selection->GetMuons(), selection->GetElectrons());
    }
    
    selection->SelectMuonsDiLeptNonIso();
    selection->SelectElectronsDiLeptNonIsoeID();
    selection->RemoveElecClose2Mu(0.1);
    
    if( (lookAtDiElectronsChannel_   && selection->GetElectrons().size() >=2) ||
	(lookAtDiMuonsChannel_       && selection->GetMuons().size()     >=2) ||
	(lookAtElectronMuonChannel_  && selection->GetMuons().size()     > 0 && selection->GetElectrons().size() >0 )
	){
      metCheckerNonIsoLept->analyze(vmets);
      jetCheckerNonIsoLept->analyze(selection->GetJets(), useJES_, iEvent, iSetup);
      if(!lookAtDiElectronsChannel_) muonCheckerNonIsoLept->analyze(selection->GetMuons());
      if(!lookAtDiMuonsChannel_)     electronCheckerNonIsoLept->analyze(selection->GetElectrons());
      kinematicsCheckerNonIsoLept->analyze(selection->GetSelectedJets(), vmets, selection->GetMuons(), selection->GetElectrons());
      histocontainer_["Selection"]->Fill(2);
    }
    selection->SelectMuonsDiLeptIso();
    selection->SelectElectronsDiLeptIsoeID();
    
    if( (lookAtDiElectronsChannel_   && selection->GetElectrons().size() >=2) ||
	(lookAtDiMuonsChannel_       && selection->GetMuons().size()     >=2) ||
	(lookAtElectronMuonChannel_  && selection->GetMuons().size()     > 0 && selection->GetElectrons().size() >0 )
	){
      metCheckerIsoLept->analyze(vmets);
      jetCheckerIsoLept->analyze(selection->GetJets(), useJES_, iEvent, iSetup);
      if(!lookAtDiElectronsChannel_) muonCheckerIsoLept->analyze(selection->GetMuons());
      if(!lookAtDiMuonsChannel_)     electronCheckerIsoLept->analyze(selection->GetElectrons());
      kinematicsCheckerIsoLept->analyze(selection->GetSelectedJets(), vmets, selection->GetMuons(), selection->GetElectrons());
      histocontainer_["Selection"]->Fill(3); 
    }
    
    std::vector<reco::GsfElectron> selElectrons = selection->GetElectrons();
    std::vector<reco::Muon>        selMuons     = selection->GetMuons();
    
    bool selLeptonPairs = false;
    double invMass      = 0;
    bool mumu = false;
    bool emu  = false;
    bool ee   = false;
     
    std::pair<int, int> pairMuon;
    std::pair<int, int> pairElectron;
    std::pair<int, int> pairElectronMuon;
    
    double sumDiEl = 0;
    double sumDiMu = 0;
    double sumElMu = 0;
    
    for(unsigned int i=0; i< selElectrons.size(); i++){
      for(unsigned int j=0; j< selElectrons.size(); j++){
        double sumDiElTmp = (selElectrons)[i].pt() +(selElectrons)[j].pt() ;
	if(sumDiElTmp > sumDiEl && i != j){
	  sumDiEl = sumDiElTmp;
	  pairElectron.first = i;
	  pairElectron.second = j;
	}
      }
    }
    
    for(unsigned int i=0; i< selMuons.size(); i++){
      for(unsigned int j=0; j< selMuons.size(); j++){
        double sumDiMuTmp = (selMuons)[i].pt() +(selMuons)[j].pt() ;
	if(sumDiMuTmp > sumDiMu && i != j){
	  sumDiMu = sumDiMuTmp;
	  pairMuon.first = i;
	  pairMuon.second = j;
	}
      }
    }
    
    for(unsigned int i=0; i< selElectrons.size(); i++){
      for(unsigned int j=0; j< selMuons.size(); j++){
        double sumElMuTmp = (selElectrons)[i].pt() +(selMuons)[j].pt() ;
	if(sumElMuTmp > sumElMu){
	  sumElMu = sumElMuTmp;
	  pairElectronMuon.first = i;
	  pairElectronMuon.second = j;
	}
      }
    }
    
    if(sumDiEl > sumDiMu  && sumDiEl > sumElMu && selElectrons.size() >= 2                            ) ee   = true;
    if(sumDiMu > sumDiEl  && sumDiMu > sumElMu && selMuons.size()     >= 2                            ) mumu = true;
    if(sumElMu > sumDiEl  && sumElMu > sumDiMu && selMuons.size()     >=1  && selElectrons.size() >=1 ) emu  = true;
    
    if( lookAtDiElectronsChannel_  && ee ){
      selLeptonPairs = true;
      
      TLorentzVector v1 ;
      TLorentzVector v2 ;
      TLorentzVector v3 ; 
      v1.SetPtEtaPhiE((selElectrons)[pairElectron.first].pt() ,(selElectrons)[pairElectron.first].eta() ,(selElectrons)[pairElectron.first].phi() ,(selElectrons)[pairElectron.first].energy());
      v2.SetPtEtaPhiE((selElectrons)[pairElectron.second].pt(),(selElectrons)[pairElectron.second].eta(),(selElectrons)[pairElectron.second].phi(),(selElectrons)[pairElectron.second].energy());
      v3 = v1 + v2; 
      invMass= v3.M();
    }
    
    if(lookAtDiMuonsChannel_      && mumu){
      selLeptonPairs = true;
      
      TLorentzVector v1 ;
      TLorentzVector v2 ;
      TLorentzVector v3 ; 
      v1.SetPtEtaPhiE((selMuons)[pairMuon.first].pt() ,(selMuons)[pairMuon.first].eta() ,(selMuons)[pairMuon.first].phi() ,(selMuons)[pairMuon.first].energy());
      v2.SetPtEtaPhiE((selMuons)[pairMuon.second].pt(),(selMuons)[pairMuon.second].eta(),(selMuons)[pairMuon.second].phi(),(selMuons)[pairMuon.second].energy());
      v3 = v1 + v2; 
      invMass= v3.M();
    } 
    
    bool passZPeakInvMass = false;
    if(invMass < 76 || invMass > 106) passZPeakInvMass = true;
    
    if(lookAtElectronMuonChannel_ &&  emu)
      { 
	passZPeakInvMass = true;
	selLeptonPairs   = true;    
      }
    
     if(selLeptonPairs == true){
      const JetCorrector *acorrector = JetCorrector::getJetCorrector(jetCorrector_,iSetup);
      
      selection->SelectJets(iEvent, iSetup, acorrector);
      histocontainer_["Selection"]->Fill(4);
      
      metCheckerLeptPair->analyze(vmets);
      jetCheckerLeptPair->analyze(selection->GetJets(), useJES_, iEvent, iSetup);
      if(!lookAtDiElectronsChannel_) muonCheckerLeptPair->analyze(selection->GetMuons());
      if(!lookAtDiMuonsChannel_)     electronCheckerLeptPair->analyze(selection->GetElectrons());
      kinematicsCheckerLeptPair->analyze(selection->GetSelectedJets(), vmets, selection->GetMuons(), selection->GetElectrons());
      
      if(passZPeakInvMass){
        histocontainer_["Selection"]->Fill(5);	  
        metCheckerInvM->analyze(vmets);
        jetCheckerInvM->analyze(selection->GetJets(), useJES_, iEvent, iSetup);
        if(!lookAtDiElectronsChannel_) muonCheckerInvM->analyze(selection->GetMuons());
        if(!lookAtDiMuonsChannel_)     electronCheckerInvM->analyze(selection->GetElectrons());
        kinematicsCheckerInvM->analyze(selection->GetSelectedJets(), vmets, selection->GetMuons(), selection->GetElectrons()); 
	
      }
      
      //selection->RemoveJetClose2Muon(0.4);
      selection->RemoveJetClose2Electron(0.4);
      
      if( selection->GetJets().size()>0 && passZPeakInvMass) histocontainer_["Selection"]->Fill(6);    
      
      if(selection->GetJets().size()>1 && passZPeakInvMass){
	histocontainer_["Selection"]->Fill(7);
	metCheckerJet->analyze(vmets);
	jetCheckerJet->analyze(selection->GetSelectedJets(), useJES_, iEvent, iSetup);
	if(!lookAtDiElectronsChannel_) muonCheckerJet->analyze(selection->GetMuons());
	if(!lookAtDiMuonsChannel_)     electronCheckerJet->analyze(selection->GetElectrons());
	kinematicsCheckerJet->analyze(selection->GetSelectedJets(), vmets, selection->GetMuons(), selection->GetElectrons());
	
	if(vmets[0].pt() > metCut_){
	  histocontainer_["Selection"]->Fill(8);
	  
	  metCheckerMet->analyze(vmets);
	  jetCheckerMet->analyze(selection->GetJets(), useJES_, iEvent, iSetup);
	  if(!lookAtDiElectronsChannel_) muonCheckerMet->analyze(selection->GetMuons());
	  if(!lookAtDiMuonsChannel_)     electronCheckerMet->analyze(selection->GetElectrons());
	  kinematicsCheckerMet->analyze(selection->GetSelectedJets(), vmets, selection->GetMuons(), selection->GetElectrons());
	
	  //fixme add btagging information
	}//met
      }//jet
    }//lept
    
    
  }
  delete selection;
}

void 
DiLeptonsChecker::beginJob(const edm::EventSetup& es)
{
  //edm::Service<TFileService> fs;
  //if (!fs) throw edm::Exception(edm::errors::Configuration, "TFileService missing from configuration!");
  
  dqmStore_->setCurrentFolder( relativePath_ );
  
  //define the histograms booked
  //TH1D
  histocontainer_["Selection"] = dqmStore_->book1D("Selection" ,"Nof events selected ",12,0, 12);
  //modules
  
  metChecker->begin(es);
  jetChecker->begin(es, jetCorrector_ );
  muonChecker->begin(es);
  electronChecker->begin(es);
  kinematicsChecker->begin(es);
  
  metCheckerTrigger->begin(es);
  jetCheckerTrigger->begin(es, jetCorrector_ );
  muonCheckerTrigger->begin(es);
  electronCheckerTrigger->begin(es);
  kinematicsCheckerTrigger->begin(es);
  
  metCheckerNonIsoLept->begin(es);
  jetCheckerNonIsoLept->begin(es, jetCorrector_ );
  muonCheckerNonIsoLept->begin(es);
  electronCheckerNonIsoLept->begin(es);
  kinematicsCheckerNonIsoLept->begin(es);
  
  metCheckerIsoLept->begin(es);
  jetCheckerIsoLept->begin(es, jetCorrector_ );
  muonCheckerIsoLept->begin(es);
  electronCheckerIsoLept->begin(es);
  kinematicsCheckerIsoLept->begin(es);
  
  metCheckerLeptPair->begin(es);
  jetCheckerLeptPair->begin(es, jetCorrector_ );
  muonCheckerLeptPair->begin(es);
  electronCheckerLeptPair->begin(es);
  kinematicsCheckerLeptPair->begin(es);
  
  metCheckerInvM->begin(es);
  jetCheckerInvM->begin(es, jetCorrector_ );
  muonCheckerInvM->begin(es);
  electronCheckerInvM->begin(es);
  kinematicsCheckerInvM->begin(es);
  
  metCheckerJet->begin(es);
  jetCheckerJet->begin(es, jetCorrector_ );
  muonCheckerJet->begin(es);
  electronCheckerJet->begin(es);
  kinematicsCheckerJet->begin(es);
  
  metCheckerMet->begin(es);
  jetCheckerMet->begin(es, jetCorrector_ );
  muonCheckerMet->begin(es);
  electronCheckerMet->begin(es);
  kinematicsCheckerMet->begin(es);
  
  metCheckerBtag->begin(es);
  jetCheckerBtag->begin(es, jetCorrector_ );
  muonCheckerBtag->begin(es);
  electronCheckerBtag->begin(es);
  kinematicsCheckerBtag->begin(es);
  
  metCheckerDBtag->begin(es);
  jetCheckerDBtag->begin(es, jetCorrector_ );
  muonCheckerDBtag->begin(es);
  electronCheckerDBtag->begin(es);
  kinematicsCheckerDBtag->begin(es);
  //Can declare many time the same module and use them for different selectition
}

void 
DiLeptonsChecker::endJob() 
{
  //use LogError to summarise the error that happen in the execution (by example from warning) (ex: Nof where we cannot access such variable)
  //edm::LogError  ("SummaryError") << "My error message \n";    // or  edm::LogProblem  (not formated)
  //use LogInfo to summarise information (ex: pourcentage of events matched ...)
  
                                 edm::LogVerbatim ("MainResults") << " -------------------------------------------"<<std::endl;
                                 edm::LogVerbatim ("MainResults") << " -------------------------------------------"<<std::endl;
  if(lookAtDiMuonsChannel_)      edm::LogVerbatim ("MainResults") << " --       Report from di-muon channel     --"<<std::endl;
  if(lookAtDiElectronsChannel_)  edm::LogVerbatim ("MainResults") << " --     Report from di-electron channel   --"<<std::endl;
  if(lookAtElectronMuonChannel_) edm::LogVerbatim ("MainResults") << " --       Report from electron-muon       --"<<std::endl;
                                 edm::LogVerbatim ("MainResults") << " -------------------------------------------"<<std::endl;
                                 edm::LogVerbatim ("MainResults") << " -------------------------------------------"<<std::endl;
  
  //Write the main important numbers here ...
  //Selection table by example
  
  //modules
  
  metChecker->end();
  jetChecker->end();
  muonChecker->end();
  electronChecker->end();
  kinematicsChecker->end();
  
  metCheckerTrigger->end();
  jetCheckerTrigger->end();
  muonCheckerTrigger->end();
  electronCheckerTrigger->end();
  kinematicsCheckerTrigger->end();
  
  metCheckerNonIsoLept->end();
  jetCheckerNonIsoLept->end();
  muonCheckerNonIsoLept->end();
  electronCheckerNonIsoLept->end();
  kinematicsCheckerNonIsoLept->end();
  
  metCheckerIsoLept->end();
  jetCheckerIsoLept->end();
  muonCheckerIsoLept->end();
  electronCheckerIsoLept->end();
  kinematicsCheckerIsoLept->end();
  
  metCheckerLeptPair->end();
  jetCheckerLeptPair->end();
  muonCheckerLeptPair->end();
  electronCheckerLeptPair->end();
  kinematicsCheckerLeptPair->end();
  
  metCheckerInvM->end();
  jetCheckerInvM->end();
  muonCheckerInvM->end();
  electronCheckerInvM->end();
  kinematicsCheckerInvM->end();
  
  metCheckerJet->end();
  jetCheckerJet->end();
  muonCheckerJet->end();
  electronCheckerJet->end();
  kinematicsCheckerJet->end();
  
  metCheckerMet->end();
  jetCheckerMet->end();
  muonCheckerMet->end();
  electronCheckerMet->end();
  kinematicsCheckerMet->end();
  
  metCheckerBtag->end();
  jetCheckerBtag->end();
  muonCheckerBtag->end();
  electronCheckerBtag->end();
  kinematicsCheckerBtag->end();
  
  metCheckerDBtag->end();
  jetCheckerDBtag->end();
  muonCheckerDBtag->end();
  electronCheckerDBtag->end();
  kinematicsCheckerDBtag->end();
  
  //////////////////////////////
  //  Selection Table
  //////////////////////////////
  edm::LogVerbatim ("MainResults") <<        histocontainer_["Selection"]->getEntries()<<" entries"<<std::endl;
  edm::LogVerbatim ("MainResults") << " All events:                " << ComputeNbEvent( histocontainer_["Selection"] , 0) << " +/- " << ComputeNbEventError( histocontainer_["Selection"] , 0) << std::endl;
  edm::LogVerbatim ("MainResults") << " Trigger:                   " << ComputeNbEvent( histocontainer_["Selection"] , 1) << " +/- " << ComputeNbEventError( histocontainer_["Selection"] , 1) << std::endl;
  edm::LogVerbatim ("MainResults") << " 2 leptons :                " << ComputeNbEvent( histocontainer_["Selection"] , 2) << " +/- " << ComputeNbEventError( histocontainer_["Selection"] , 2) << std::endl;
  edm::LogVerbatim ("MainResults") << " 2 Iso lepts :              " << ComputeNbEvent( histocontainer_["Selection"] , 3) << " +/- " << ComputeNbEventError( histocontainer_["Selection"] , 3) << std::endl;
  edm::LogVerbatim ("MainResults") << " Leptons pair sel:          " << ComputeNbEvent( histocontainer_["Selection"] , 4) << " +/- " << ComputeNbEventError( histocontainer_["Selection"] , 4) << std::endl;
  edm::LogVerbatim ("MainResults") << " Inv Mass :                 " << ComputeNbEvent( histocontainer_["Selection"] , 5) << " +/- " << ComputeNbEventError( histocontainer_["Selection"] , 5) << std::endl;
  edm::LogVerbatim ("MainResults") << " >0 jet:                    " << ComputeNbEvent( histocontainer_["Selection"] , 6) << " +/- " << ComputeNbEventError( histocontainer_["Selection"] , 6) << std::endl;
  edm::LogVerbatim ("MainResults") << " >1 jet:                    " << ComputeNbEvent( histocontainer_["Selection"] , 7) << " +/- " << ComputeNbEventError( histocontainer_["Selection"] , 7) << std::endl;
  edm::LogVerbatim ("MainResults") << " met cut:                   " << ComputeNbEvent( histocontainer_["Selection"] , 8) << " +/- " << ComputeNbEventError( histocontainer_["Selection"] , 8) << std::endl;
  
  dqmStore_->save(outputFileName_);
}













//define this as a plug-in
DEFINE_FWK_MODULE(DiLeptonsChecker);
