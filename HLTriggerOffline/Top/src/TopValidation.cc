// -*- C++ -*-
//
// Package:    TopValidation
// Class:      TopValidation
// 
/**\class TopValidation TopValidation.cc DQM/TopValidation/src/TopValidation.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Patricia LOBELLE PARDO ()
//         Created:  Tue Sep 23 11:06:32 CEST 2008
// $Id: TopValidation.cc,v 1.1 2008/11/20 11:20:36 lobelle Exp $
//
//


# include "HLTriggerOffline/Top/interface/TopValidation.h"




TopValidation::TopValidation(const edm::ParameterSet& iConfig)

{
  
  
  
     inputTag_ = iConfig.getParameter<edm::InputTag>("TriggerResultsCollection"); 
  
     triggerTag_= iConfig.getUntrackedParameter<string>("DQMFolder","HLT/Top");

    outputFileName = iConfig.getParameter<std::string>("OutputFileName");
     outputMEsInRootFile = iConfig.getParameter<bool>("OutputMEsInRootFile");
  
 
      
}


TopValidation::~TopValidation()
{
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//
// member functions
//


// ------------ method called to for each event  ------------
void
TopValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
 
  // muon collection
  Handle<MuonCollection> muonsH;
  iEvent.getByLabel("muons", muonsH);
  
  // tracks collection
  Handle<TrackCollection> tracks;
  iEvent.getByLabel("ctfWithMaterialTracks", tracks);
    
 
    
  // get calo jet collection
  Handle<CaloJetCollection> jetsHandle;
  iEvent.getByLabel("iterativeCone5CaloJets", jetsHandle);
  
  // electron collection
  Handle<GsfElectronCollection> electronsH;
  iEvent.getByLabel("pixelMatchGsfElectrons",electronsH);
  
  // Trigger 
  Handle<TriggerResults> trh;
  try {iEvent.getByLabel(inputTag_,trh);} catch(...) {;}
  
  triggerNames_.init(*trh);
 
  HLT1MuonNonIso           = false;
  HLT1MuonIso              = false;
  HLT1ElectronRelaxed      = false;
  HLT1Electron             = false;
  HLT2MuonNonIso           = false;
  HLT2ElectronRelaxed      = false;
  HLTXElectronMuonRelaxed  = false;
  HLTXElectronMuon         = false;
  HLT1jet                  = false;
  
  // -----new paths
  
  HLT4jet30               = false;
  HLT1Electron15_NI       = false;
  HLT1Electron15_LI       = false;
 
 
  //------
  
  // Trigger Bits
  const unsigned int n_TriggerResults( trh.product()->size());
  for ( unsigned int itrig= 0 ; itrig < n_TriggerResults; ++itrig ){
  
  //  cout<<itrig<<"-->"<<triggerNames_.triggerName(itrig)<<endl;
    
      if ( triggerNames_.triggerName( itrig ) == "HLT_Mu15" ){
	if ( trh.product()->accept( itrig ) ) HLT1MuonNonIso=true;
      }
      if (triggerNames_.triggerName(itrig) == "HLT_IsoMu11"){
        if ( trh.product()->accept( itrig ) ) HLT1MuonIso=true;
      }
      if (triggerNames_.triggerName(itrig) == "HLT_IsoEle18_L1R"){
        if ( trh.product()->accept( itrig ) ) HLT1ElectronRelaxed=true;
      }
      if (triggerNames_.triggerName(itrig) == "HLT_IsoEle15_L1I"){
        if ( trh.product()->accept( itrig ) ) HLT1Electron=true;
      }
      if (triggerNames_.triggerName(itrig) == "HLT_DoubleMu3"){
        if ( trh.product()->accept( itrig ) ) HLT2MuonNonIso=true;
      }
      if (triggerNames_.triggerName(itrig) == "HLT_DoubleIsoEle12_L1R"){
        if ( trh.product()->accept( itrig ) ) HLT2ElectronRelaxed=true;
      }	
      if (triggerNames_.triggerName(itrig) == "HLT_IsoEle10_Mu10_L1R"){
        if ( trh.product()->accept( itrig ) ) HLTXElectronMuonRelaxed=true;
      }
      if (triggerNames_.triggerName(itrig) == "HLT_IsoEle8_IsoMu7"){
        if ( trh.product()->accept( itrig ) ) HLTXElectronMuon=true;
      }
      if (triggerNames_.triggerName(itrig) == "HLT1jet"){
        if ( trh.product()->accept( itrig ) ) HLT1jet=true;
      }
      
      //---- new paths
       if (triggerNames_.triggerName(itrig) == "HLT_QuadJet30"){
        if ( trh.product()->accept( itrig ) ) HLT4jet30=true;
      }
       if (triggerNames_.triggerName(itrig) == "HLT_Ele15_SW_L1R"){
        if ( trh.product()->accept( itrig ) ) HLT1Electron15_NI=true;
      }
       // if (triggerNames_.triggerName(itrig) == "HLT_LooseIsoEle15_SW_L1R"){
        if (triggerNames_.triggerName(itrig) == "HLT_LooseIsoEle15_LW_L1R"){
        if ( trh.product()->accept( itrig ) ) HLT1Electron15_LI=true;
      }
       
       
      
      //-----
  }

//l1eteff->Fill(1);

  //generation  info                                     
 
  bool topevent = false;
 
  int ntop = 0;
  int ngenel=0;
  int ngenmu=0;
  int ngentau=0;
  int ngenlep=0;
  int nmuaccept=0;
  int nelaccept=0;
 
  // Gen Particles Collection
  Handle <GenParticleCollection> genParticles;
   iEvent.getByLabel("genParticles", genParticles);
  
   for (size_t i=0; i < genParticles->size(); ++i){
    const Candidate & p = (*genParticles)[i];
    int id = p.pdgId();
    int st = p.status();
 
    
    if (abs(id) == 6 && st == 3) ntop++;
   
    if (st==3 && abs(id)==11) {
      ngenel++;
       if ( p.pt()>10 && fabs(p.eta())<2.4)   nelaccept++;
    }
    
    if (st==3 && abs(id)==13) {
      ngenmu++;      
      if ( p.pt()>10 && fabs(p.eta())<2.4)    nmuaccept++;     
    }
    
    if (st==3 && abs(id)==15)  ngentau++;
    if (st==3 && ( abs(id)==11 || abs(id)==13 || abs(id)==15)) ngenlep++;
    
  }
  if (ntop == 2) topevent = true; 
  
  nEvents++;
  

  if (topevent){
   
    nAccepted++;
//    Nttbar->Fill(1);  
    
    // event type 
       
//    if (ngenlep==2) {dilepEvent++;    Ndilep->    Fill(1);}
//    if (ngenlep==1) {semilepEvent++;  Nsemilep->  Fill(1);}
  //  if (ngenlep==0) {hadronicEvent++; Nhadronic-> Fill(1);}
    
   // if (ngenlep==1 && ngenmu==1)                     {  muEvent++;    Nsemimu-> Fill(1);}
   // if (ngenlep==1 && ngenel==1)                     {  elecEvent++;  Nsemiel-> Fill(1);}
   // if (ngenlep==1 && ngentau==1)                    {  tauEvent++;   Nsemitau->Fill(1);}  
   // if (ngenlep==2 && ngenel==2)                     {  dielEvent++;  Ndiel->   Fill(1);}
   // if (ngenlep==2 && ngenmu==2)                     {  dimuEvent++;  Ndimu->   Fill(1);}   
   // if (ngenlep==2 && ngenmu==1 && ngenel==1)        {  emuEvent++;   Nemu->    Fill(1);}
    //if (ngenlep==2 && ngentau==2)                    {  ditauEvent++; Nditau->  Fill(1);}
    //if (ngenlep==2 && ngenmu==1 && ngentau==1)       {  taumuEvent++; Ntaumu->  Fill(1);}
    //if (ngenlep==2  && ngenel==1 && ngentau==1)      {  tauelEvent++; Ntauel->  Fill(1);}
  
    
    // ** MUONS **

    //Muon Collection to use
    std::map<double,reco::Muon> muonMap; 
  
    for (size_t i = 0; i< muonsH->size(); ++i){    
      if ( (*muonsH)[i].isGlobalMuon()){  
      muonMap[(*muonsH)[i].pt()] = (*muonsH)[i];
       }
    }     

    //Muon selection
    bool TwoMuonsAccepted = false;
    int  numberMuonsPt20 = 0;
    std::vector<reco::Muon> muonPt;
    reco::Muon muon1,muon2;// Keep the 2 muons with higher Pt [ Pt > 15 GeV/c and |eta|<2.0 ]
  
    for( std::map<double,reco::Muon>::reverse_iterator rit=muonMap.rbegin(); rit!=muonMap.rend(); ++rit){
      muonPt.push_back( (*rit).second );
    }


     for(size_t i=0;i<muonPt.size();++i){
      if(muonPt[i].pt()>15 && fabs(muonPt[i].eta())<2.0){
      numberMuonsPt20++;
	if(numberMuonsPt20==1) muon1 = muonPt[i];
	if(numberMuonsPt20==2) muon2 = muonPt[i];
      }
    }

 
    if(  numberMuonsPt20>1 && (muon1.pt() >20 || muon2.pt()>20) && muon1.charge()*muon2.charge()<0 ) TwoMuonsAccepted = true;

    // ** ELECTRONS **
    
    //Electron Collection to use
    std::map<double,reco::GsfElectron> electronMap;
   
    for (size_t i = 0; i<electronsH->size();++i){
      electronMap[(*electronsH)[i].pt()] = (*electronsH)[i];
    }     
        
    //Electron selection
    bool TwoElectronsAccepted = false;
    int  numberElectronsPt20 = 0;
    std::vector<reco::GsfElectron> electronPt;
    reco::GsfElectron electron1, electron2;
  
    for( std::map<double,reco::GsfElectron>::reverse_iterator rit=electronMap.rbegin(); rit!=electronMap.rend(); ++rit){
      electronPt.push_back( (*rit).second );
    }

   
    for(size_t i=0;i!=electronPt.size();++i){
      if(electronPt[i].pt()>15 && fabs(electronPt[i].eta() )<2.0){
	numberElectronsPt20++;
	if (numberElectronsPt20==1) electron1=electronPt[i];
	if (numberElectronsPt20==2) electron2=electronPt[i];
	} 
    }
   
     if(  numberElectronsPt20>1 && (electron1.pt() >20 || electron2.pt()>20) &&
     electron1.charge()*electron2.charge()<0 ) TwoElectronsAccepted = true;
  
    //Jet Collection to use
    
    //MC corrected
 
    /*   const JetCorrector* corrector = JetCorrector::getJetCorrector ( "MCJetCorrectorMcone5", iSetup);
	 float et_jet_cor[100], et_jet_normal[100];
  
	 int jj=0;
	 for( CaloJetCollection::const_iterator cal = jets->begin(); cal != jets->end(); ++ cal)
	 {
	 double scale = corrector->correction (*cal);
	 double corPt=scale*cal->et();
	 et_jet_cor[jj] = corPt;
	 et_jet_normal[jj] = (*cal).et();
	 jj++;
	 } */
    
    // Raw jets
    const CaloJetCollection *jets = jetsHandle.product();
    CaloJetCollection::const_iterator jet;
    
    int n_jets_20=0;
    
    for (jet = jets->begin(); jet != jets->end(); jet++){    
      
      //if (fabs(etajet[i]) <2.4 && et_jet_cor[i] > 20) n_jets_20++;
      if (fabs(jet->eta()) <2.4 && jet->et() > 13) n_jets_20++;
      
    } 
    
    
    // offline selection
    
    bool offline_mu       = false;
    bool offline_dimu     = false;
    bool offline_el       = false;
    bool offline_diel     = false;
    bool offline_emu      = false;
    
    
    if ( numberMuonsPt20>0 && muon1.pt()>20 && n_jets_20>1)                                                 offline_mu=true;
    if ( TwoMuonsAccepted && n_jets_20>1)                                                                         offline_dimu=true;
    if ( numberElectronsPt20>0 && electron1.pt()>20 && n_jets_20>1)                                         offline_el=true;
    if ( TwoElectronsAccepted && n_jets_20>1)                                                                     offline_diel=true;
    if ( numberMuonsPt20>0 && numberElectronsPt20>0 && (muon1.pt()>20 || electron1.pt()>20) && (muon1.charge()!= electron1.charge()) && n_jets_20>1) offline_emu=true;
    
    
    // Efficiencies 
																						     
																						  
    /// **** tt->munubjjb *****    
    if ( ngenlep==1 && ngenmu==1 && nmuaccept==1){  //Select events within acceptance
      
      // Efficiencies wrt MC
     
efficiencies(acceptmu,hlt1muon_semimu,hlt1muoniso_semimu,hlt1elecrelax_semimu,hlt1elec_semimu,hlt2muon_semimu,hlt2elec_semimu,
hltelecmurelax_semimu,hltelecmu_semimu,hlt1jet_semimu,hlt4jet30_semimu,hlt1elec15_li_semimu,hlt1elec15_ni_semimu); 
      
      // Efficiencies wrt offline
      if (offline_mu){
	

efficiencies(Noffmu,hlt1muon_off_semimu,hlt1muoniso_off_semimu,hlt1elecrelax_off_semimu,hlt1elec_off_semimu,hlt2muon_off_semimu,hlt2elec_off_semimu,hltelecmurelax_off_semimu,hltelecmu_off_semimu,
hlt1jet_off_semimu,hlt4jet30_off_semimu,hlt1elec15_li_off_semimu, hlt1elec15_ni_off_semimu);
	
	eta_off_mu->Fill(muon1.eta()); 
       pt_off_mu-> Fill(muon1.pt());
	
	if(HLT1MuonNonIso){  
	  eta_trig_off_mu->Fill(muon1.eta());
	  pt_trig_off_mu->Fill(muon1.pt());
	}
      }
    }
    
    // *****  tt->enubjjb *****
    if ( ngenlep==1 && ngenel==1 && nelaccept==1){   
      
      // Efficiencies wrt MC
      efficiencies(acceptel,hlt1muon_semiel,hlt1muoniso_semiel,hlt1elecrelax_semiel,
      hlt1elec_semiel,hlt2muon_semiel,hlt2elec_semiel,
      hltelecmurelax_semiel,hltelecmu_semiel,
      hlt1jet_semiel,hlt4jet30_semiel,hlt1elec15_li_semiel, hlt1elec15_ni_semiel); 
      
      // Efficiencies wrt offline
      if (offline_el){
	

efficiencies(Noffel,hlt1muon_off_semiel,hlt1muoniso_off_semiel,hlt1elecrelax_off_semiel,hlt1elec_off_semiel,hlt2muon_off_semiel,hlt2elec_off_semiel,
hltelecmurelax_off_semiel,hltelecmu_off_semiel,
hlt1jet_off_semiel,hlt4jet30_off_semiel,hlt1elec15_li_off_semiel, hlt1elec15_ni_off_semiel);
	       
         eta_off_el->Fill(electron1.eta()); 
         pt_off_el->Fill(electron1.pt());
	
      	if(HLT1ElectronRelaxed){
	   eta_trig_off_el->Fill(electron1.eta());
	   pt_trig_off_el->Fill(electron1.pt());	  
	}
	
	//-----------
	if (HLT1Electron15_NI){
	eta_trig_off_el_ni->Fill(electron1.eta());
	pt_trig_off_el_ni->Fill(electron1.pt());
	}
	
	if (HLT1Electron15_LI){
 	eta_trig_off_el_li->Fill(electron1.eta());
	pt_trig_off_el_li->Fill(electron1.pt());
	}
	//-------	
      }    
    }

    // ****** tt->munubmunub *****
    if ( ngenlep==2 && ngenmu==2 && nmuaccept==2){  
      
      // Efficiencies wrt MC
      efficiencies(acceptdimu,hlt1muon_dimu,hlt1muoniso_dimu,hlt1elecrelax_dimu,
      hlt1elec_dimu,hlt2muon_dimu,hlt2elec_dimu,hltelecmurelax_dimu,hltelecmu_dimu,
      hlt1jet_dimu,hlt4jet30_dimu,hlt1elec15_li_dimu,hlt1elec15_ni_dimu); 
      
      // Efficiencies wrt offline
      if (offline_dimu){
	

  efficiencies(Noffdimu,hlt1muon_off_dimu,hlt1muoniso_off_dimu,hlt1elecrelax_off_dimu,hlt1elec_off_dimu,hlt2muon_off_dimu,hlt2elec_off_dimu,
   hltelecmurelax_off_dimu,hltelecmu_off_dimu,
   hlt1jet_off_dimu,hlt4jet30_off_dimu,hlt1elec15_li_off_dimu, hlt1elec15_ni_off_dimu);
        
	eta_off_dimu0->Fill(muon1.eta());
	eta_off_dimu1->Fill(muon2.eta());
 	pt_off_dimu0->Fill(muon1.pt());
 	pt_off_dimu1->Fill(muon2.pt());
	
       	if(HLT2MuonNonIso){
	  eta_trig_off_dimu0->Fill(muon1.eta()); 
	  eta_trig_off_dimu1->Fill(muon2.eta());
	  pt_trig_off_dimu0->Fill(muon1.pt());
	  pt_trig_off_dimu1->Fill(muon2.pt());
	}           
      } 
    }
    
    // *****   tt->enubenub *****
    if ( ngenlep==2 && ngenel==2 && nelaccept==2){   
      
      // Efficiencies wrt MC
      efficiencies(acceptdiel,hlt1muon_diel,hlt1muoniso_diel,hlt1elecrelax_diel,
      hlt1elec_diel,hlt2muon_diel,hlt2elec_diel, hltelecmurelax_diel,hltelecmu_diel, hlt1jet_diel,
      hlt4jet30_diel, hlt1elec15_li_diel, hlt1elec15_ni_diel); 
      
      // Efficiencies wrt offline
      if (offline_diel){
	

efficiencies(Noffdiel,hlt1muon_off_diel,hlt1muoniso_off_diel,hlt1elecrelax_off_diel,hlt1elec_off_diel,hlt2muon_off_diel,hlt2elec_off_diel,
hltelecmurelax_off_diel,hltelecmu_off_diel, hlt1jet_off_diel,hlt4jet30_off_diel,
hlt1elec15_li_off_diel, hlt1elec15_ni_off_diel);
	
	eta_off_diel0->Fill(electron1.eta()); 
	eta_off_diel1->Fill(electron2.eta());
	pt_off_diel0->Fill(electron1.pt());
	pt_off_diel1->Fill(electron2.pt());
	
       if(HLT2ElectronRelaxed){
	 eta_trig_off_diel0->Fill(electron1.eta()); 
	 eta_trig_off_diel1->Fill(electron2.eta());
	 pt_trig_off_diel0->Fill(electron1.pt());
	 pt_trig_off_diel1->Fill(electron2.pt());
       }
      }    
    }
    
    // *****  tt->enubmunub
    if ( ngenlep==2 && ngenel==1 && ngenmu==1 && nmuaccept==1 && nelaccept==1){   // tt->e mu events passing acceptance
      
      // Efficiencies wrt MC
      efficiencies(acceptemu,hlt1muon_emu,hlt1muoniso_emu,hlt1elecrelax_emu,
      hlt1elec_emu,hlt2muon_emu,hlt2elec_emu, hltelecmurelax_emu,hltelecmu_emu, hlt1jet_emu,
      hlt4jet30_emu, hlt1elec15_li_emu, hlt1elec15_ni_emu);
      
//          if (HLT1MuonNonIso || HLT1ElectronRelaxed ) OR_emu ->Fill(1);
//	  if (HLT1MuonNonIso || HLT1Electron15_NI )   OR_emu_ni ->Fill(1);
//	  if (HLT1MuonNonIso || HLT1Electron15_LI)    OR_emu_li ->Fill(1);
      
      // Efficiencies wrt offline      
      if (offline_emu){
	
//	    if (HLT1MuonNonIso || HLT1ElectronRelaxed ) OR_off_emu   ->Fill(1);
//	    if (HLT1MuonNonIso || HLT1Electron15_NI )   OR_off_emu_ni ->Fill(1);
//	    if (HLT1MuonNonIso || HLT1Electron15_LI)    OR_off_emu_li ->Fill(1);

efficiencies(Noffemu,hlt1muon_off_emu,hlt1muoniso_off_emu,hlt1elecrelax_off_emu,hlt1elec_off_emu,hlt2muon_off_emu,hlt2elec_off_emu,
hltelecmurelax_off_emu,hltelecmu_off_emu, hlt1jet_off_emu, hlt4jet30_off_emu, hlt1elec15_li_off_emu,
hlt1elec15_ni_off_emu);
	
       	eta_off_emu_muon->Fill(muon1.eta()); 
 	pt_off_emu_muon->Fill(muon1.pt());
 	eta_off_emu_electron->Fill(electron1.eta()); 
	pt_off_emu_electron->Fill(electron1.pt());
	
	if(HLTXElectronMuonRelaxed){
	  eta_trig_off_emu_muon->Fill(muon1.eta());  
	  pt_trig_off_emu_muon->Fill(muon1.pt());
	  eta_trig_off_emu_electron->Fill(electron1.eta()); 
	  pt_trig_off_emu_electron->Fill(electron1.pt());
	}
      }    
    }
    
    
    
    
 
    // End efficiencies  
    
  }
  
}
//////
void TopValidation::efficiencies (TH1F *select, TH1F *path1,TH1F *path2, TH1F *path3, TH1F *path4,
				 TH1F* path5, TH1F *path6, TH1F *path7, TH1F *path8,
				 TH1F* path9, TH1F* path10, TH1F* path11, TH1F* path12)
{
  
//  select->Fill(1);
  
  
 // if (HLT1MuonNonIso)          path1->Fill(1);
 // if (HLT1MuonIso)             path2->Fill(1);
  //if (HLT1ElectronRelaxed)     path3->Fill(1);
  //if (HLT1Electron)            path4->Fill(1);
 // if (HLT2MuonNonIso)          path5->Fill(1);
 // if (HLT2ElectronRelaxed)     path6->Fill(1);
 // if (HLTXElectronMuonRelaxed) path7->Fill(1);
 // if (HLTXElectronMuon)        path8->Fill(1);
 // if (HLT1jet)                 path9->Fill(1);
  
  //---- new paths
//  if (HLT4jet30)               path10->Fill(1);
 // if (HLT1Electron15_LI)       path11->Fill(1);
 // if (HLT1Electron15_NI)       path12->Fill(1);
  
  //------
  
  return;
 
}
 

 

//////

// ------------ method called once each job just before starting event loop  ------------
void 
TopValidation::beginJob(const edm::EventSetup&)
{
  
   dbe = edm::Service<DQMStore>().operator->();

	//dbe->setCurrentFolder("HLT/Top");
	dbe->setCurrentFolder(triggerTag_);
      //  l1eteff=dbe->book1D("prueba","prueba", 100,0.0,10.0);
  
 
 
  /////// histos pt, eta
  
   //semimu events 
    eta_trig_off_mu          = dbe->book1D ("eta_trig_off_mu","eta_trig_off_mu",51,-2.1,2.1);
    eta_off_mu               = dbe->book1D ("eta_off_mu","eta_off_mu",51,-2.1,2.1);
    pt_trig_off_mu           = dbe->book1D ("pt_trig_off_mu","pt_trig_off_mu",120,0.0,150.0);
    pt_off_mu                = dbe->book1D ("pt_off_mu","pt_off_mu",120,0.0,150.0);
    
 
     //semiel events  
    eta_trig_off_el          = dbe->book1D ("eta_trig_off_el","eta_trig_off_el",51,-2.1,2.1);
    eta_off_el               = dbe->book1D ("eta_off_el","eta_off_el",51,-2.1,2.1);
    pt_trig_off_el           = dbe->book1D ("pt_trig_off_el","pt_trig_off_el",120,0.0,150.0);
    pt_off_el                = dbe->book1D ("pt_off_el","pt_off_el",120,0.0,150.0);
    
     pt_trig_off_el_ni           = dbe->book1D ("pt_trig_off_el_ni","pt_trig_off_el_ni",120,0.0,150.0);
     pt_trig_off_el_li           = dbe->book1D ("pt_trig_off_el_li","pt_trig_off_el_li",120,0.0,150.0);
     eta_trig_off_el_ni          = dbe->book1D ("eta_trig_off_el_ni","eta_trig_off_el_ni",51,-2.1,2.1);
     eta_trig_off_el_li          = dbe->book1D ("eta_trig_off_el_li","eta_trig_off_el_li",51,-2.1,2.1);
 
    
    //dimu events
    eta_trig_off_dimu0       = dbe->book1D ("eta_trig_off_dimu0","eta_trig_off_dimu0",51,-2.1,2.1);
    eta_off_dimu0            = dbe->book1D ("eta_off_dimu0","eta_off_dimu0",51,-2.1,2.1);
    pt_trig_off_dimu0        = dbe->book1D ("pt_trig_off_dimu0","pt_trig_off_dimu0",120,0.0,150.0);
    pt_off_dimu0             = dbe->book1D ("pt_off_dimu0","pt_off_dimu0",120,0.0,150.0);
    
    eta_trig_off_dimu1       = dbe->book1D ("eta_trig_off_dimu1","eta_trig_off_dimu1",51,-2.1,2.1);
    eta_off_dimu1            = dbe->book1D ("eta_off_dimu1","eta_off_dimu1",51,-2.1,2.1);
    pt_trig_off_dimu1        = dbe->book1D ("pt_trig_off_dimu1","pt_trig_off_dimu1",120,0.0,150.0);
    pt_off_dimu1             = dbe->book1D ("pt_off_dimu1","pt_off_dimu1",120,0.0,150.0);
  
    
    //diel events
    
    eta_trig_off_diel0       = dbe->book1D ("eta_trig_off_diel0","eta_trig_off_diel0",51,-2.1,2.1);
    eta_off_diel0            = dbe->book1D ("eta_off_diel0","eta_off_diel0",51,-2.1,2.1);
    pt_trig_off_diel0        = dbe->book1D ("pt_trig_off_diel0","pt_trig_off_diel0",120,0.0,150.0);
    pt_off_diel0             = dbe->book1D ("pt_off_diel0","pt_off_diel0",120,0.0,150.0);
    
    eta_trig_off_diel1       = dbe->book1D ("eta_trig_off_diel1","eta_trig_off_diel1",51,-2.1,2.1);
    eta_off_diel1            = dbe->book1D ("eta_off_diel1","eta_off_diel1",51,-2.1,2.1);
    pt_trig_off_diel1        = dbe->book1D ("pt_trig_off_diel1","pt_trig_off_diel1",120,0.0,150.0);
    pt_off_diel1             = dbe->book1D ("pt_off_diel1","pt_off_diel1",120,0.0,150.0);
  
    
    //emu events
    
    eta_trig_off_emu_muon    = dbe->book1D ("eta_trig_off_emu_muon","eta_trig_off_emu_muon",51,-2.1,2.1);
    eta_off_emu_muon         = dbe->book1D ("eta_off_emu_muon","eta_off_emu_muon",51,-2.1,2.1);
    pt_trig_off_emu_muon     = dbe->book1D ("pt_trig_off_emu_muon","pt_trig_off_emu_muon",120,0.0,150.0);
    pt_off_emu_muon          = dbe->book1D ("pt_off_emu_muon","pt_off_emu_muon",120,0.0,150.0);
    
    eta_trig_off_emu_electron = dbe->book1D ("eta_trig_off_emu_electron","eta_trig_off_emu_electron",51,-2.1,2.1);
    eta_off_emu_electron      = dbe->book1D ("eta_off_emu_electron","eta_off_emu_electron",51,-2.1,2.1);
    pt_trig_off_emu_electron  = dbe->book1D ("pt_trig_off_emu_electron","pt_trig_off_emu_electron",120,0.0,150.0);
    pt_off_emu_electron       = dbe->book1D ("pt_off_emu_electron","pt_off_emu_electron",120,0.0,150.0);
  
  /*  OR_emu                    =   dbe->book1D ("or_emu","or_emu",10,0.0,2.0);
    OR_off_emu                =   dbe->book1D ("or_off_emu","or_off_emu",10,0.0,2.0);
    
    OR_emu_li                 =   dbe->book1D ("or_emu_li","or_emu_li",10,0.0,2.0);
    OR_off_emu_li             =   dbe->book1D ("or_off_emu_li","or_off_emu_li",10,0.0,2.0);
    
    OR_emu_ni                 =   dbe->book1D ("or_emu_ni","or_emu_ni",10,0.0,2.0);
    OR_off_emu_ni             =   dbe->book1D ("or_off_emu_ni","or_off_emu_ni",10,0.0,2.0);*/
 
  
 /* ////// histos event type  
 
    Nttbar                    = subDirEventType.make<TH1F> ("NttbarEvents","NttbarEvents",10,0.0,2.0);
  
    Nsemilep                  = subDirEventType.make<TH1F> ("NsemilepEvents","NsemilepEvents",10,0.0,2.0);
    Ndilep                    = subDirEventType.make<TH1F> ("NdilepEvents","NdilepEvents",10,0.0,2.0);
    Nhadronic                 = subDirEventType.make<TH1F> ("NhadronicEvents","NhadronicEvents",10,0.0,2.0);
    
    Nsemimu                   = subDirmu.make<TH1F>("NsemimuEvents","NsemimuEvents",10,0.0,2.0);
    Nsemiel                   = subDirel.make<TH1F>("NsemielEvents","NsemielEvents",10,0.0,2.0);
    Nsemitau                  = subDirEventType.make<TH1F>("NsemitauEvents","NsemitauEvents",10,0.0,2.0);
    
    Ndimu                     = subDirdimu.make<TH1F>("NdimuEvents","NdimuEvents",10,0.0,2.0);
    Ndiel                     = subDirdiel.make<TH1F>("NdielEvents","NdielEvents",10,0.0,2.0);
    Nemu                      = subDiremu.make<TH1F>("NemuEvents","NemuEvents",10,0.0,2.0);
    Ntaumu                    = subDirEventType.make<TH1F>("NtaumuEvents","NtaumuEvents",10,0.0,2.0);
    Ntauel                    = subDirEventType.make<TH1F>("NtauelEvents","NtauelEvents",10,0.0,2.0);
    Nditau                    = subDirEventType.make<TH1F> ("NditauEvents","NditauEvents",10,0.0,2.0);
    
 
    
    ///// histos #events within acceptance
    
    acceptmu                  = subDirmu.make<TH1F>("NsemimuAcceptedEvents","NsemimuAcceptedEvents",10,0.0,2.0);
    acceptel                  = subDirel.make<TH1F>("NsemielAcceptedEvents","NsemielAcceptedEvents",10,0.0,2.0);
    acceptdimu                = subDirdimu.make<TH1F>("NdimuAcceptedEvents","NdimuAcceptedEvents",10,0.0,2.0);
    acceptdiel                = subDirdiel.make<TH1F>("NdielAcceptedEvents","NdielAcceptedEvents",10,0.0,2.0);
    acceptemu                 = subDiremu.make<TH1F>("NemuAcceptedEvents","NemuAcceptedEvents",10,0.0,2.0);

   
    
  ///// histos EFFICIENCIES ///////
     
    ///semimu events    
    Noffmu                    =  subDirmu.make<TH1F> ("Noff_semimu","Noff_semimu",10,0.0,2.0); 
    hlt1muon_semimu           =  subDirmu.make<TH1F> ("hlt1muon_semimu","hlt1muon_semimu",10,0.0,2.0);
    hlt1muon_off_semimu       =  subDirmu.make<TH1F> ("hlt1muon_off_semimu","hl1muon_off_semimu",10,0.0,2.0);
    
    hlt1muoniso_semimu        =  subDirmu.make<TH1F> ("hlt1muoniso_semimu","hlt1muoniso_semimu",10,0.0,2.0); 
    hlt1muoniso_off_semimu    =  subDirmu.make<TH1F> ("hlt1muoniso_off_semimu","hl1muoniso_off_semimu",10,0.0,2.0);
    
    hlt2muon_semimu           =  subDirmu.make<TH1F> ("hlt2muon_semimu","hlt2muon_semimu",10,0.0,2.0);
    hlt2muon_off_semimu       =  subDirmu.make<TH1F> ("hlt2muon_off_semimu","hl2muon_off_semimu",10,0.0,2.0);
    
    hlt1elecrelax_semimu      =  subDirmu.make<TH1F> ("hlt1elecrelax_semimu","hlt1elecrelax_semimu",10,0.0,2.0);
    hlt1elecrelax_off_semimu  =  subDirmu.make<TH1F> ("hlt1elecrelax_off_semimu","hl1elecrelax_off_semimu",10,0.0,2.0);
    
    hlt1elec_semimu           =  subDirmu.make<TH1F> ("hlt1elec_semimu","hlt1elec_semimu",10,0.0,2.0);
    hlt1elec_off_semimu       =  subDirmu.make<TH1F> ("hlt1elec_off_semimu","hl1elec_off_semimu",10,0.0,2.0);
    
    hlt2elec_semimu           =  subDirmu.make<TH1F> ("hlt2elec_semimu","hlt2elec_semimu",10,0.0,2.0);
    hlt2elec_off_semimu       =  subDirmu.make<TH1F> ("hlt2elec_off_semimu","hl2elec_off_semimu",10,0.0,2.0);
    
    hltelecmu_semimu          =  subDirmu.make<TH1F> ("hltelecmu_semimu","hltelecmu_semimu",10,0.0,2.0); 
    hltelecmu_off_semimu      =  subDirmu.make<TH1F> ("hltelecmu_off_semimu","hlelecmu_off_semimu",10,0.0,2.0);
    
    hltelecmurelax_semimu     =  subDirmu.make<TH1F> ("hltelecmurelax_semimu","hltelecmurelax_semimu",10,0.0,2.0);
    hltelecmurelax_off_semimu =  subDirmu.make<TH1F> ("hltelecmurelax_off_semimu","hlelecmurelax_off_semimu",10,0.0,2.0);
    
    hlt1jet_semimu            =  subDirmu.make<TH1F> ("hlt1jet_semimu","hlt1jet_semimu",10,0.0,2.0); 
    hlt1jet_off_semimu        =  subDirmu.make<TH1F> ("hlt1jet_off_semimu","hl1jet_off_semimu",10,0.0,2.0);
    
    //----new paths
    hlt4jet30_semimu          =  subDirmu.make<TH1F> ("hlt4jet30_semimu","hlt4jet30_semimu",10,0.0,2.0);
    hlt1elec15_li_semimu      =  subDirmu.make<TH1F> ("hlt1elec15_li_semimu","hlt1elec15_li_semimu",10,0.0,2.0);
    hlt1elec15_ni_semimu      =  subDirmu.make<TH1F> ("hlt1elec15_ni_semimu","hlt1elec15_ni_semimu",10,0.0,2.0);
    
    hlt4jet30_off_semimu          =  subDirmu.make<TH1F> ("hlt4jet30_off_semimu","hlt4jet30_off_semimu",10,0.0,2.0);
    hlt1elec15_li_off_semimu      =  subDirmu.make<TH1F> ("hlt1elec15_li_off_semimu","hlt1elec15_li_off_semimu",10,0.0,2.0);
    hlt1elec15_ni_off_semimu      =  subDirmu.make<TH1F> ("hlt1elec15_ni_off_semimu","hlt1elec15_ni_off_semimu",10,0.0,2.0);
      
	  
     ///semiel events    
    Noffel                    =  subDirel.make<TH1F> ("Noff_semiel","Noff_semiel",10,0.0,2.0); 
    hlt1muon_semiel           =  subDirel.make<TH1F> ("hlt1muon_semiel","hlt1muon_semiel",10,0.0,2.0); 
    hlt1muon_off_semiel       =  subDirel.make<TH1F> ("hlt1muon_off_semiel","hl1muon_off_semiel",10,0.0,2.0);
    
    hlt1muoniso_semiel        =  subDirel.make<TH1F> ("hlt1muoniso_semiel","hlt1muoniso_semiel",10,0.0,2.0); 
    hlt1muoniso_off_semiel    =  subDirel.make<TH1F> ("hlt1muoniso_off_semiel","hl1muoniso_off_semiel",10,0.0,2.0);
    
    hlt2muon_semiel           =  subDirel.make<TH1F> ("hlt2muon_semiel","hlt2muon_semiel",10,0.0,2.0);  
    hlt2muon_off_semiel       =  subDirel.make<TH1F> ("hlt2muon_off_semiel","hl2muon_off_semiel",10,0.0,2.0);
    
    hlt1elecrelax_semiel      =  subDirel.make<TH1F> ("hlt1elecrelax_semiel","hlt1elecrelax_semiel",10,0.0,2.0);
    hlt1elecrelax_off_semiel  =  subDirel.make<TH1F> ("hlt1elecrelax_off_semiel","hl1elecrelax_off_semiel",10,0.0,2.0);
    
    hlt1elec_semiel           =  subDirel.make<TH1F> ("hlt1elec_semiel","hlt1elec_semiel",10,0.0,2.0); 
    hlt1elec_off_semiel       =  subDirel.make<TH1F> ("hlt1elec_off_semiel","hl1elec_off_semiel",10,0.0,2.0);
    
    hlt2elec_semiel           =  subDirel.make<TH1F> ("hlt2elec_semiel","hlt2elec_semiel",10,0.0,2.0); 
    hlt2elec_off_semiel       =  subDirel.make<TH1F> ("hlt2elec_off_semiel","hl2elec_off_semiel",10,0.0,2.0);
    
    hltelecmu_semiel          =  subDirel.make<TH1F> ("hltelecmu_semiel","hltelecmu_semiel",10,0.0,2.0);
    hltelecmu_off_semiel      =  subDirel.make<TH1F> ("hltelecmu_off_semiel","hlelecmu_off_semiel",10,0.0,2.0);
    
    hltelecmurelax_semiel     =  subDirel.make<TH1F> ("hltelecmurelax_semiel","hltelecmurelax_semiel",10,0.0,2.0);
    hltelecmurelax_off_semiel =  subDirel.make<TH1F> ("hltelecmurelax_off_semiel","hlelecmurelax_off_semiel",10,0.0,2.0);
    
    hlt1jet_semiel            =  subDirel.make<TH1F> ("hlt1jet_semiel","hlt1jet_semiel",10,0.0,2.0);
    hlt1jet_off_semiel        =  subDirel.make<TH1F> ("hlt1jet_off_semiel","hl1jet_off_semiel",10,0.0,2.0);
    
     //----new paths
    hlt4jet30_semiel          =  subDirel.make<TH1F> ("hlt4jet30_semiel","hlt4jet30_semiel",10,0.0,2.0);
    hlt1elec15_li_semiel      =  subDirel.make<TH1F> ("hlt1elec15_li_semiel","hlt1elec15_li_semiel",10,0.0,2.0);
    hlt1elec15_ni_semiel      =  subDirel.make<TH1F> ("hlt1elec15_ni_semiel","hlt1elec15_ni_semiel",10,0.0,2.0);
    
    hlt4jet30_off_semiel          =  subDirel.make<TH1F> ("hlt4jet30_off_semiel","hlt4jet30_off_semiel",10,0.0,2.0);
    hlt1elec15_li_off_semiel      =  subDirel.make<TH1F> ("hlt1elec15_li_off_semiel","hlt1elec15_li_off_semiel",10,0.0,2.0);
    hlt1elec15_ni_off_semiel      =  subDirel.make<TH1F> ("hlt1elec15_ni_off_semiel","hlt1elec15_ni_off_semiel",10,0.0,2.0);
      
 
      
    
     ///dimu. events    
    Noffdimu                  =   subDirdimu.make<TH1F> ("Noff_dimu","Noff_dimu",10,0.0,2.0); 
    hlt1muon_dimu             =   subDirdimu.make<TH1F> ("hlt1muon_dimu","hlt1muon_dimu",10,0.0,2.0);   
    hlt1muon_off_dimu         =   subDirdimu.make<TH1F> ("hlt1muon_off_dimu","hl1muon_off_dimu",10,0.0,2.0);
    
    hlt1muoniso_dimu          =   subDirdimu.make<TH1F> ("hlt1muoniso_dimu","hlt1muoniso_dimu",10,0.0,2.0); 
    hlt1muoniso_off_dimu      =   subDirdimu.make<TH1F> ("hlt1muoniso_off_dimu","hl1muoniso_off_dimu",10,0.0,2.0);
    
    hlt2muon_dimu             =   subDirdimu.make<TH1F> ("hlt2muon_dimu","hlt2muon_dimu",10,0.0,2.0);
    hlt2muon_off_dimu         =   subDirdimu.make<TH1F> ("hlt2muon_off_dimu","hl2muon_off_dimu",10,0.0,2.0);
    
    hlt1elecrelax_dimu        =   subDirdimu.make<TH1F> ("hlt1elecrelax_dimu","hlt1elecrelax_dimu",10,0.0,2.0);
    hlt1elecrelax_off_dimu    =   subDirdimu.make<TH1F> ("hlt1elecrelax_off_dimu","hl1elecrelax_off_dimu",10,0.0,2.0);
    
    hlt1elec_dimu             =   subDirdimu.make<TH1F> ("hlt1elec_dimu","hlt1elec_dimu",10,0.0,2.0);  
    hlt1elec_off_dimu         =   subDirdimu.make<TH1F> ("hlt1elec_off_dimu","hl1elec_off_dimu",10,0.0,2.0);
    
    hlt2elec_dimu             =   subDirdimu.make<TH1F> ("hlt2elec_dimu","hlt2elec_dimu",10,0.0,2.0);
    hlt2elec_off_dimu         =   subDirdimu.make<TH1F> ("hlt2elec_off_dimu","hl2elec_off_dimu",10,0.0,2.0);
    
    hltelecmu_dimu            =   subDirdimu.make<TH1F> ("hltelecmu_dimu","hltelecmu_dimu",10,0.0,2.0);
    hltelecmu_off_dimu        =   subDirdimu.make<TH1F> ("hltelecmu_off_dimu","hlelecmu_off_dimu",10,0.0,2.0);
    
    hltelecmurelax_dimu       =   subDirdimu.make<TH1F> ("hltelecmurelax_dimu","hltelecmurelax_dimu",10,0.0,2.0);
    hltelecmurelax_off_dimu   =   subDirdimu.make<TH1F> ("hltelecmurelax_off_dimu","hlelecmurelax_off_dimu",10,0.0,2.0);
    
    hlt1jet_dimu              =   subDirdimu.make<TH1F> ("hlt1jet_dimu","hlt1jet_dimu",10,0.0,2.0); 
    hlt1jet_off_dimu          =   subDirdimu.make<TH1F> ("hlt1jet_off_dimu","hl1jet_off_dimu",10,0.0,2.0);
    
     //----new paths
    hlt4jet30_dimu          =   subDirdimu.make<TH1F> ("hlt4jet30_dimu","hlt4jet30_dimu",10,0.0,2.0);
    hlt1elec15_li_dimu      =   subDirdimu.make<TH1F> ("hlt1elec15_li_dimu","hlt1elec15_li_dimu",10,0.0,2.0);
    hlt1elec15_ni_dimu      =   subDirdimu.make<TH1F> ("hlt1elec15_ni_dimu","hlt1elec15_ni_dimu",10,0.0,2.0);
    
    hlt4jet30_off_dimu          =  subDirdimu.make<TH1F> ("hlt4jet30_off_dimu","hlt4jet30_off_dimu",10,0.0,2.0);
    hlt1elec15_li_off_dimu      =  subDirdimu.make<TH1F> ("hlt1elec15_li_off_dimu","hlt1elec15_li_off_dimu",10,0.0,2.0);
    hlt1elec15_ni_off_dimu      =  subDirdimu.make<TH1F> ("hlt1elec15_ni_off_dimu","hlt1elec15_ni_off_dimu",10,0.0,2.0);
      
    
      
       ///diel events    
    Noffdiel                  =   subDirdiel.make<TH1F> ("Noff_diel","Noff_diel",10,0.0,2.0); 
    hlt1muon_diel             =   subDirdiel.make<TH1F> ("hlt1muon_diel","hlt1muon_diel",10,0.0,2.0);
    hlt1muon_off_diel         =   subDirdiel.make<TH1F> ("hlt1muon_off_diel","hl1muon_off_diel",10,0.0,2.0);   
   
    hlt1muoniso_diel          =   subDirdiel.make<TH1F> ("hlt1muoniso_diel","hlt1muoniso_diel",10,0.0,2.0);  
    hlt1muoniso_off_diel      =   subDirdiel.make<TH1F> ("hlt1muoniso_off_diel","hl1muoniso_off_diel",10,0.0,2.0);   
 
    hlt2muon_diel             =   subDirdiel.make<TH1F> ("hlt2muon_diel","hlt2muon_diel",10,0.0,2.0); 
    hlt2muon_off_diel         =   subDirdiel.make<TH1F> ("hlt2muon_off_diel","hl2muon_off_diel",10,0.0,2.0);  
    
    hlt1elecrelax_diel        =   subDirdiel.make<TH1F> ("hlt1elecrelax_diel","hlt1elecrelax_diel",10,0.0,2.0);
    hlt1elecrelax_off_diel    =   subDirdiel.make<TH1F> ("hlt1elecrelax_off_diel","hl1elecrelax_off_diel",10,0.0,2.0);  
  
    hlt1elec_diel             =   subDirdiel.make<TH1F> ("hlt1elec_diel","hlt1elec_diel",10,0.0,2.0);
    hlt1elec_off_diel         =   subDirdiel.make<TH1F> ("hlt1elec_off_diel","hl1elec_off_diel",10,0.0,2.0);
    
    hlt2elec_diel             =   subDirdiel.make<TH1F> ("hlt2elec_diel","hlt2elec_diel",10,0.0,2.0);
    hlt2elec_off_diel         =   subDirdiel.make<TH1F> ("hlt2elec_off_diel","hl2elec_off_diel",10,0.0,2.0);
    
    hltelecmu_diel            =   subDirdiel.make<TH1F> ("hltelecmu_diel","hltelecmu_diel",10,0.0,2.0); 
    hltelecmu_off_diel        =   subDirdiel.make<TH1F> ("hltelecmu_off_diel","hlelecmu_off_diel",10,0.0,2.0);
    
    hltelecmurelax_diel       =   subDirdiel.make<TH1F> ("hltelecmurelax_diel","hltelecmurelax_diel",10,0.0,2.0);  
    hltelecmurelax_off_diel   =   subDirdiel.make<TH1F> ("hltelecmurelax_off_diel","hlelecmurelax_off_diel",10,0.0,2.0);
    
    hlt1jet_diel              =   subDirdiel.make<TH1F> ("hlt1jet_diel","hlt1jet_diel",10,0.0,2.0);
    hlt1jet_off_diel          =   subDirdiel.make<TH1F> ("hlt1jet_off_diel","hl1jet_off_diel",10,0.0,2.0);
    
     //----new paths
    hlt4jet30_diel          =  subDirdiel.make<TH1F> ("hlt4jet30_diel","hlt4jet30_diel",10,0.0,2.0);
    hlt1elec15_li_diel      =  subDirdiel.make<TH1F> ("hlt1elec15_li_diel","hlt1elec15_li_diel",10,0.0,2.0);
    hlt1elec15_ni_diel      =  subDirdiel.make<TH1F> ("hlt1elec15_ni_diel","hlt1elec15_ni_diel",10,0.0,2.0);
    
    hlt4jet30_off_diel          =  subDirdiel.make<TH1F> ("hlt4jet30_off_diel","hlt4jet30_off_diel",10,0.0,2.0);
    hlt1elec15_li_off_diel      =  subDirdiel.make<TH1F> ("hlt1elec15_li_off_diel","hlt1elec15_li_off_diel",10,0.0,2.0);
    hlt1elec15_ni_off_diel      =  subDirdiel.make<TH1F> ("hlt1elec15_ni_off_diel","hlt1elec15_ni_off_diel",10,0.0,2.0);
      
    
    
       ///emu events    
    Noffemu                   =   subDiremu.make<TH1F> ("Noff_emu","Noff_emu",10,0.0,2.0); 
   
    hlt1muon_emu              =   subDiremu.make<TH1F> ("hlt1muon_emu","hlt1muon_emu",10,0.0,2.0);
    hlt1muon_off_emu          =   subDiremu.make<TH1F> ("hlt1muon_off_emu","hl1muon_off_emu",10,0.0,2.0);
    
    hlt1muoniso_emu           =   subDiremu.make<TH1F> ("hlt1muoniso_emu","hlt1muoniso_emu",10,0.0,2.0);
    hlt1muoniso_off_emu       =   subDiremu.make<TH1F> ("hlt1muoniso_off_emu","hl1muoniso_off_emu",10,0.0,2.0);
    
    hlt2muon_emu              =   subDiremu.make<TH1F> ("hlt2muon_emu","hlt2muon_emu",10,0.0,2.0);
    hlt2muon_off_emu          =   subDiremu.make<TH1F> ("hlt2muon_off_emu","hl2muon_off_emu",10,0.0,2.0);
    
    hlt1elecrelax_emu         =   subDiremu.make<TH1F> ("hlt1elecrelax_emu","hlt1elecrelax_emu",10,0.0,2.0);   
    hlt1elecrelax_off_emu     =   subDiremu.make<TH1F> ("hlt1elecrelax_off_emu","hl1elecrelax_off_emu",10,0.0,2.0);
    
    hlt1elec_emu              =   subDiremu.make<TH1F> ("hlt1elec_emu","hlt1elec_emu",10,0.0,2.0);
    hlt1elec_off_emu          =   subDiremu.make<TH1F> ("hlt1elec_off_emu","hl1elec_off_emu",10,0.0,2.0);
    
    hlt2elec_emu              =   subDiremu.make<TH1F> ("hlt2elec_emu","hlt2elec_emu",10,0.0,2.0);
    hlt2elec_off_emu          =   subDiremu.make<TH1F> ("hlt2elec_off_emu","hl2elec_off_emu",10,0.0,2.0);
    
    hltelecmu_emu             =   subDiremu.make<TH1F> ("hltelecmu_emu","hltelecmu_emu",10,0.0,2.0);
    hltelecmu_off_emu         =   subDiremu.make<TH1F> ("hltelecmu_off_emu","hlelecmu_off_emu",10,0.0,2.0);
    
    hltelecmurelax_emu        =   subDiremu.make<TH1F> ("hltelecmurelax_emu","hltelecmurelax_emu",10,0.0,2.0);   
    hltelecmurelax_off_emu    =   subDiremu.make<TH1F> ("hltelecmurelax_off_emu","hlelecmurelax_off_emu",10,0.0,2.0);
    
    hlt1jet_emu               =   subDiremu.make<TH1F> ("hlt1jet_emu","hlt1jet_emu",10,0.0,2.0);  
    hlt1jet_off_emu           =   subDiremu.make<TH1F> ("hlt1jet_off_emu","hl1jet_off_emu",10,0.0,2.0);
    
     //----new paths
    hlt4jet30_emu          =  subDiremu.make<TH1F> ("hlt4jet30_emu","hlt4jet30_emu",10,0.0,2.0);
    hlt1elec15_li_emu      =  subDiremu.make<TH1F> ("hlt1elec15_li_emu","hlt1elec15_li_emu",10,0.0,2.0);
    hlt1elec15_ni_emu      =  subDiremu.make<TH1F> ("hlt1elec15_ni_emu","hlt1elec15_ni_emu",10,0.0,2.0);
    
    hlt4jet30_off_emu          =  subDiremu.make<TH1F> ("hlt4jet30_off_emu","hlt4jet30_off_emu",10,0.0,2.0);
    hlt1elec15_li_off_emu      =  subDiremu.make<TH1F> ("hlt1elec15_li_off_emu","hlt1elec15_li_off_emu",10,0.0,2.0);
    hlt1elec15_ni_off_emu      =  subDiremu.make<TH1F> ("hlt1elec15_ni_off_emu","hlt1elec15_ni_off_emu",10,0.0,2.0);
      
    
 
   
 */
   
  
  nEvents = 0;
  nAccepted = 0;
 
  semilepEvent=0;
  dilepEvent=0;
  hadronicEvent=0;
  
  dimuEvent=0;
  dielEvent=0;
  ditauEvent=0;
  muEvent=0;
  tauEvent=0;
  elecEvent=0;
  emuEvent=0;
  taumuEvent=0;
  tauelEvent=0;
  
 
  

  

  return ;
  
  
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TopValidation::endJob() {  
//Write DQM thing..
 // outFile_  =  "prueba";
  //if(outFile_.size()>0)
 // if (&*edm::Service<DQMStore>()) edm::Service<DQMStore>()->save (outFile_);
  if(outputMEsInRootFile){
    dbe->showDirStructure();
    dbe->save(outputFileName);
  }


 /* cout << endl;
 
  cout << endl;
  cout << "Total: " << nEvents << endl;  
  cout << "Acceptd top events: " << nAccepted<< endl;  

  cout << endl;
 

  cout<<"semileptonic events="<<semilepEvent<<"  --> "<<muEvent<<" mu, "<<elecEvent<<" e, "<<tauEvent<<" tau"<<endl;
  cout<<"dileptonic events="<<dilepEvent<<" --> "<<dimuEvent<< " dimu, "<< dielEvent<<" diel, "<<ditauEvent<<" ditau,"
  <<taumuEvent<<" mutau, "<<tauelEvent<<" etau , "<<emuEvent<<" emu"<<endl;
  cout<<"hadronic events="<<hadronicEvent<<endl;


 

cout<<endl;  */

   

  return ;
}

//define this as a plug-in
DEFINE_FWK_MODULE(TopValidation);
