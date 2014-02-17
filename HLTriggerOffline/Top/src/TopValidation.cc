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
// $Id: TopValidation.cc,v 1.13 2012/07/09 16:17:37 ajafari Exp $
//
//


# include "HLTriggerOffline/Top/interface/TopValidation.h"
#include "FWCore/Common/interface/TriggerNames.h"



TopValidation::TopValidation(const edm::ParameterSet& iConfig)

{
  
   
     inputTag_           = iConfig.getParameter<edm::InputTag>("TriggerResultsCollection");
     hlt_bitnames        = iConfig.getParameter<std::vector<std::string> >("hltPaths");     
     hlt_bitnamesMu      = iConfig.getParameter<std::vector<std::string> >("hltMuonPaths");
     hlt_bitnamesEg      = iConfig.getParameter<std::vector<std::string> >("hltEgPaths");  
     hlt_bitnamesJet      = iConfig.getParameter<std::vector<std::string> >("hltJetPaths");  
  //   triggerTag_         = iConfig.getUntrackedParameter<string>("DQMFolder","HLT/Top");
     outputFileName      = iConfig.getParameter<std::string>("OutputFileName");
     outputMEsInRootFile = iConfig.getParameter<bool>("OutputMEsInRootFile");
      FolderName_ = iConfig.getParameter<std::string>("FolderName");
   
      topFolder << FolderName_ ;  

 
      
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
  Handle<reco::MuonCollection> muonsH;
  iEvent.getByLabel("muons", muonsH);
  
  // tracks collection
  Handle<reco::TrackCollection> tracks;
  iEvent.getByLabel("ctfWithMaterialTracks", tracks);
    
     
  // get calo jet collection
  Handle<reco::CaloJetCollection> jetsHandle;
  iEvent.getByLabel("iterativeCone5CaloJets", jetsHandle);

  
  // electron collection
  Handle<reco::GsfElectronCollection> electronsH;
  //  iEvent.getByLabel("pixelMatchGsfElectrons",electronsH);
  iEvent.getByLabel("gsfElectrons",electronsH);

  // Trigger 
  Handle<TriggerResults> trh;
  iEvent.getByLabel(inputTag_,trh);
  if( ! trh.isValid() ) {
    LogDebug("") << "HL TriggerResults with label ["+inputTag_.encode()+"] not found!";
    return;
  }  



  const edm::TriggerNames & triggerNames = iEvent.triggerNames(*trh);
 
  //////////////////////////////////
  //   generation  info                                     
  /////////////////////////////////
  
  //bool topevent = false;
 
  int ntop     = 0;
  int ngenel   = 0;
  int ngenmu   = 0;
  int ngentau  = 0;
  int ngenlep  = 0;
  int nmuaccept= 0;
  int nelaccept= 0;
 
  // Gen Particles Collection
  Handle <reco::GenParticleCollection> genParticles;
   iEvent.getByLabel("genParticles", genParticles);
  
   for (size_t i=0; i < genParticles->size(); ++i){
     const reco::Candidate & p = (*genParticles)[i];
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
  
  // if (ntop == 2) topevent = true; 
  
 

 // if (topevent){
 
  ////////////////////////////
  ////////  Muons
  /////////////////////////////////
  
    //Muon Collection to use
    std::map<double,reco::Muon> muonMap; 
  
    for (size_t i = 0; i< muonsH->size(); ++i){    
      if ( (*muonsH)[i].isGlobalMuon() && (*muonsH)[i].pt()>15 && fabs((*muonsH)[i].eta())<2.1){  
      muonMap[(*muonsH)[i].pt()] = (*muonsH)[i];
       }
    }     

    //Muon selection
    bool TwoMuonsAccepted = false;
  
    std::vector<reco::Muon> selected_muons;
    reco::Muon muon1,muon2;
  
    for( std::map<double,reco::Muon>::reverse_iterator rit=muonMap.rbegin(); rit!=muonMap.rend(); ++rit){
      selected_muons.push_back( (*rit).second );
    }
    
       if (selected_muons.size()==1) muon1 = selected_muons[0];

       if (selected_muons.size()>1){
          muon1 = selected_muons[0];
          muon2 = selected_muons[1];   
       }
 
 
    if( selected_muons.size()>1 && muon1.pt() >20 && muon1.charge()*muon2.charge()<0 ) TwoMuonsAccepted = true;

   
 //////////////////////////////////
 /////////  Electrons
 ///////////////////////////////////////  
   
    //Electron Collection to use
    std::map<double,reco::GsfElectron> electronMap;
   
    for (size_t i = 0; i<electronsH->size();++i){
     if( (*electronsH)[i].pt()>15 && fabs( (*electronsH)[i].eta() )<2.4) {
      electronMap[(*electronsH)[i].pt()] = (*electronsH)[i];
       }
    }     
        
    //Electron selection
    bool TwoElectronsAccepted = false;
   
    std::vector<reco::GsfElectron> selected_electrons;
    reco::GsfElectron electron1, electron2;
  
    for( std::map<double,reco::GsfElectron>::reverse_iterator rit=electronMap.rbegin(); rit!=electronMap.rend(); ++rit){
      selected_electrons.push_back( (*rit).second );
    }

     if (selected_electrons.size()==1) electron1 = selected_electrons[0];
    
      if (selected_electrons.size()>1){
      
        electron1 = selected_electrons[0];
	electron2 = selected_electrons[1];
      }
  
  
     if( selected_electrons.size()>1 && electron1.pt() >20 && electron1.charge()*electron2.charge()<0 ) TwoElectronsAccepted = true;
  
  
 
 /////////////////////////////////
 //////////  Jets
 /////////////////////////////////  
   
    //Jet Collection to use
     
    // Raw jets
    const reco::CaloJetCollection *jets = jetsHandle.product();
    reco::CaloJetCollection::const_iterator jet;
  
    int n_jets_20=0;
    
      for (jet = jets->begin(); jet != jets->end(); jet++){        
       // if (fabs(jet->eta()) <2.4 && jet->et() > 20) n_jets_20++; 
     //  if (fabs(jet->eta()) <2.4 && jet->et() > 13) n_jets_20++; 
      if (fabs(jet->eta()) <2.4 && jet->et() > 13) n_jets_20++;       
      } 
    

//// sort jets by et
 /*  std::map<double,reco::CaloJet> jetMap;
    
 for (size_t i = 0; i<jetsHandle->size();++i){
     if ((*jetsHandle)[i].et()>13 && fabs( (*jetsHandle)[i].eta())<2.4){
     
      
      jetMap[(*jetsHandle)[i].et()] = (*jetsHandle)[i];
      }
    }    
    
    std::vector<reco::CaloJet> selected_jets;
    reco::CaloJet jet1, jet2;
  
    for( std::map<double,reco::CaloJet>::reverse_iterator rit=jetMap.rbegin(); rit!=jetMap.rend(); ++rit){
      selected_jets.push_back( (*rit).second );
    }
    
    if (selected_jets.size()>1){
      jet1 = selected_jets[0];
      jet2 = selected_jets[1];
      }
    
*/

//////////////////////////////////////////////////////////
/////
////     "Offline" selection
/////
///////////////////////////////////////////////// 
    
 
    
    bool offline_mu       = false;
    bool offline_dimu     = false;
    bool offline_el       = false;
    bool offline_diel     = false;
    bool offline_emu      = false;
    
    
    if ( selected_muons.size()>0 && muon1.pt()>20 && n_jets_20>1)         offline_mu=true;
    if ( TwoMuonsAccepted && n_jets_20>1)                                 offline_dimu=true;
    if ( selected_electrons.size()>0 && electron1.pt()>20 && n_jets_20>1)  offline_el=true;
    if ( TwoElectronsAccepted && n_jets_20>1)                             offline_diel=true;
    if ( selected_muons.size()>0 && selected_electrons.size()>0 && (muon1.pt()>20 || electron1.pt()>20) && (muon1.charge()!= electron1.charge()) && n_jets_20>1) offline_emu=true;
    
    
  
    
  //////////////////////////////        
 // store fired bits
 ///////////////////////////////
    
  int wtrig_[100]  ={0}; 
  int wtrig_m[100] ={0};
  int wtrig_eg[100]={0};
  int wtrig_jet[100]={0};
  
  bool HLTQuadJet30 = false;
  bool HLTMu9       = false;
  
  
  int n_hlt_bits    = hlt_bitnames.size(); 
  int n_hlt_bits_mu = hlt_bitnamesMu.size();
  int n_hlt_bits_eg = hlt_bitnamesEg.size();
  int n_hlt_bits_jet= hlt_bitnamesJet.size();
  
  const unsigned int n_TriggerResults( trh.product()->size());

  for (unsigned int itrig=0; itrig< n_TriggerResults; ++itrig) {
  
     ///////////
           if (triggerNames.triggerName(itrig) == "HLT_QuadJet30"){
               if ( trh.product()->accept( itrig ) ) HLTQuadJet30=true;
             } 
	   if (triggerNames.triggerName(itrig) == "HLT_Mu9"){
               if ( trh.product()->accept( itrig ) ) HLTMu9=true;
          } 
  //////////////////
    if (trh.product()->accept(itrig)) {
    
         for (int i=0;i<n_hlt_bits;i++) {
            if ( triggerNames.triggerName(itrig)== hlt_bitnames[i]) {     
	       wtrig_[i]=1;
            }
         }
         for (int j=0;j<n_hlt_bits_mu;j++) {
            if ( triggerNames.triggerName(itrig)== hlt_bitnamesMu[j]) {     
	       wtrig_m[j]=1;
             }
         }
         for (int k=0;k<n_hlt_bits_eg;k++) {
             if ( triggerNames.triggerName(itrig)== hlt_bitnamesEg[k]) {     
	        wtrig_eg[k]=1;
             }
          }
	
	 for (int l=0;l<n_hlt_bits_jet;l++) {
             if ( triggerNames.triggerName(itrig)== hlt_bitnamesJet[l]) {     
	        wtrig_jet[l]=1;
             }
          }
    
       }
     } 
    
    
          
 /////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////
 ///////////////////////////////////////////////////////////////////////
    
     events_acc_off_muon->Fill(1,1); 
     if (ngenlep==1 && ngenmu==1)  events_acc_off_muon->Fill(2,1); 
     events_acc_off_electron->Fill(1,1); 
      if (ngenlep==1 && ngenel==1)  events_acc_off_electron->Fill(2,1); 
        																						    																						  
    /// **** tt->munubjjb *****    
    
  
      
    if ( ngenlep==1 && ngenmu==1 && nmuaccept==1){  //Select events within acceptance
      
            events_acc_off_muon->Fill(3,1); 
	    
          for (int j=0;j<n_hlt_bits_mu;j++) {
                h_mu_gen->Fill(j+1);
              if (wtrig_m[j]==1) {   
	       hlt_bitmu_hist_gen->Fill(j+1);
              }
           }
	   
	     for (int it=0; it<n_hlt_bits_jet;it++){
	   h_jet_gen->Fill(it+1);
	   if (wtrig_jet[it]==1) {
	   hlt_bitjet_hist_gen->Fill(it+1);
	     }   
	   }
     
      
      // Efficiencies wrt MC + offline
      if (offline_mu){
      
             events_acc_off_muon->Fill(4,1); 
      
         //  et_off_jet_mu -> Fill(jet1.et());
	  // eta_off_jet_mu-> Fill(jet1.eta());
	   
           for (int it=0; it<n_hlt_bits_jet;it++){
	   h_jet_reco->Fill(it+1);
	   if (wtrig_jet[it]==1) {
	   hlt_bitjet_hist_reco->Fill(it+1);
	 //  h_etjet1_trig_mu[it]->Fill(jet1.et());
	 //  h_etajet1_trig_mu[it]->Fill(jet1.eta());
	   
	   
	   }
	   
	   }
      
          eta_off_mu->Fill(muon1.eta()); 
          pt_off_mu-> Fill(muon1.pt());
      
            for (int j=0;j<n_hlt_bits_mu;j++) {
                h_mu_reco->Fill(j+1);
              if (wtrig_m[j]==1) {   
               h_ptmu1_trig[j]->Fill(muon1.pt());
	       h_etamu1_trig[j]->Fill(muon1.eta());
	       hlt_bitmu_hist_reco->Fill(j+1);
              }
           }
    	
         }
     }



///////// 4jets+1muon efficiency monitoring

   if (HLTQuadJet30){    // events firing the 4jet30 trigger
   
    if (offline_mu){    // events with 1 rec muon+ 2jets
   
      ptmuon_4jet1muSel->Fill(muon1.pt());
      etamuon_4jet1muSel->Fill(muon1.eta());
      Njets_4jet1muSel->Fill(n_jets_20);
      
        if (HLTMu9){
	
	 ptmuon_4jet1muSel_hltmu9->Fill(muon1.pt());
         etamuon_4jet1muSel_hltmu9->Fill(muon1.eta());
         Njets_4jet1muSel_hltmu9->Fill(n_jets_20);	
	}
      }
   }
   

////////////////////////////
     
    
    // *****  tt->enubjjb *****
    if ( ngenlep==1 && ngenel==1 && nelaccept==1){   
    
         events_acc_off_electron->Fill(3,1); 
    
           for (int j=0;j<n_hlt_bits_eg;j++) {
                h_el_gen->Fill(j+1);
              if (wtrig_eg[j]==1) {   
	       hlt_bitel_hist_gen->Fill(j+1);
              }
           }
	   
	     for (int it=0; it<n_hlt_bits_jet;it++){
	   h_jet_gen_el->Fill(it+1);
	   if (wtrig_jet[it]==1) {
	   hlt_bitjet_hist_gen_el->Fill(it+1);
	     }   
	   }
      
      // Efficiencies wrt mc + offline
          if (offline_el){
	  
	   events_acc_off_electron->Fill(4,1); 
	  
	    /*/// jets
	       et_off_jet_el -> Fill(jet1.et());
	       eta_off_jet_el-> Fill(jet1.eta());*/
	       
	           for (int it=0; it<n_hlt_bits_jet;it++){
	             h_jet_reco_el->Fill(it+1);
	               if (wtrig_jet[it]==1) {
	                 hlt_bitjet_hist_reco_el->Fill(it+1);
	              //    h_etjet1_trig_el[it]->Fill(jet1.et());
	              //    h_etajet1_trig_el[it]->Fill(jet1.eta());   
	                      }
	   
	                       }
	       
	       
	    ///   
	      eta_off_el->Fill(electron1.eta()); 
              pt_off_el->Fill(electron1.pt());
	
	       for (int k=0;k<n_hlt_bits_eg;k++) {
	              h_el_reco->Fill(k+1);
       
               if (wtrig_eg[k]==1) {   
                h_ptel1_trig[k]->Fill(electron1.pt());
	        h_etael1_trig[k]->Fill(electron1.eta()); 
		hlt_bitel_hist_reco->Fill(k+1);       
               }
             }
	       
         }    
    }
    
    

    // ****** tt->munubmunub *****
    if ( ngenlep==2 && ngenmu==2 && nmuaccept==2){  
      
      
      // Efficiencies wrt mc+offline
        if (offline_dimu){
	
	    eta_off_dimu1->Fill(muon1.eta());
	    eta_off_dimu2->Fill(muon2.eta());
 	    pt_off_dimu1->Fill(muon1.pt());
 	    pt_off_dimu2->Fill(muon2.pt());
	
           for (int j=0;j<n_hlt_bits_mu;j++) {
       
               if (wtrig_m[j]==1) {   
	   
                 h_ptmu1_trig_dimu[j]->Fill(muon1.pt());
	         h_etamu1_trig_dimu[j]->Fill(muon1.eta());        
                }
           }               
        } 
    }
    
    
    
    // *****   tt->enubenub *****
    if ( ngenlep==2 && ngenel==2 && nelaccept==2){   
      
     
      // Efficiencies wrt mc+offline
         if (offline_diel){
	 
	    eta_off_diel1->Fill(electron1.eta()); 
	    eta_off_diel2->Fill(electron2.eta());
	    pt_off_diel1->Fill(electron1.pt());
	    pt_off_diel2->Fill(electron2.pt());
		
	   for (int k=0;k<n_hlt_bits_eg;k++) {
       
          if (wtrig_eg[k]==1) {   
	   
              h_ptel1_trig_diel[k]->Fill(electron1.pt());
	      h_etael1_trig_diel[k]->Fill(electron1.eta());        
            }
        }
	      
      }    
    }
    
    
    // *****  tt->enubmunub
    if ( ngenlep==2 && ngenel==1 && ngenmu==1 && nmuaccept==1 && nelaccept==1){   // tt->e mu events passing acceptance
      
    
      // Efficiencies wrt mc+offline      
      if (offline_emu){
      
            eta_off_emu_muon->Fill(muon1.eta()); 
 	    pt_off_emu_muon->Fill(muon1.pt());
 	    eta_off_emu_electron->Fill(electron1.eta()); 
	    pt_off_emu_electron->Fill(electron1.pt());
	
	 for (int i=0;i<n_hlt_bits;i++) {
       
          if (wtrig_[i]==1) {   
	   
              h_ptel1_trig_em[i]->Fill(electron1.pt());
	      h_etael1_trig_em[i]->Fill(electron1.eta()); 
	      h_ptmu1_trig_em[i]->Fill(muon1.pt());
	      h_etamu1_trig_em[i]->Fill(muon1.eta());       
            }
        }
		
      }    
    }
    
    
 ////////////////////////////////////////////////////////////////
  //}
  
}



// ------------ method called once each job just before starting event loop  ------------
void 
TopValidation::beginJob()
{
  
       dbe = edm::Service<DQMStore>().operator->();

	//dbe->setCurrentFolder("HLT/Top");
	//dbe->setCurrentFolder(triggerTag_);
   
  

  
  ////////////////////////////////////////////
  //////   histos lepton pt, eta for events passing hlt, efficiencies
  /////////////////////////////////////////////
  
   
    dbe->setCurrentFolder(topFolder.str()+"Semileptonic_muon");
    
    hlt_bitmu_hist_reco = dbe->book1D("muHLT","muHLT",hlt_bitnamesMu.size(),0.5,hlt_bitnamesMu.size()+0.5);
    h_mu_reco = dbe->book1D("MuonEvents","MuonEvents",hlt_bitnamesMu.size(),0.5,hlt_bitnamesMu.size()+0.5);
    
    hlt_bitmu_hist_gen = dbe->book1D("genmuHLT","genmuHLT",hlt_bitnamesMu.size(),0.5,hlt_bitnamesMu.size()+0.5);
    h_mu_gen = dbe->book1D("genMuonEvents","genMuonEvents",hlt_bitnamesMu.size(),0.5,hlt_bitnamesMu.size()+0.5);
   
    events_acc_off_muon = dbe->book1D("NEvents_acc_off","NEvents_acc_off",4,0.5,4.5);
    events_acc_off_muon -> setBinLabel(1,"Total Events");
    events_acc_off_muon -> setBinLabel(2,"Gen semimuon");
    events_acc_off_muon -> setBinLabel(3,"Acceptance");
    events_acc_off_muon -> setBinLabel(4,"Acceptance+offline");
 
     
    dbe->setCurrentFolder(topFolder.str()+"Semileptonic_electron"); 
    
    hlt_bitel_hist_reco = dbe->book1D("elHLT","elHLT",hlt_bitnamesEg.size(),0.5,hlt_bitnamesEg.size()+0.5);
    h_el_reco = dbe->book1D("ElectronEvents","ElectronEvents",hlt_bitnamesEg.size(),0.5,hlt_bitnamesEg.size()+0.5);
    
  
    hlt_bitel_hist_gen = dbe->book1D("genelHLT","genelHLT",hlt_bitnamesEg.size(),0.5,hlt_bitnamesEg.size()+0.5);
    h_el_gen = dbe->book1D("genElectronEvents","genElectronEvents",hlt_bitnamesEg.size(),0.5,hlt_bitnamesEg.size()+0.5);
    
    
    events_acc_off_electron = dbe->book1D("NEvents_acc_off","NEvents_acc_off",4,0.5,4.5);
    events_acc_off_electron -> setBinLabel(1,"Total Events");
    events_acc_off_electron -> setBinLabel(2,"Gen semielectron");
    events_acc_off_electron -> setBinLabel(3,"Acceptance");
    events_acc_off_electron -> setBinLabel(4,"Acceptance+offline");
   
      
    dbe->setCurrentFolder(topFolder.str()+"Jets");
       
     h_jet_reco =  dbe->book1D("denom","denom",hlt_bitnamesJet.size(),0.5,hlt_bitnamesJet.size()+0.5);
     hlt_bitjet_hist_reco =  dbe->book1D("numer","numer",hlt_bitnamesJet.size(),0.5,hlt_bitnamesJet.size()+0.5);
   
     h_jet_reco_el =  dbe->book1D("denom_el","denom_el",hlt_bitnamesJet.size(),0.5,hlt_bitnamesJet.size()+0.5);
     hlt_bitjet_hist_reco_el = dbe->book1D("numer_el","numer_el",hlt_bitnamesJet.size(),0.5,hlt_bitnamesJet.size()+0.5);
   
   ///
     h_jet_gen =  dbe->book1D("denom_gen","denom_gen",hlt_bitnamesJet.size(),0.5,hlt_bitnamesJet.size()+0.5);
     hlt_bitjet_hist_gen =  dbe->book1D("numer_gen","numer_gen",hlt_bitnamesJet.size(),0.5,hlt_bitnamesJet.size()+0.5);
   
     h_jet_gen_el =  dbe->book1D("denom_el_gen","denom_el_gen",hlt_bitnamesJet.size(),0.5,hlt_bitnamesJet.size()+0.5);
     hlt_bitjet_hist_gen_el = dbe->book1D("numer_el_gen","numer_el_gen",hlt_bitnamesJet.size(),0.5,hlt_bitnamesJet.size()+0.5);
   
   
   /////////
    /* et_off_jet_mu                = dbe->book1D ("Jet1Et_M","Jet1Et_M",51,0.0,150.0);
     et_off_jet_el                = dbe->book1D ("Jet1Et_E","Jet1Et_E",51,0.0,150.0);
     eta_off_jet_mu               = dbe->book1D ("Jet1Eta_M","Jet1Eta_M",51,-2.5,2.5);
     eta_off_jet_el               = dbe->book1D ("Jet1Eta_E","Jet1Eta_E",51,-2.5,2.5);
     njets_off_mu                 = dbe->book1D ("NJets_M","NJets_M",11,-0.5,10.5);
     njets_off_el                 = dbe->book1D ("NJets_E","NJets_E",11,-0.5,10.5);
    
  
  
    
     for (size_t j=0;j<hlt_bitnamesJet.size();j++) { 
    
     string histname_etjet       = "Jet1Et_M_"+hlt_bitnamesJet[j];
     string histname_etajet       = "Jet1Eta_M_"+hlt_bitnamesJet[j];
     string histname_etjet_el       = "Jet1Et_E_"+hlt_bitnamesJet[j];
     string histname_etajet_el       = "Jet1Eta_E_"+hlt_bitnamesJet[j];
     hlt_bitjet_hist_reco -> setBinLabel(j+1,hlt_bitnamesJet[j].c_str());
     h_jet_reco -> setBinLabel(j+1,hlt_bitnamesJet[j].c_str());
     h_jet_reco_el -> setBinLabel(j+1,hlt_bitnamesJet[j].c_str());
     h_etjet1_trig_mu[j]   =   dbe->book1D((histname_etjet).c_str(),(hlt_bitnamesJet[j]+"jetEt_M").c_str(),51,0.0,150.); 
     h_etjet1_trig_el[j]   =   dbe->book1D((histname_etjet_el).c_str(),(hlt_bitnamesJet[j]+"jetEt_E").c_str(),51,0.0,150.); 
     h_etajet1_trig_mu[j]  =   dbe->book1D((histname_etajet).c_str(),(hlt_bitnamesJet[j]+"jetEta_M").c_str(),51,-2.5,2.5); 
     h_etajet1_trig_el[j]  =   dbe->book1D((histname_etajet_el).c_str(),(hlt_bitnamesJet[j]+"jetEta_E").c_str(),51,-2.5,2.5); 
    
     
  }
    */
    
    
   	
    dbe->setCurrentFolder(topFolder.str()+"Dileptonic_emu");  
     
 
   for (size_t j=0;j<hlt_bitnames.size();j++) { 
     std::string histname_ptmu_em  = "Muon1Pt_EM_"+hlt_bitnames[j];
     std::string histname_etamu_em = "Muon1Eta_EM_"+hlt_bitnames[j];
     std::string histname_ptel_em  = "Electron1Pt_EM_"+hlt_bitnames[j];
     std::string histname_etael_em = "Electron1Eta_EM_"+hlt_bitnames[j];
     
     h_ptmu1_trig_em[j]  = dbe->book1D((histname_ptmu_em).c_str(),(hlt_bitnames[j]+"_muonPt_EM").c_str(),40,0.0,150.); 
     h_etamu1_trig_em[j] = dbe->book1D((histname_etamu_em).c_str(),(hlt_bitnames[j]+"_muonEta_EM").c_str(),51,-2.5,2.5);
     
     h_ptel1_trig_em[j]  = dbe->book1D((histname_ptel_em).c_str(),(hlt_bitnames[j]+"_electronPt_EM").c_str(),40,0.0,150.); 
     h_etael1_trig_em[j] = dbe->book1D((histname_etael_em).c_str(),(hlt_bitnames[j]+"_electronEta_EM").c_str(),51,-2.5,2.5); 
    
  }
  
 
  
 
   for (size_t jj=0;jj<hlt_bitnamesMu.size();jj++) { 
     std::string histname_ptmu       = "Muon1Pt_M_"+hlt_bitnamesMu[jj];
     std::string histname_etamu      = "Muon1Eta_M_"+hlt_bitnamesMu[jj];
     std::string histname_ptmu_dimu  = "Muon1Pt_MM_"+hlt_bitnamesMu[jj];
     std::string histname_etamu_dimu = "Muon1Eta_MM_"+hlt_bitnamesMu[jj];
    
     dbe->setCurrentFolder(topFolder.str()+"Semileptonic_muon");
     h_ptmu1_trig[jj]       = dbe->book1D((histname_ptmu).c_str(),(hlt_bitnamesMu[jj]+"muonPt_M").c_str(),40,0.0,150.); 
     h_etamu1_trig[jj]      = dbe->book1D((histname_etamu).c_str(),(hlt_bitnamesMu[jj]+"muonEta_M").c_str(),51,-2.5,2.5);
     
      hlt_bitmu_hist_reco -> setBinLabel(jj+1,hlt_bitnamesMu[jj].c_str());
     h_mu_reco -> setBinLabel(jj+1,hlt_bitnamesMu[jj].c_str());
     
     hlt_bitmu_hist_gen -> setBinLabel(jj+1,hlt_bitnamesMu[jj].c_str());
     h_mu_gen -> setBinLabel(jj+1,hlt_bitnamesMu[jj].c_str());
     
      dbe->setCurrentFolder(topFolder.str()+"Dileptonic_muon");
     h_ptmu1_trig_dimu[jj]  = dbe->book1D((histname_ptmu_dimu).c_str(),(hlt_bitnamesMu[jj]+"muon1Pt_MM").c_str(),40,0.0,150.); 
     h_etamu1_trig_dimu[jj] = dbe->book1D((histname_etamu_dimu).c_str(),(hlt_bitnamesMu[jj]+"muon1Pt_MM").c_str(),51,-2.5,2.5); 
    
     
   
    
  }
  
  
  
   
  
   
   for (size_t k=0;k<hlt_bitnamesEg.size();k++) { 
   
 
     std::string histname_ptel       = "Electron1Pt_E_"+hlt_bitnamesEg[k];
     std::string histname_etael      = "Electron1Eta_E_"+hlt_bitnamesEg[k];
     std::string histname_ptel_diel  = "Electron1Pt_EE_"+hlt_bitnamesEg[k];
     std::string histname_etael_diel = "Electron1Eta_EE_"+hlt_bitnamesEg[k];
     
    
     dbe->setCurrentFolder(topFolder.str()+"Semileptonic_electron");
    
     h_ptel1_trig[k]       = dbe->book1D((histname_ptel).c_str(),(hlt_bitnamesEg[k]+"electronPt_E").c_str(),40,0.0,150.); 
     h_etael1_trig[k]      = dbe->book1D((histname_etael).c_str(),(hlt_bitnamesEg[k]+"electronEta_E").c_str(),51,-2.5,2.5);
     
     hlt_bitel_hist_reco -> setBinLabel(k+1,hlt_bitnamesEg[k].c_str());
     h_el_reco -> setBinLabel(k+1,hlt_bitnamesEg[k].c_str());
      
     hlt_bitel_hist_gen -> setBinLabel(k+1,hlt_bitnamesEg[k].c_str());
     h_el_gen -> setBinLabel(k+1,hlt_bitnamesEg[k].c_str());
     
     
      dbe->setCurrentFolder(topFolder.str()+"Dileptonic_electron");
     h_ptel1_trig_diel[k]  = dbe->book1D((histname_ptel_diel).c_str(),(hlt_bitnamesEg[k]+"electron1Pt_EE").c_str(),40,0.0,150.); 
     h_etael1_trig_diel[k] = dbe->book1D((histname_etael_diel).c_str(),(hlt_bitnamesEg[k]+"electron1Eta_EE").c_str(),51,-2.5,2.5); 
    
   
  
    
    
  }
 
  
  
/////////////////////////////////////////
///    histos lepton pt, eta
//////////////////////////////////////////7


   // 4 jets+1mu eff monitoring
   
    dbe->setCurrentFolder(topFolder.str()+"4JetsPlus1MuonToCompareWithData");
    
    ptmuon_4jet1muSel     = dbe->book1D ("Muon1Pt_4Jets1MuonMon", "Muon1Pt_4Jets1MuonMon",40, 0.0,150.0);
    etamuon_4jet1muSel    = dbe->book1D ("Muon1Eta_4Jets1MuonMon","Muon1Eta_4Jets1MuonMon",51, -2.5,2.5);
    Njets_4jet1muSel      = dbe->book1D ("NJets_4Jets1MuonMon",   "NJets_4Jets1MuonMon",11, -0.5,10.5);
    
    ptmuon_4jet1muSel_hltmu9     = dbe->book1D ("Muon1Pt_4Jets1MuonHLTMu9Mon", "Muon1Pt_4Jets1MuonHLTMu9Mon",40, 0.0,150.0);
    etamuon_4jet1muSel_hltmu9    = dbe->book1D ("Muon1Eta_4Jets1MuonHLTMu9Mon","Muon1Eta_4Jets1MuonHLTMu9Mon",51, -2.5,2.5);
    Njets_4jet1muSel_hltmu9      = dbe->book1D ("NJets_4Jets1MuonHLTMu9Mon",   "NJets_4Jets1MuonHLTMu9Mon",11, -0.5,10.5);


  
   //semimu events 
    dbe->setCurrentFolder(topFolder.str()+"Semileptonic_muon");
    
  
  
    eta_off_mu               = dbe->book1D ("Muon1Eta_M","Muon1Eta_M",51,-2.5,2.5);
    pt_off_mu                = dbe->book1D ("Muon1Pt_M","Muon1Pt_M",40,0.0,150.0);
    
  
     //semiel events  
      dbe->setCurrentFolder(topFolder.str()+"Semileptonic_electron");
   
    eta_off_el               = dbe->book1D ("Electron1Eta_E","Electron1Eta_E",51,-2.5,2.5);
    pt_off_el                = dbe->book1D ("Electron1Pt_E","Electron1Pt_E",40,0.0,150.0);
    
    
        
    //dimu events
     dbe->setCurrentFolder(topFolder.str()+"Dileptonic_muon");
   
 
    eta_off_dimu1            = dbe->book1D ("Muon1Eta_MM","Muon1Eta_MM",51,-2.5,2.5);
    pt_off_dimu1             = dbe->book1D ("Muon1Pt_MM","Muon1Pt_MM",40,0.0,150.0);
    eta_off_dimu2            = dbe->book1D ("Muon2Eta_MM","Muon2Eta_MM",51,-2.5,2.5);
    pt_off_dimu2             = dbe->book1D ("Muon2Pt_MM","Muon2Pt_MM",40,0.0,150.0);
    
  
    
    //diel events
     dbe->setCurrentFolder(topFolder.str()+"Dileptonic_electron");
     
   
    eta_off_diel1            = dbe->book1D ("Electron1Eta_EE","Electron1Eta_EE",51,-2.5,2.5);
    pt_off_diel1             = dbe->book1D ("Electron1Pt_EE","Electron1Pt_EE",40,0.0,150.0);
    eta_off_diel2            = dbe->book1D ("Electron2Eta_EE","Electron2Eta_EE",51,-2.5,2.5);
    pt_off_diel2             = dbe->book1D ("Electron2Pt_EE","Electron2Pt_EE",40,0.0,150.0);
  
    
    //emu events
     dbe->setCurrentFolder(topFolder.str()+"Dileptonic_emu");
     
   
    
    eta_off_emu_muon         = dbe->book1D ("Muon1Eta_EM","Muon1Eta_EM",51,-2.5,2.5);
    pt_off_emu_muon          = dbe->book1D ("Muon1Pt_EM","Muon1Pt_EM",40,0.0,150.0);
    
    eta_off_emu_electron     = dbe->book1D ("Electron1Eta_EM","Electron1Eta_EM",51,-2.5,2.5);
    pt_off_emu_electron      = dbe->book1D ("Electron1Pt_EM","Electron1Pt_EM",40,0.0,150.0);
  
  


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



   

  return ;
}

//define this as a plug-in
//DEFINE_FWK_MODULE(TopValidation);
