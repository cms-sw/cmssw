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
// $Id: TopValidation.cc,v 1.4 2009/06/22 13:23:37 lobelle Exp $
//
//


# include "HLTriggerOffline/Top/interface/TopValidation.h"




TopValidation::TopValidation(const edm::ParameterSet& iConfig)

{
  
   
     inputTag_           = iConfig.getParameter<edm::InputTag>("TriggerResultsCollection");
     hlt_bitnames        = iConfig.getParameter<std::vector<string> >("hltPaths");     
     hlt_bitnamesMu      = iConfig.getParameter<std::vector<string> >("hltMuonPaths");
     hlt_bitnamesEg      = iConfig.getParameter<std::vector<string> >("hltEgPaths");  
     triggerTag_         = iConfig.getUntrackedParameter<string>("DQMFolder","HLT/Top");
     outputFileName      = iConfig.getParameter<std::string>("OutputFileName");
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
  //  iEvent.getByLabel("pixelMatchGsfElectrons",electronsH);
  iEvent.getByLabel("gsfElectrons",electronsH);

  // Trigger 
  Handle<TriggerResults> trh;
  try {iEvent.getByLabel(inputTag_,trh);} catch(...) {;}
  
  triggerNames_.init(*trh);
 
 
  //////////////////////////////////
  //   generation  info                                     
  /////////////////////////////////
  
  bool topevent = false;
 
  int ntop     = 0;
  int ngenel   = 0;
  int ngenmu   = 0;
  int ngentau  = 0;
  int ngenlep  = 0;
  int nmuaccept= 0;
  int nelaccept= 0;
 
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
     if( (*electronsH)[i].pt()>15 && fabs( (*electronsH)[i].eta() )<2.4){
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
    const CaloJetCollection *jets = jetsHandle.product();
    CaloJetCollection::const_iterator jet;
    
    int n_jets_20=0;
    
      for (jet = jets->begin(); jet != jets->end(); jet++){        
       // if (fabs(jet->eta()) <2.4 && jet->et() > 20) n_jets_20++; 
       if (fabs(jet->eta()) <2.4 && jet->et() > 13) n_jets_20++;    
      } 
    

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
  
  
  int n_hlt_bits    = hlt_bitnames.size(); 
  int n_hlt_bits_mu = hlt_bitnamesMu.size();
  int n_hlt_bits_eg = hlt_bitnamesEg.size();
  
  const unsigned int n_TriggerResults( trh.product()->size());

  for (unsigned int itrig=0; itrig< n_TriggerResults; ++itrig) {
    if (trh.product()->accept(itrig)) {
    
         for (int i=0;i<n_hlt_bits;i++) {
            if ( triggerNames_.triggerName(itrig)== hlt_bitnames[i]) {     
	       wtrig_[i]=1;
            }
         }
         for (int j=0;j<n_hlt_bits_mu;j++) {
            if ( triggerNames_.triggerName(itrig)== hlt_bitnamesMu[j]) {     
	       wtrig_m[j]=1;
             }
         }
         for (int k=0;k<n_hlt_bits_eg;k++) {
             if ( triggerNames_.triggerName(itrig)== hlt_bitnamesEg[k]) {     
	        wtrig_eg[k]=1;
             }
          }
         
       }
     } 
    
    
          
 /////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////
 ///////////////////////////////////////////////////////////////////////
    
        																						    																						  
    /// **** tt->munubjjb *****    
    if ( ngenlep==1 && ngenmu==1 && nmuaccept==1){  //Select events within acceptance
      
      
      // Efficiencies wrt MC + offline
      if (offline_mu){
      
          eta_off_mu->Fill(muon1.eta()); 
          pt_off_mu-> Fill(muon1.pt());
      
            for (int j=0;j<n_hlt_bits_mu;j++) {
            
              if (wtrig_m[j]==1) {   
               h_ptmu1_trig[j]->Fill(muon1.pt());
	       h_etamu1_trig[j]->Fill(muon1.eta());
              }
           }
    	
         }
     }
     
    
    // *****  tt->enubjjb *****
    if ( ngenlep==1 && ngenel==1 && nelaccept==1){   
      
      
      
      // Efficiencies wrt mc + offline
          if (offline_el){
	  
	      eta_off_el->Fill(electron1.eta()); 
              pt_off_el->Fill(electron1.pt());
	
	       for (int k=0;k<n_hlt_bits_eg;k++) {
       
               if (wtrig_eg[k]==1) {   
                h_ptel1_trig[k]->Fill(electron1.pt());
	        h_etael1_trig[k]->Fill(electron1.eta());        
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
TopValidation::beginJob(const edm::EventSetup&)
{
  
       dbe = edm::Service<DQMStore>().operator->();

	//dbe->setCurrentFolder("HLT/Top");
	dbe->setCurrentFolder(triggerTag_);
   
  
  
  ////////////////////////////////////////////
  //////   histos lepton pt, eta for events passing hlt
  /////////////////////////////////////////////
   
     
   for (size_t j=0;j<hlt_bitnames.size();j++) { 
     string histname_ptmu_em  = "Muon1Pt_EM_"+hlt_bitnames[j];
     string histname_etamu_em = "Muon1Eta_EM_"+hlt_bitnames[j];
     string histname_ptel_em  = "Electron1Pt_EM_"+hlt_bitnames[j];
     string histname_etael_em = "Electron1Eta_EM_"+hlt_bitnames[j];
     
     h_ptmu1_trig_em[j]  = dbe->book1D((histname_ptmu_em).c_str(),(hlt_bitnames[j]+"_muonPt_EM").c_str(),51,0.0,150.); 
     h_etamu1_trig_em[j] = dbe->book1D((histname_etamu_em).c_str(),(hlt_bitnames[j]+"_muonEta_EM").c_str(),51,-2.5,2.5);
     
     h_ptel1_trig_em[j]  = dbe->book1D((histname_ptel_em).c_str(),(hlt_bitnames[j]+"_electronPt_EM").c_str(),51,0.0,150.); 
     h_etael1_trig_em[j] = dbe->book1D((histname_etael_em).c_str(),(hlt_bitnames[j]+"_electronEta_EM").c_str(),51,-2.5,2.5); 
    
  }
  
   for (size_t jj=0;jj<hlt_bitnamesMu.size();jj++) { 
     string histname_ptmu       = "Muon1Pt_M_"+hlt_bitnamesMu[jj];
     string histname_etamu      = "Muon1Eta_M_"+hlt_bitnamesMu[jj];
     string histname_ptmu_dimu  = "Muon1Pt_MM_"+hlt_bitnamesMu[jj];
     string histname_etamu_dimu = "Muon1Eta_MM_"+hlt_bitnamesMu[jj];
    
     h_ptmu1_trig[jj]       = dbe->book1D((histname_ptmu).c_str(),(hlt_bitnamesMu[jj]+"muonPt_M").c_str(),51,0.0,150.); 
     h_etamu1_trig[jj]      = dbe->book1D((histname_etamu).c_str(),(hlt_bitnamesMu[jj]+"muonEta_M").c_str(),51,-2.5,2.5);
     
     h_ptmu1_trig_dimu[jj]  = dbe->book1D((histname_ptmu_dimu).c_str(),(hlt_bitnamesMu[jj]+"muon1Pt_MM").c_str(),51,0.0,150.); 
     h_etamu1_trig_dimu[jj] = dbe->book1D((histname_etamu_dimu).c_str(),(hlt_bitnamesMu[jj]+"muon1Pt_MM").c_str(),51,-2.5,2.5); 
    
  }
  
   for (size_t k=0;k<hlt_bitnamesEg.size();k++) { 
     string histname_ptel       = "Electron1Pt_E_"+hlt_bitnamesEg[k];
     string histname_etael      = "Electron1Eta_E_"+hlt_bitnamesEg[k];
     string histname_ptel_diel  = "Electron1Pt_EE_"+hlt_bitnamesEg[k];
     string histname_etael_diel = "Electron1Eta_EE_"+hlt_bitnamesEg[k];
    
     h_ptel1_trig[k]       = dbe->book1D((histname_ptel).c_str(),(hlt_bitnamesEg[k]+"electronPt_E").c_str(),51,0.0,150.); 
     h_etael1_trig[k]      = dbe->book1D((histname_etael).c_str(),(hlt_bitnamesEg[k]+"electronEta_E").c_str(),51,-2.5,2.5);
     
     h_ptel1_trig_diel[k]  = dbe->book1D((histname_ptel_diel).c_str(),(hlt_bitnamesEg[k]+"electron1Pt_EE").c_str(),51,0.0,150.); 
     h_etael1_trig_diel[k] = dbe->book1D((histname_etael_diel).c_str(),(hlt_bitnamesEg[k]+"electron1Eta_EE").c_str(),51,-2.5,2.5); 
    
  }
  
  
/////////////////////////////////////////
///    histos lepton pt, eta
//////////////////////////////////////////7
  
   //semimu events 
  
    eta_off_mu               = dbe->book1D ("Muon1Eta_M","Muon1Eta_M",51,-2.5,2.5);
    pt_off_mu                = dbe->book1D ("Muon1Pt_M","Muon1Pt_M",51,0.0,150.0);
    
 
     //semiel events  
   
    eta_off_el               = dbe->book1D ("Electron1Eta_E","Electron1Eta_E",51,-2.5,2.5);
    pt_off_el                = dbe->book1D ("Electron1Pt_E","Electron1Pt_E",51,0.0,150.0);
    
        
    //dimu events
   
    eta_off_dimu1            = dbe->book1D ("Muon1Eta_MM","Muon1Eta_MM",51,-2.5,2.5);
    pt_off_dimu1             = dbe->book1D ("Muon1Pt_MM","Muon1Pt_MM",51,0.0,150.0);
    eta_off_dimu2            = dbe->book1D ("Muon2Eta_MM","Muon2Eta_MM",51,-2.5,2.5);
    pt_off_dimu2             = dbe->book1D ("Muon2Pt_MM","Muon2Pt_MM",51,0.0,150.0);
    
  
    
    //diel events
    
    eta_off_diel1            = dbe->book1D ("Electron1Eta_EE","Electron1Eta_EE",51,-2.5,2.5);
    pt_off_diel1             = dbe->book1D ("Electron1Pt_EE","Electron1Pt_EE",51,0.0,150.0);
    eta_off_diel2            = dbe->book1D ("Electron2Eta_EE","Electron2Eta_EE",51,-2.5,2.5);
    pt_off_diel2             = dbe->book1D ("Electron2Pt_EE","Electron2Pt_EE",51,0.0,150.0);
  
    
    //emu events
    
    eta_off_emu_muon         = dbe->book1D ("Muon1Eta_EM","Muon1Eta_EM",51,-2.5,2.5);
    pt_off_emu_muon          = dbe->book1D ("Muon1Pt_EM","Muon1Pt_EM",51,0.0,150.0);
    
    eta_off_emu_electron     = dbe->book1D ("Electron1Eta_EM","Electron1Eta_EM",51,-2.5,2.5);
    pt_off_emu_electron      = dbe->book1D ("Electron1Pt_EM","Electron1Pt_EM",51,0.0,150.0);
  
  


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
DEFINE_FWK_MODULE(TopValidation);
