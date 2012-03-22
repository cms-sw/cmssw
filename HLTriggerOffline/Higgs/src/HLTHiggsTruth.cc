#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>
#include <stdlib.h>
#include <string.h>

#include "HLTriggerOffline/Higgs/interface/HLTHiggsTruth.h"

HLTHiggsTruth::HLTHiggsTruth() {

  //set parameter defaults 
  _Monte=false;  
  _Debug=false;
  isvisible=false;
 // isMuonDecay=false;
 // isElecDecay=false;
  //isEMuDecay=false;
 // isPhotonDecay=false;
//  isMuonDecay_acc=false;
 // isElecDecay_acc=false;
 // isEMuDecay_acc=false;
  isPhotonDecay_acc=false;
  isTauDecay_acc =false;
  isMuonDecay_recoacc=false;
  isElecDecay_recoacc=false;
  isEMuDecay_recoacc=false;
  isPhotonDecay_recoacc=false;
  isTauDecay_recoacc =false;
  isvisible_reco= false;
  
  
  
}

/*  Setup the analysis to put the branch-variables into the tree. */
void HLTHiggsTruth::setup(const edm::ParameterSet& pSet, TTree* HltTree) {

  edm::ParameterSet myMCParams = pSet.getParameter<edm::ParameterSet>("RunParameters") ;
  std::vector<std::string> parameterNames = myMCParams.getParameterNames() ;
  
  for ( std::vector<std::string>::iterator iParam = parameterNames.begin();
	iParam != parameterNames.end(); iParam++ ){
    if  ( (*iParam) == "Monte" ) _Monte =  myMCParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "Debug" ) _Debug =  myMCParams.getParameter<bool>( *iParam );
  }




//  const int kMaxMcTruth = 50000;
//  mcpid = new float[kMaxMcTruth];
//  mcvx = new float[kMaxMcTruth];
//  mcvy = new float[kMaxMcTruth];
//  mcvz = new float[kMaxMcTruth];
//  mcpt = new float[kMaxMcTruth];

   

  // MCtruth-specific branches of the tree 
//  HltTree->Branch("NobjMCPart",&nmcpart,"NobjMCPart/I");
//  HltTree->Branch("MCPid",mcpid,"MCPid[NobjMCPart]/I");
//  HltTree->Branch("MCVtxX",mcvx,"MCVtxX[NobjMCPart]/F");
//  HltTree->Branch("MCVtxY",mcvy,"MCVtxY[NobjMCPart]/F");
//  HltTree->Branch("MCVtxZ",mcvz,"MCVtxZ[NobjMCPart]/F");
//  HltTree->Branch("MCPt",mcpt,"MCPt[NobjMCPart]/F");

 }

/* **Analyze the event** */

//// HWW->2l selection

void HLTHiggsTruth::analyzeHWW2l(const reco::CandidateView& mctruth,const reco::CaloMETCollection&
caloMet, const reco::TrackCollection& Tracks, const reco::MuonCollection& muonHandle, 
const reco::GsfElectronCollection& electronHandle, TTree* HltTree) {
  if (_Monte) {
  
   // if (&mctruth){
  
  
           
 
   //////////////////////
   ////    
   /////     //----  reco selection ---
  //////////////////////////////////
  
     std::map<double, reco::Muon> muonMap;
     std::map<double,reco::GsfElectron> electronMap;
    
      // keep globalmuons with pt>10 in eta<2.4
        for (size_t i = 0; i < muonHandle.size(); ++ i) {
           if ( (muonHandle)[i].isGlobalMuon() && (muonHandle)[i].pt()>10 &&
	    fabs((muonHandle)[i].eta())<2.4 ){ //&& (muonHandle)[i].isolationR03().sumPt<2 &&
	  // ((muonHandle)[i].isolationR03().emEt + (muonHandle)[i].isolationR03().hadEt)<5  ){
                muonMap[(muonHandle)[i].pt()]= (muonHandle)[i];
           }  
        }
	
	// keep electrons with pt>10, eta<2.4, H/E<0.05, 0.6<E/p<2.5
	   for (size_t ii = 0; ii < electronHandle.size(); ++ ii) {
           if (  (electronHandle)[ii].pt()>10 && fabs((electronHandle)[ii].eta())<2.4 &&
	   (electronHandle)[ii].hadronicOverEm()<0.05 &&
        (electronHandle)[ii].eSuperClusterOverP()>0.6 && (electronHandle)[ii].eSuperClusterOverP()<2.5 ){
                electronMap[(electronHandle)[ii].pt()]= (electronHandle)[ii];
           }  
        }
    
    /////// 
   
	  std::vector<reco::Muon> selected_muons;  
    for( std::map<double,reco::Muon>::reverse_iterator rit=muonMap.rbegin(); rit!=muonMap.rend(); ++rit){   
    selected_muons.push_back( (*rit).second );  // sort muons by pt
    }
    
   
	   std::vector<reco::GsfElectron> selected_electrons;
     for( std::map<double,reco::GsfElectron>::reverse_iterator rit=electronMap.rbegin(); rit!=electronMap.rend(); ++rit){
    selected_electrons.push_back( (*rit).second );  // sort electrons by pt
    }
 
  //------------------------------------
 
   /// Event classification ,take lepton pair with highest pt
    
   reco::Muon muon1, muon2;
   reco::GsfElectron electron1, electron2;
   
  if (selected_electrons.size() ==1 ) electron1 = selected_electrons[0];
  if (selected_electrons.size() > 1){
    electron1 = selected_electrons[0];
    electron2 = selected_electrons[1];
  }
  
  if (selected_muons.size() ==1 ) muon1 = selected_muons[0];
   if (selected_muons.size() > 1){
    muon1 = selected_muons[0];
    muon2 = selected_muons[1];
  }
   
       
     //  bool dimuEvent = false;
      // bool emuEvent  = false;
      // bool dielEvent = false;
       
       
        double ptel1=0.;
	double ptel2=0.;
	double ptmu1=0.;
	double ptmu2=0.;
	
       if (selected_electrons.size()==0) { ptel1 = 0;               ptel2 = 0;              }
       if (selected_electrons.size()==1) { ptel1 = electron1.pt();  ptel2 = 0;              }
       if (selected_electrons.size()>1)  { ptel1 = electron1.pt() ; ptel2 = electron2.pt();} 
       
       if (selected_muons.size()==0)     { ptmu1 = 0;               ptmu2 = 0;         }
       if (selected_muons.size()==1)     { ptmu1 = muon1.pt();      ptmu2 = 0;         }
       if (selected_muons.size()>1)      { ptmu1 = muon1.pt();      ptmu2 =muon2.pt();}
      
     
       
       if (selected_muons.size() + selected_electrons.size() > 1){
       
       if (ptel2 > ptmu1){ 
	 if (electron1.charge()*electron2.charge()<0 && electron1.pt()>20 /*&& electron2.pt()>20*/ ) {
	         //   dielEvent=true; 
	            isElecDecay_recoacc=true;
	            isMuonDecay_recoacc=false;
	            isEMuDecay_recoacc=false;
	       
	          Electron1 = selected_electrons[0];
                  Electron2 = selected_electrons[1];
		  
		  met_hwwdiel_ = caloMet[0].pt();
	       }
       }
       
       else if (ptmu2 > ptel1){
      //   if (muon1.charge()*muon2.charge()<0 && muon1.pt()>20 /*&& muon2.pt()>20*/ ) dimuEvent =true;
      if (muon1.charge()*muon2.charge()<0 && muon1.pt()>20 /*&& fabs(muon1.eta())<2.1*/ ){
                 // dimuEvent =true;
	          isElecDecay_recoacc=false;
	          isMuonDecay_recoacc=true;
	          isEMuDecay_recoacc=false;
	          
		   Muon1 = selected_muons[0];
                   Muon2 = selected_muons[1];
		   
		   met_hwwdimu_ =   caloMet[0].pt();
	     
	       }
       }
       
       else {
       
         // if (muon1.charge()*electron1.charge()<0 &&  (muon1.pt()>20 || electron1.pt()>20) ) emuEvent = true;
      if (muon1.charge()*electron1.charge()<0 &&  (muon1.pt()>20 || electron1.pt()>20) /*&&
      fabs(muon1.eta())<2.1 */) {
             //  emuEvent = true;
	           isElecDecay_recoacc=false;
	           isMuonDecay_recoacc=false;
	           isEMuDecay_recoacc=true;
	       
	   
		    Muon1     = selected_muons[0];
                    Electron1 = selected_electrons[0];
		    
		    met_hwwemu_ =  caloMet[0].pt();
	     }
       }
       
       }
       else{
       
       
               isElecDecay_recoacc=false;
	       isMuonDecay_recoacc=false;
	       isEMuDecay_recoacc=false;
       
       }
     
   
        
 //   }
  //  else {std::cout << "%HLTHiggsTruth -- No MC truth information" << std::endl;}
//    std::cout << "nleptons = " << nleptons << " " << isvisible_WW << std::endl;
  }
  
}

////////////////  HZZ-->4l selection

void HLTHiggsTruth::analyzeHZZ4l(const reco::CandidateView& mctruth, const reco::MuonCollection& muonHandle, 
const reco::GsfElectronCollection& electronHandle, TTree* HltTree) {
  if (_Monte) {
 
    
  //  if (&mctruth){
        
        //----selection based on reco---
  
   
        std::map<double,reco::Muon> muonMap;
   
      // keep globalmuons with pt>10, eta<2.4
        for (size_t i = 0; i < muonHandle.size(); ++ i) {
           if ( (muonHandle)[i].isGlobalMuon() && (muonHandle)[i].pt()>10 &&
	    fabs((muonHandle)[i].eta())<2.4 ){ 
                muonMap[(muonHandle)[i].pt()]= (muonHandle)[i];		
           }  
        }
	// keep electrons with pt>10, eta<2.4, H/E<0.05, 0.6<E/p<2.5
	   std::map<double,reco::GsfElectron> electronMap;
	   for (size_t ii = 0; ii < electronHandle.size(); ++ ii) {
           if (  (electronHandle)[ii].pt()>10 && fabs((electronHandle)[ii].eta())<2.4 &&
	   (electronHandle)[ii].hadronicOverEm()<0.05 &&
        (electronHandle)[ii].eSuperClusterOverP()>0.6 && (electronHandle)[ii].eSuperClusterOverP()<2.5){
                electronMap[(electronHandle)[ii].pt()]= (electronHandle)[ii];
	
           }  
        }
      
      /////
        std::vector<reco::Muon> selected_muons;  

     for( std::map<double,reco::Muon>::reverse_iterator rit=muonMap.rbegin(); rit!=muonMap.rend(); ++rit){
    selected_muons.push_back( (*rit).second );  // sort muons by pt
    }
      
   
         std::vector<reco::GsfElectron> selected_electrons;

     for( std::map<double,reco::GsfElectron>::reverse_iterator rit=electronMap.rbegin(); rit!=electronMap.rend(); ++rit){
    selected_electrons.push_back( (*rit).second );  // sort electrons by pt
    }
      
 
      ///////   4 lepton selection
       
       size_t posEle=0;
       size_t negEle=0;
       size_t posMu=0;
       size_t negMu=0;
       
                       
       for (size_t k=0; k<selected_muons.size();k++){    
         if (selected_muons[k].charge()>0) posMu++; //n muons pos charge
	 else if (selected_muons[k].charge()<0) negMu++;  // n muons neg charge         
       }
       
        for (size_t k=0; k<selected_electrons.size();k++){   
         if (selected_electrons[k].charge()>0) posEle++;  // n electrons pos charge
	 else if (selected_electrons[k].charge()<0) negEle++; // n electrons neg charge            
       }
      
     //----------
     /// Event selection : 2 pairs of opp charge leptons (4mu, 4e or 2e2mu)
    
    int nElectron=0;
    int nMuon=0;
   
    bool hzz2e2mu_decay = false;   // at least 4 reco muons in the event
    bool hzz4e_decay =false;       // at least 4 electrons in the event
    bool hzz4mu_decay=false;       // at least 2 muons and 2 electrons
    
 
    if (selected_muons.size()>=4)  hzz4mu_decay=true;
    else if (selected_electrons.size()>=4) hzz4e_decay=true;
    else if (selected_muons.size()>=2 && selected_electrons.size()>=2) hzz2e2mu_decay=true;
    
 
     if (hzz2e2mu_decay) {
        if ( posEle>=1 && negEle>=1 ) {
           nElectron=posEle+negEle;
          }
        if ( posMu>=1 && negMu>=1 ) {
            nMuon=posMu+negMu;
        }
    }
     else if (hzz4e_decay) {
         if ( posEle>=2 && negEle>=2 ) {
          nElectron=posEle+negEle;
        }
     }
     else if (hzz4mu_decay) {
        if ( posMu>=2 && negMu>=2 ) {
          nMuon=posMu+negMu;
    }  
  }
  
///////
  
  if (hzz2e2mu_decay && nElectron>=2 && nMuon>=2 ){  // at least 2 electrons and 2 muons
                                                       // with opp charge
                isEMuDecay_recoacc =true;
                 Muon1     = selected_muons[0];   
                 Electron1 = selected_electrons[0];  
   
   }
  else if (hzz4e_decay && nElectron>=4){
   isElecDecay_recoacc=true;
       Electron1 = selected_electrons[0];
       Electron2 = selected_electrons[1];
   
   }
  else if (hzz4mu_decay && nMuon>=4){
   isMuonDecay_recoacc =true;
       Muon1 = selected_muons[0];
       Muon2 = selected_muons[1];
   
   }
  else {
 
               isElecDecay_recoacc=false;
	       isMuonDecay_recoacc=false;
	       isEMuDecay_recoacc=false;
	       }
	       
 
      
   // }
    //else {std::cout << "%HLTHiggsTruth -- No MC truth information" << std::endl;}
//    std::cout << "nepair, nmupair = " << nepair << ", " << nmupair << " " << isvisible << std::endl;
  }
}

void HLTHiggsTruth::analyzeHgg(const reco::CandidateView& mctruth, const reco::PhotonCollection& photonHandle, TTree* HltTree) {
  if (_Monte) {
  //  int nphotons=0;  
    std::map<double,reco::Photon> photonMap;
  
     
  //  if (&mctruth){
    
  
        //keep reco photons with pt>20, eta<2.4
         for (size_t i = 0; i < photonHandle.size(); ++ i) {
           if ( (photonHandle)[i].pt()>20 && fabs((photonHandle)[i].eta())<2.4 ){ 
	        photonMap[(photonHandle)[i].pt()]= (photonHandle)[i];
            }
          }
		
      
       std::vector<reco::Photon> selected_photons;

        for( std::map<double,reco::Photon>::reverse_iterator rit=photonMap.rbegin(); rit!=photonMap.rend(); ++rit){
         selected_photons.push_back( (*rit).second );
        }

      
      // request 2 photons (or more)
    //  isvisible = nphotons > 1; 
      
     if (selected_photons.size()>1){    // at least 2 selected photons in the event
       
        isPhotonDecay_acc=true;  
        isvisible_reco= true;
   
         Photon1  = selected_photons[0];
         Photon2  = selected_photons[1];      
      }
        else{
          isPhotonDecay_acc=false;
          isvisible_reco=false;   
      }
      
    //}
    //else {std::cout << "%HLTHiggsTruth -- No MC truth information" << std::endl;}
//    std::cout << "nphotons = " << nphotons << " " << isvisible_gg << std::endl;
  }
}

void HLTHiggsTruth::analyzeA2mu(const reco::CandidateView& mctruth,TTree* HltTree) {
  if (_Monte) {
    int nmuons=0;
    if (&mctruth){
      for (size_t i = 0; i < mctruth.size(); ++ i) {
	const reco::Candidate & p = (mctruth)[i];
        int status = p.status();
	if (status==1) {
          int pid=p.pdgId();
	  bool ismuon = std::abs(pid)==13;
          bool inacceptance = (std::abs(p.eta()) < 2.4);
	  bool aboveptcut = (p.pt() > 3.0);
	  if (inacceptance && aboveptcut && ismuon) {
	    if (nmuons==0) {
	      nmuons=int(pid/std::abs(pid));
	    } else if (pid<0 && nmuons==1) {
	      nmuons=2;
	    } else if (pid>0 && nmuons==-1) {
	      nmuons=2;
            }
          }
        }
      }
      // request 2  opposite charge muons
      isvisible = nmuons==2; 
    }
    else {std::cout << "%HLTHiggsTruth -- No MC truth information" << std::endl;}
//    std::cout << "nmuons = " << nmuons << " " << isvisible_2mu << std::endl;
  }
}


void HLTHiggsTruth::analyzeH2tau(const reco::CandidateView& mctruth,TTree* HltTree) {
  if (_Monte) {
  //  int ntaus=0;
    int ngentau=0;
    int itauel=0;
    int itauelaccept=0;
    int itaumu=0;
    int itaumuaccept=0;
    //int itauq=0;
    
    std::vector<double> ptMuFromTau_,ptElFromTau_;
    std::vector<double> etaMuFromTau_,etaElFromTau_;
    
      
    if (&mctruth){
    
    
      for (size_t i = 0; i < mctruth.size(); ++ i) {
	const reco::Candidate & p = (mctruth)[i];
        int status = p.status();
        int pid=p.pdgId();
	
	//==========
	
	 if (status==3 && fabs(pid)==15) {
           ngentau++;
           bool elecdec = false, muondec = false;
           LeptonicTauDecay(p, elecdec, muondec);
      
          if (elecdec) {
            itauel++;
             if (PtElFromTau > 15 && fabs(EtaElFromTau)<2.4) {
	     itauelaccept++;	  // keep electrons from tau decay with pt>15
	     ptElFromTau_.push_back(PtElFromTau);
	     etaElFromTau_.push_back(EtaElFromTau);
	     }
          }
         if (muondec) {
            itaumu++;
            if (PtMuFromTau>15 && fabs(EtaMuFromTau)<2.4) {
	    itaumuaccept++;   // keep muons from tau decay with pt>15
	     ptMuFromTau_.push_back(PtMuFromTau);
	     etaMuFromTau_.push_back(EtaMuFromTau);
	    }
         } 
    }
    
}	
/*	
	
	//===============
	if (status==1 || status==2) {
         
	  bool istau = std::abs(pid)==15;
          bool inacceptance = (fabs(p.eta()) < 2.4);
	  bool aboveptcut = (p.pt() > 20.0);
	  if (inacceptance && aboveptcut && istau) {
	    if (ntaus==0) {
	      ntaus=int(pid/std::abs(pid));
	    } else if (pid<0 && ntaus==1) {
	      ntaus=2;
	    } else if (pid>0 && ntaus==-1) {
	      ntaus=2;
            }
          }
        }
      }
      // request 2  opposite charge taus
      isvisible = ntaus==2; */
      
      /// semileptonic decay study  H->2tau,  tau->mu & tau->had
      
       int iTauQ = ngentau - itauel - itaumu;
       if (ngentau==2 && itaumu==1 && iTauQ==1 && itaumuaccept==1){  //H->tautau->muq
    
             isMuonDecay_recoacc=true;
              ptMuMax = ptMuFromTau_[0];
             //  ptMuMin = 0.;     
             etaMuMax = etaMuFromTau_[0];
             // etaMuMin = 0.;   
        }
        else {isMuonDecay_recoacc=false;}
      
      /// H->2tau, tau->el & tau->had
       if (ngentau==2 && itauel==1 && iTauQ==1 && itauelaccept==1){  //H->tautau->eq
     //  if (ngentau==2 && itauel==2 && itauelaccept==2){    // dileptonic case
             isElecDecay_recoacc=true;
             ptElMax=ptElFromTau_[0];
          //   ptElMin=0.;
             etaElMax=etaElFromTau_[0];
          //   etaElMin=0.;
        }
         else { isElecDecay_recoacc=false;}
         
    }
    else {std::cout << "%HLTHiggsTruth -- No MC truth information" << std::endl;}
//    std::cout << "ntaus = " << ntaus << " " << isvisible << std::endl;
  }
}

void HLTHiggsTruth::analyzeHtaunu(const reco::CandidateView& mctruth,TTree* HltTree) {
  if (_Monte) {
    
    int ntaus=0;
  
     std::map<double,reco::Particle> tauMap;
    if (&mctruth){
      for (size_t i = 0; i < mctruth.size(); ++ i) {
	   const reco::Candidate & p = (mctruth)[i];
           int status = p.status();
	   int pid=p.pdgId();
	 
	// const reco::Candidate *m=p.mother();
	
	   if (status==1 || status==2) {
     
	      bool istau = std::abs(pid)==15;  	  
              bool inacceptance = (fabs(p.eta()) < 2.4);
	      bool aboveptcut = (p.pt() > 100);
	      if (inacceptance && aboveptcut && istau) {
	     	
	          ntaus++;
              }
           }
      }
      
      /////////////////////////////////////
      
    
  
    isvisible= (ntaus>0);   
    isTauDecay_acc = isvisible;
  
      
    }
    else {std::cout << "%HLTHiggsTruth -- No MC truth information" << std::endl;}
//    std::cout << "ntaus = " << ntaus << " " << isvisible_taunu << std::endl;
  }
}

void HLTHiggsTruth::analyzeHinv(const reco::CandidateView& mctruth,TTree* HltTree) {
  if (_Monte) {
    if (&mctruth){
      isvisible = true; 
    } else {
      std::cout << "%HLTHiggsTruth -- No MC truth information" << std::endl;
    }
//    std::cout << "Invisible: MC exists, accept " << std::endl;
  }
}

void HLTHiggsTruth::LeptonicTauDecay(const reco::Candidate& tau, bool& elecdec, bool& muondec)

{
  
  //if (tau.begin() == tau.end()) std::cout << "No_llega_a_entrar_en_el_bucle" << std::endl;
  // loop on tau decays, check for an electron or muon
  for(reco::Candidate::const_iterator daughter=tau.begin();daughter!=tau.end(); ++daughter){
    //cout << "daughter_x" << std::endl;
    // if the tau daughter is a tau, it means the particle has still to be propagated.
    // In that case, return the result of the same method on that daughter.
    if(daughter->pdgId()==tau.pdgId()) return LeptonicTauDecay(*daughter, elecdec, muondec);
    // check for leptons
    elecdec |= std::abs(daughter->pdgId())==11;
    muondec |= std::abs(daughter->pdgId())==13;
    
    if (std::abs(daughter->pdgId())==11) {
      PtElFromTau = daughter->pt();
      EtaElFromTau = daughter->eta();
    }
    
    if (std::abs(daughter->pdgId())==13){
    PtMuFromTau = daughter->pt();
    EtaMuFromTau = daughter->eta();
    }
  }
    
}







