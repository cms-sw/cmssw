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

#include "HLTrigger/HLTanalyzers/interface/HLTJets.h"

HLTJets::HLTJets() {
    evtCounter=0;
    
    //set parameter defaults 
    _Monte=false;
    _Debug=false;
    _CalJetMin=0.;
    _GenJetMin=0.;
}

/*  Setup the analysis to put the branch-variables into the tree. */
void HLTJets::setup(const edm::ParameterSet& pSet, TTree* HltTree) {
    
    edm::ParameterSet myJetParams = pSet.getParameter<edm::ParameterSet>("RunParameters") ;
    std::vector<std::string> parameterNames = myJetParams.getParameterNames() ;
    
    for ( std::vector<std::string>::iterator iParam = parameterNames.begin();
         iParam != parameterNames.end(); iParam++ ){
        if  ( (*iParam) == "Monte" ) _Monte =  myJetParams.getParameter<bool>( *iParam );
        else if ( (*iParam) == "Debug" ) _Debug =  myJetParams.getParameter<bool>( *iParam );
        else if ( (*iParam) == "CalJetMin" ) _CalJetMin =  myJetParams.getParameter<double>( *iParam );
        else if ( (*iParam) == "GenJetMin" ) _GenJetMin =  myJetParams.getParameter<double>( *iParam );
    }

    const int kMaxRecoPFJet = 10000;
    jpfrecopt=new float[kMaxRecoPFJet];
    jpfrecophi=new float[kMaxRecoPFJet];
    jpfrecoeta=new float[kMaxRecoPFJet];
    jpfreconeutralHadronFraction=new float[kMaxRecoPFJet];
    jpfreconeutralEMFraction=new float[kMaxRecoPFJet];
    jpfrecochargedHadronFraction=new float[kMaxRecoPFJet];
    jpfrecochargedEMFraction=new float[kMaxRecoPFJet];
    jpfreconeutralMultiplicity=new int[kMaxRecoPFJet];
    jpfrecochargedMultiplicity=new int[kMaxRecoPFJet];
    
    const int kMaxJetCal = 10000;
    jcalpt = new float[kMaxJetCal];
    jcalphi = new float[kMaxJetCal];
    jcaleta = new float[kMaxJetCal];
    jcale = new float[kMaxJetCal];
    jcalemf = new float[kMaxJetCal]; 
    jcaln90 = new float[kMaxJetCal]; 
    
    jcorcalpt = new float[kMaxJetCal]; 
    jcorcalphi = new float[kMaxJetCal]; 
    jcorcaleta = new float[kMaxJetCal]; 
    jcorcale = new float[kMaxJetCal]; 
    jcorcalemf = new float[kMaxJetCal]; 
    jcorcaln90 = new float[kMaxJetCal]; 
    
    const int kMaxJetgen = 10000;
    jgenpt = new float[kMaxJetgen];
    jgenphi = new float[kMaxJetgen];
    jgeneta = new float[kMaxJetgen];
    jgene = new float[kMaxJetgen];
    const int kMaxTower = 10000;
    towet = new float[kMaxTower];
    toweta = new float[kMaxTower];
    towphi = new float[kMaxTower];
    towen = new float[kMaxTower];
    towem = new float[kMaxTower];
    towhd = new float[kMaxTower];
    towoe = new float[kMaxTower];
    const int kMaxTau = 500;
    l2tauemiso = new float[kMaxTau];
    l25tauPt = new float[kMaxTau];
    l3tautckiso = new int[kMaxTau];
    tauEta = new float[kMaxTau];
    tauPt = new float[kMaxTau];
    tauPhi = new float[kMaxTau];
    
    const int kMaxPFTau = 500;
    ohpfTauEta         =  new float[kMaxPFTau];
    ohpfTauPhi         =  new float[kMaxPFTau];
    ohpfTauPt          =  new float[kMaxPFTau];
    ohpfTauJetPt       =  new float[kMaxPFTau];
    ohpfTauLeadTrackPt =  new float[kMaxPFTau];
    ohpfTauLeadPionPt  =  new float[kMaxPFTau];
    ohpfTauTrkIso      =  new float[kMaxPFTau];
    ohpfTauGammaIso    =  new float[kMaxPFTau];
    
    recopfTauEta 	 =  new float[kMaxPFTau];
    recopfTauPhi 	 =  new float[kMaxPFTau];
    recopfTauPt  	 =  new float[kMaxPFTau];
    recopfTauJetPt	 =  new float[kMaxPFTau];
    recopfTauLeadTrackPt =  new float[kMaxPFTau];
    recopfTauLeadPionPt  =  new float[kMaxPFTau];
    recopfTauTrkIso	 =  new int[kMaxPFTau];
    recopfTauGammaIso	 =  new int[kMaxPFTau];
    recopfTauDiscrByTancOnePercent     =  new float[kMaxPFTau];
    recopfTauDiscrByTancHalfPercent    =  new float[kMaxPFTau];
    recopfTauDiscrByTancQuarterPercent =  new float[kMaxPFTau]; 
    recopfTauDiscrByTancTenthPercent   =  new float[kMaxPFTau];
    recopfTauDiscrByIso        =  new float[kMaxPFTau]; 
    recopfTauDiscrAgainstMuon  =  new float[kMaxPFTau];
    recopfTauDiscrAgainstElec  =  new float[kMaxPFTau];
    
    pfMHT   = -100;    
    const int kMaxPFJet = 500;
    pfJetEta         = new float[kMaxPFJet];
    pfJetPhi         = new float[kMaxPFJet];
    pfJetPt         = new float[kMaxPFJet];
    
    // Jet- MEt-specific branches of the tree 
    HltTree->Branch("NrecoJetCal",&njetcal,"NrecoJetCal/I");
    HltTree->Branch("NrecoJetGen",&njetgen,"NrecoJetGen/I");
    HltTree->Branch("NrecoTowCal",&ntowcal,"NrecoTowCal/I");
    HltTree->Branch("recoJetCalPt",jcalpt,"recoJetCalPt[NrecoJetCal]/F");
    HltTree->Branch("recoJetCalPhi",jcalphi,"recoJetCalPhi[NrecoJetCal]/F");
    HltTree->Branch("recoJetCalEta",jcaleta,"recoJetCalEta[NrecoJetCal]/F");
    HltTree->Branch("recoJetCalE",jcale,"recoJetCalE[NrecoJetCal]/F");
    HltTree->Branch("recoJetCalEMF",jcalemf,"recoJetCalEMF[NrecoJetCal]/F");
    HltTree->Branch("recoJetCalN90",jcaln90,"recoJetCalN90[NrecoJetCal]/F");
    
    HltTree->Branch("recoJetGenPt",jgenpt,"recoJetGenPt[NrecoJetGen]/F");
    HltTree->Branch("recoJetGenPhi",jgenphi,"recoJetGenPhi[NrecoJetGen]/F");
    HltTree->Branch("recoJetGenEta",jgeneta,"recoJetGenEta[NrecoJetGen]/F");
    HltTree->Branch("recoJetGenE",jgene,"recoJetGenE[NrecoJetGen]/F");
    HltTree->Branch("recoTowEt",towet,"recoTowEt[NrecoTowCal]/F");
    HltTree->Branch("recoTowEta",toweta,"recoTowEta[NrecoTowCal]/F");
    HltTree->Branch("recoTowPhi",towphi,"recoTowPhi[NrecoTowCal]/F");
    HltTree->Branch("recoTowE",towen,"recoTowE[NrecoTowCal]/F");
    HltTree->Branch("recoTowEm",towem,"recoTowEm[NrecoTowCal]/F");
    HltTree->Branch("recoTowHad",towhd,"recoTowHad[NrecoTowCal]/F");
    HltTree->Branch("recoTowOE",towoe,"recoTowOE[NrecoTowCal]/F");
    HltTree->Branch("recoMetCal",&mcalmet,"recoMetCal/F");
    HltTree->Branch("recoMetCalPhi",&mcalphi,"recoMetCalPhi/F");
    HltTree->Branch("recoMetCalSum",&mcalsum,"recoMetCalSum/F");
    HltTree->Branch("recoMetGen",&mgenmet,"recoMetGen/F");
    HltTree->Branch("recoMetGenPhi",&mgenphi,"recoMetGenPhi/F");
    HltTree->Branch("recoMetGenSum",&mgensum,"recoMetGenSum/F");
    HltTree->Branch("recoHTCal",&htcalet,"recoHTCal/F");
    HltTree->Branch("recoHTCalPhi",&htcalphi,"recoHTCalPhi/F");
    HltTree->Branch("recoHTCalSum",&htcalsum,"recoHTCalSum/F");
    //for(int ieta=0;ieta<NETA;ieta++){std::cout << " ieta " << ieta << " eta min " << CaloTowerEtaBoundries[ieta] <<std::endl;}
    
    HltTree->Branch("NrecoJetCorCal",&ncorjetcal,"NrecoJetCorCal/I"); 
    HltTree->Branch("recoJetCorCalPt",jcorcalpt,"recoJetCorCalPt[NrecoJetCorCal]/F"); 
    HltTree->Branch("recoJetCorCalPhi",jcorcalphi,"recoJetCorCalPhi[NrecoJetCorCal]/F"); 
    HltTree->Branch("recoJetCorCalEta",jcorcaleta,"recoJetCorCalEta[NrecoJetCorCal]/F"); 
    HltTree->Branch("recoJetCorCalE",jcorcale,"recoJetCorCalE[NrecoJetCorCal]/F"); 
    HltTree->Branch("recoJetCorCalEMF",jcorcalemf,"recoJetCorCalEMF[NrecoJetCorCal]/F");
    HltTree->Branch("recoJetCorCalN90",jcorcaln90,"recoJetCorCalN90[NrecoJetCorCal]/F");
    
    // Taus
    HltTree->Branch("NohTau",&nohtau,"NohTau/I");
    HltTree->Branch("ohTauEta",tauEta,"ohTauEta[NohTau]/F");
    HltTree->Branch("ohTauPhi",tauPhi,"ohTauPhi[NohTau]/F");
    HltTree->Branch("ohTauPt",tauPt,"ohTauPt[NohTau]/F");
    HltTree->Branch("ohTauEiso",l2tauemiso,"ohTauEiso[NohTau]/F");
    HltTree->Branch("ohTauL25Tpt",l25tauPt,"ohTauL25Tpt[NohTau]/F");
    HltTree->Branch("ohTauL3Tiso",l3tautckiso,"ohTauL3Tiso[NohTau]/I");

    //ohpfTaus
    HltTree->Branch("NohpfTau",&nohPFTau,"NohpfTau/I");
    HltTree->Branch("ohpfTauPt",ohpfTauPt,"ohpfTauPt[NohpfTau]/F");
    HltTree->Branch("ohpfTauEta",ohpfTauEta,"ohpfTauEta[NohpfTau]/F");
    HltTree->Branch("ohpfTauPhi",ohpfTauPhi,"ohpfTauPhi[NohpfTau]/F");
    HltTree->Branch("ohpfTauLeadTrackPt",ohpfTauLeadTrackPt,"ohpfTauLeadTrackPt[NohpfTau]/F");
    HltTree->Branch("ohpfTauLeadPionPt",ohpfTauLeadPionPt,"ohpfTauLeadPionPt[NohpfTau]/F");
    HltTree->Branch("ohpfTauTrkIso",ohpfTauTrkIso,"ohpfTauTrkIso[NohpfTau]/F");
    HltTree->Branch("ohpfTauGammaIso",ohpfTauGammaIso,"ohpfTauGammaIso[NohpfTau]/F");
    HltTree->Branch("ohpfTauJetPt",ohpfTauJetPt,"ohpfTauJetPt[NohpfTau]/F");    
   
   //Reco PFTaus
    nRecoPFTau = 0;
    HltTree->Branch("NRecoPFTau",&nRecoPFTau,"NRecoPFTau/I");
    HltTree->Branch("recopfTauPt",recopfTauPt,"recopfTauPt[NRecoPFTau]/F");
    HltTree->Branch("recopfTauEta",recopfTauEta,"recopfTauEta[NRecoPFTau]/F");
    HltTree->Branch("recopfTauPhi",recopfTauPhi,"recopfTauPhi[NRecoPFTau]/F");
    HltTree->Branch("recopfTauLeadTrackPt",recopfTauLeadTrackPt,"recopfTauLeadTrackPt[NRecoPFTau]/F");
    HltTree->Branch("recopfTauLeadPionPt",recopfTauLeadPionPt,"recopfTauLeadPionPt[NRecoPFTau]/F");
    HltTree->Branch("recopfTauTrkIso",recopfTauTrkIso,"recopfTauTrkIso[NRecoPFTau]/I");
    HltTree->Branch("recopfTauGammaIso",recopfTauGammaIso,"recopfTauGammaIso[NRecoPFTau]/I");
    HltTree->Branch("recopfTauJetPt",recopfTauJetPt,"recopfTauJetPt[NRecoPFTau]/F");   
    HltTree->Branch("recopfTauDiscrByTancOnePercent",recopfTauDiscrByTancOnePercent,"recopfTauDiscrByTancOnePercent[NRecoPFTau]/F");   
    HltTree->Branch("recopfTauDiscrByTancHalfPercent",recopfTauDiscrByTancHalfPercent,"recopfTauDiscrByTancHalfPercent[NRecoPFTau]/F");   
    HltTree->Branch("recopfTauDiscrByTancQuarterPercent",recopfTauDiscrByTancQuarterPercent,"recopfTauDiscrByTancQuarterPercent[NRecoPFTau]/F");   
    HltTree->Branch("recopfTauDiscrByTancTenthPercent",recopfTauDiscrByTancTenthPercent,"recopfTauDiscrByTancTenthPercent[NRecoPFTau]/F");	 
    HltTree->Branch("recopfTauDiscrByIso",recopfTauDiscrByIso,"recopfTauDiscrByIso[NRecoPFTau]/F");   
    HltTree->Branch("recopfTauDiscrAgainstMuon",recopfTauDiscrAgainstMuon,"recopfTauDiscrAgainstMuon[NRecoPFTau]/F");
    HltTree->Branch("recopfTauDiscrAgainstElec",recopfTauDiscrAgainstElec,"recopfTauDiscrAgainstElec[NRecoPFTau]/F");
  
    //PFJets
    HltTree->Branch("pfMHT",&pfMHT,"pfMHT/F");
    HltTree->Branch("NohPFJet",&nohPFJet,"NohPFJet/I");
    HltTree->Branch("pfJetPt",pfJetPt,"pfJetPt[NohPFJet]/F");
    HltTree->Branch("pfJetEta",pfJetEta,"pfJetEta[NohPFJet]/F");
    HltTree->Branch("pfJetPhi",pfJetPhi,"pfJetPhi[NohPFJet]/F");

    //RECO PFJets
    HltTree->Branch("nrpj",&nrpj,"nrpj/I");
    HltTree->Branch("recopfJetpt",                    jpfrecopt,                     "recopfJetpt[nrpj]/F");
    HltTree->Branch("recopfJetphi",                   jpfrecophi,                    "recopfJetphi[nrpj]/F");
    HltTree->Branch("recopfJeteta",                   jpfrecoeta,                    "recopfJeteta[nrpj]/F");
    HltTree->Branch("recopfJetneutralHadronFraction", jpfreconeutralHadronFraction,  "recopfJetneutralHadronFraction[nrpj]/F");
    HltTree->Branch("recopfJetneutralEMFraction",     jpfreconeutralEMFraction,      "recopfJetneutralEMFraction[nrpj]/F");
    HltTree->Branch("recopfJetchargedHadronFraction", jpfrecochargedHadronFraction,  "recopfJetchargedHadronFraction[nrpj]/F");
    HltTree->Branch("recopfJetchargedEMFraction",     jpfrecochargedEMFraction,      "recopfJetchargedEMFraction[nrpj]/F");
    HltTree->Branch("recopfJetneutralMultiplicity",   jpfreconeutralMultiplicity,    "recopfJetneutralMultiplicity[nrpj]/I");
    HltTree->Branch("recopfJetchargedMultiplicity",   jpfrecochargedMultiplicity,    "recopfJetchargedMultiplicity[nrpj]/I"); 
    
}

/* **Analyze the event** */
void HLTJets::analyze(const edm::Handle<reco::CaloJetCollection>      & calojets,
                      const edm::Handle<reco::CaloJetCollection>      & calocorjets,
                      const edm::Handle<reco::GenJetCollection>       & genjets,
                      const edm::Handle<reco::CaloMETCollection>      & recmets,
                      const edm::Handle<reco::GenMETCollection>       & genmets,
                      const edm::Handle<reco::METCollection>          & ht,
                      const edm::Handle<reco::HLTTauCollection>       & taujets,
                      const edm::Handle<reco::PFTauCollection>        & pfTaus,
                      const edm::Handle<reco::PFJetCollection>        & pfJets,
                      const edm::Handle<reco::PFTauCollection>        & recoPfTaus,  
		      const edm::Handle<reco::PFTauDiscriminator>	      & theRecoPFTauDiscrByTanCOnePercent,
		      const edm::Handle<reco::PFTauDiscriminator>	      & theRecoPFTauDiscrByTanCHalfPercent,
		      const edm::Handle<reco::PFTauDiscriminator>	      & theRecoPFTauDiscrByTanCQuarterPercent,
		      const edm::Handle<reco::PFTauDiscriminator>	      & theRecoPFTauDiscrByTanCTenthPercent,			  
		      const edm::Handle<reco::PFTauDiscriminator>	      & theRecoPFTauDiscrByIsolation,
		      const edm::Handle<reco::PFTauDiscriminator>	      & theRecoPFTauDiscrAgainstElec,
		      const edm::Handle<reco::PFTauDiscriminator>	      & theRecoPFTauDiscrAgainstMuon,
                      const edm::Handle<reco::PFJetCollection>        & recoPFJets,
		      const edm::Handle<CaloTowerCollection>          & caloTowers,
                      double thresholdForSavingTowers, 
                      double		    minPtCH,
                      double		   minPtGamma,
                      TTree * HltTree) {
    
    if (_Debug) std::cout << " Beginning HLTJets " << std::endl;
    
    //initialize branch variables
    njetcal=0; ncorjetcal=0; njetgen=0;ntowcal=0;
    mcalmet=0.; mcalphi=0.;
    mgenmet=0.; mgenphi=0.;
    htcalet=0.,htcalphi=0.,htcalsum=0.;
    
    if (calojets.isValid()) {
        reco::CaloJetCollection mycalojets;
        mycalojets=*calojets;
        std::sort(mycalojets.begin(),mycalojets.end(),PtGreater());
        typedef reco::CaloJetCollection::const_iterator cjiter;
        int jcal=0;
        for ( cjiter i=mycalojets.begin(); i!=mycalojets.end(); i++) {
            
            if (i->pt()>_CalJetMin){
                jcalpt[jcal] = i->pt();
                jcalphi[jcal] = i->phi();
                jcaleta[jcal] = i->eta();
                jcale[jcal] = i->energy();
                jcalemf[jcal] = i->emEnergyFraction();
                jcaln90[jcal] = i->n90();
                jcal++;
            }
            
        }
        njetcal = jcal;
    }
    else {njetcal = 0;}
    
    if (calocorjets.isValid()) {
        reco::CaloJetCollection mycalocorjets;
        mycalocorjets=*calocorjets;
        std::sort(mycalocorjets.begin(),mycalocorjets.end(),PtGreater());
        typedef reco::CaloJetCollection::const_iterator ccorjiter;
        int jcorcal=0;
        for ( ccorjiter i=mycalocorjets.begin(); i!=mycalocorjets.end(); i++) {
            
            if (i->pt()>_CalJetMin){
                jcorcalpt[jcorcal] = i->pt();
                jcorcalphi[jcorcal] = i->phi();
                jcorcaleta[jcorcal] = i->eta();
                jcorcale[jcorcal] = i->energy();
                jcorcalemf[jcorcal] = i->emEnergyFraction();
                jcorcaln90[jcorcal] = i->n90();
                jcorcal++;
            }
            
        }
        ncorjetcal = jcorcal;
    }
    else {ncorjetcal = 0;}
    
    if (caloTowers.isValid()) {
        //    ntowcal = caloTowers->size();
        int jtow = 0;
        for ( CaloTowerCollection::const_iterator tower=caloTowers->begin(); tower!=caloTowers->end(); tower++) {
            if(tower->energy() > thresholdForSavingTowers)
            {
                towet[jtow] = tower->et();
                toweta[jtow] = tower->eta();
                towphi[jtow] = tower->phi();
                towen[jtow] = tower->energy();
                towem[jtow] = tower->emEnergy();
                towhd[jtow] = tower->hadEnergy();
                towoe[jtow] = tower->outerEnergy();
                jtow++;
            }
        }
        ntowcal = jtow;
    }
    else {ntowcal = 0;}
    
    if (recmets.isValid()) {
        typedef reco::CaloMETCollection::const_iterator cmiter;
        for ( cmiter i=recmets->begin(); i!=recmets->end(); i++) {
            mcalmet = i->pt();
            mcalphi = i->phi();
            mcalsum = i->sumEt();
        }
    }
    
    if (ht.isValid()) {
        typedef reco::METCollection::const_iterator iter;
        for ( iter i=ht->begin(); i!=ht->end(); i++) {
            htcalet = i->pt();
            htcalphi = i->phi();
            htcalsum = i->sumEt();
        }
    }
    
    if (_Monte){
        
        if (genjets.isValid()) {
            reco::GenJetCollection mygenjets;
            mygenjets=*genjets;
            std::sort(mygenjets.begin(),mygenjets.end(),PtGreater());
            typedef reco::GenJetCollection::const_iterator gjiter;
            int jgen=0;
            for ( gjiter i=mygenjets.begin(); i!=mygenjets.end(); i++) {
                
                if (i->pt()>_GenJetMin){
                    jgenpt[jgen] = i->pt();
                    jgenphi[jgen] = i->phi();
                    jgeneta[jgen] = i->eta();
                    jgene[jgen] = i->energy();
                    jgen++;
                }
                
            }
            njetgen = jgen;
        }
        else {njetgen = 0;}
        
        if (genmets.isValid()) {
            typedef reco::GenMETCollection::const_iterator gmiter;
            for ( gmiter i=genmets->begin(); i!=genmets->end(); i++) {
                mgenmet = i->pt();
                mgenphi = i->phi();
                mgensum = i->sumEt();
            }
        }
        
    }
    
    
    /////////////////////////////// Open-HLT Taus ///////////////////////////////
    
    if (taujets.isValid()) {      
        nohtau = taujets->size();
        reco::HLTTauCollection mytaujets;
        mytaujets=*taujets;
        std::sort(mytaujets.begin(),mytaujets.end(),GetPtGreater());
        typedef reco::HLTTauCollection::const_iterator tauit;
        int itau=0;
        for(tauit i=mytaujets.begin(); i!=mytaujets.end(); i++){
            //Ask for Eta,Phi and Et of the tau:
            tauEta[itau] = i->getEta();
            tauPhi[itau] = i->getPhi();
            tauPt[itau] = i->getPt();
            //Ask for L2 EMIsolation cut: Nominal cut : < 5
            l2tauemiso[itau] = i->getEMIsolationValue();
            //Get L25 LeadTrackPt : Nominal cut : > 20 GeV
            l25tauPt[itau] = i->getL25LeadTrackPtValue();
            //Get TrackIsolation response (returns 0 = failed or 1= passed)
            l3tautckiso[itau] = i->getL3TrackIsolationResponse();
            //MET : > 65
            itau++;
        }      
    }
    else {nohtau = 0;}

    
    ////////////////Particle Flow Taus ////////////////////////////////////
    if(pfTaus.isValid()) {
        //float minTrkPt = minPtCH;
        //float minGammaPt = minPtGamma;
        nohPFTau  = pfTaus->size();
        reco::PFTauCollection taus = *pfTaus;
        std::sort(taus.begin(),taus.end(),GetPFPtGreater());
        typedef reco::PFTauCollection::const_iterator pftauit;
        int ipftau=0;
        float pfMHTx = 0;
        float pfMHTy = 0;
        for(pftauit i=taus.begin(); i!=taus.end(); i++){
            //Ask for Eta,Phi and Et of the tau:
            ohpfTauEta[ipftau] = i->eta();
            ohpfTauPhi[ipftau] = i->phi();
            ohpfTauPt[ipftau] = i->pt();
            ohpfTauJetPt[ipftau] = i->pfTauTagInfoRef()->pfjetRef()->pt();

            pfMHTx = pfMHTx + i->pfTauTagInfoRef()->pfjetRef()->px();
            pfMHTy = pfMHTy + i->pfTauTagInfoRef()->pfjetRef()->py();
            
  /*
            if( (i->leadPFCand()).isNonnull())
                pfTauLeadPionPt[ipftau] = i->leadPFCand()->pt();            
*/
            if( (i->leadPFNeutralCand()).isNonnull())
                ohpfTauLeadPionPt[ipftau] = i->leadPFNeutralCand()->pt();        
            if((i->leadPFChargedHadrCand()).isNonnull())
                ohpfTauLeadTrackPt[ipftau] = i->leadPFChargedHadrCand()->pt();
            float maxPtTrkIso = 0;
            for (unsigned int iTrk = 0; iTrk < i->isolationPFChargedHadrCands().size(); iTrk++)
            {
                if(i->isolationPFChargedHadrCands()[iTrk]->pt() > maxPtTrkIso) maxPtTrkIso = i->isolationPFChargedHadrCands()[iTrk]->pt();
            }
                
            ohpfTauTrkIso[ipftau] = maxPtTrkIso;
            float maxPtGammaIso = 0;
            for (unsigned int iGamma = 0; iGamma < i->isolationPFGammaCands().size(); iGamma++)
            {
                if(i->isolationPFGammaCands()[iGamma]->pt() > maxPtGammaIso) maxPtGammaIso = i->isolationPFGammaCands()[iGamma]->pt();
            }                        
            ohpfTauGammaIso[ipftau] = maxPtGammaIso;
            ipftau++;
        } 
        pfMHT = sqrt(pfMHTx*pfMHTx + pfMHTy*pfMHTy);
        
    }
    
    ////////////////Reco Particle Flow Taus ////////////////////////////////////
      
    if(recoPfTaus.isValid()) {
        float minTrkPt = minPtCH;
        float minGammaPt = minPtGamma;
        nRecoPFTau  = pfTaus->size();
        reco::PFTauCollection taus = *recoPfTaus;
        std::sort(taus.begin(),taus.end(),GetPFPtGreater());
        typedef reco::PFTauCollection::const_iterator pftauit;
        int ipftau=0;
        
        for(pftauit i=taus.begin(); i!=taus.end(); i++){
            //Ask for Eta,Phi and Et of the tau:
            recopfTauEta[ipftau] = i->eta();
            recopfTauPhi[ipftau] = i->phi();
            recopfTauPt[ipftau]  = i->pt();
	    recopfTauJetPt[ipftau] = i->jetRef()->pt();

            if( (i->leadPFNeutralCand()).isNonnull())
                recopfTauLeadPionPt[ipftau] = i->leadPFNeutralCand()->pt();        
            if((i->leadPFChargedHadrCand()).isNonnull())
                recopfTauLeadTrackPt[ipftau] = i->leadPFChargedHadrCand()->pt();
            int myTrks=0;
            for (unsigned int iTrk = 0; iTrk < i->isolationPFChargedHadrCands().size(); iTrk++)
            {
                if(i->isolationPFChargedHadrCands()[iTrk]->pt() > minTrkPt) myTrks++;
            }
               
            recopfTauTrkIso[ipftau] = myTrks;
            int myGammas=0;
            for (unsigned int iGamma = 0; iGamma < i->isolationPFGammaCands().size(); iGamma++)
            {
                if(i->isolationPFGammaCands()[iGamma]->pt() > minGammaPt) myGammas++;
            }                        
            recopfTauGammaIso[ipftau] = myGammas;
	    
	    const reco::PFTauRef thisTauRef(recoPfTaus,ipftau);
            
	    if(theRecoPFTauDiscrByTanCOnePercent.isValid()){
	    recopfTauDiscrByTancOnePercent[ipftau] = (*theRecoPFTauDiscrByTanCOnePercent)[thisTauRef];}
	    if(theRecoPFTauDiscrByIsolation.isValid()){ 
	    recopfTauDiscrByIso[ipftau] = (*theRecoPFTauDiscrByIsolation)[thisTauRef];} 
	    if(theRecoPFTauDiscrAgainstMuon.isValid()){
	    recopfTauDiscrAgainstMuon[ipftau] = (*theRecoPFTauDiscrAgainstMuon)[thisTauRef];}
	    if(theRecoPFTauDiscrAgainstElec.isValid()){
	    recopfTauDiscrAgainstElec[ipftau] = (*theRecoPFTauDiscrAgainstElec)[thisTauRef];}
	    if(theRecoPFTauDiscrByTanCHalfPercent.isValid()){
	    recopfTauDiscrByTancHalfPercent[ipftau] = (*theRecoPFTauDiscrByTanCHalfPercent)[thisTauRef];}
	    if(theRecoPFTauDiscrByTanCQuarterPercent.isValid()){
	    recopfTauDiscrByTancQuarterPercent[ipftau] = (*theRecoPFTauDiscrByTanCQuarterPercent)[thisTauRef];}
	    if(theRecoPFTauDiscrByTanCTenthPercent.isValid()){
	    recopfTauDiscrByTancTenthPercent[ipftau] = (*theRecoPFTauDiscrByTanCTenthPercent)[thisTauRef];}

	    ipftau++;
        }        
    }
   
    ////////////////Particle Flow Jets ////////////////////////////////////
    if(pfJets.isValid()) {
        nohPFJet  = pfJets->size();
        reco::PFJetCollection Jets = *pfJets;
        std::sort(Jets.begin(),Jets.end(),GetPFPtGreater());
        typedef reco::PFJetCollection::const_iterator pfJetit;
        int ipfJet=0;
        float pfMHTx = 0;
        float pfMHTy = 0;
        for(pfJetit i=Jets.begin(); i!=Jets.end(); i++){
            //Ask for Eta,Phi and Et of the Jet:
            pfJetEta[ipfJet] = i->eta();
            pfJetPhi[ipfJet] = i->phi();
            pfJetPt[ipfJet] = i->pt();           
            
            pfMHTx = pfMHTx + i->px();
            pfMHTy = pfMHTy + i->py();
            ipfJet++;   
        } 
        pfMHT = sqrt(pfMHTx*pfMHTx + pfMHTy*pfMHTy);
        
    }
    
    //////////////// RECO Particle Flow Jets ////////////////////////////////////
    nrpj = 0;
    if(recoPFJets.isValid()){
	    nrpj = recoPFJets->size();
	    reco::PFJetCollection Jets = *recoPFJets;
	    std::sort(Jets.begin(),Jets.end(),GetPFPtGreater());
	    typedef reco::PFJetCollection::const_iterator pfJetit;
	    int ipfJet=0;
	    for(pfJetit i=Jets.begin(); i!=Jets.end(); i++){
		    //Ask for Eta,Phi and Et of the Jet:
		    jpfrecoeta[ipfJet] = i->eta();
		    jpfrecophi[ipfJet] = i->phi();
		    jpfrecopt[ipfJet] = i->pt();           
		    jpfreconeutralHadronFraction[ipfJet] = i->neutralHadronEnergyFraction ();
		    jpfrecochargedHadronFraction[ipfJet] = i->chargedHadronEnergyFraction ();
		    jpfreconeutralMultiplicity[ipfJet] = i->neutralMultiplicity ();
		    jpfrecochargedMultiplicity[ipfJet] = i->chargedMultiplicity ();
		    jpfreconeutralEMFraction[ipfJet] = i->neutralEmEnergyFraction ();
		    jpfrecochargedEMFraction[ipfJet] = i->chargedEmEnergyFraction ();

		    ipfJet++;   
	    } 

    }
    
}
