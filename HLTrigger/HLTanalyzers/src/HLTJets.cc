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
#include <set>

#include "HLTrigger/HLTanalyzers/interface/HLTJets.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

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

    jetID = new reco::helper::JetIDHelper(pSet.getParameter<edm::ParameterSet>("JetIDParams"));

    const int kMaxRecoPFJet = 10000;
    jpfrecopt=new float[kMaxRecoPFJet];
    jpfrecoe=new float[kMaxRecoPFJet];
    jpfrecophi=new float[kMaxRecoPFJet];
    jpfrecoeta=new float[kMaxRecoPFJet];
    jpfreconeutralHadronFraction=new float[kMaxRecoPFJet];
    jpfreconeutralEMFraction=new float[kMaxRecoPFJet];
    jpfrecochargedHadronFraction=new float[kMaxRecoPFJet];
    jpfrecochargedEMFraction=new float[kMaxRecoPFJet];
    jpfreconeutralMultiplicity=new int[kMaxRecoPFJet];
    jpfrecochargedMultiplicity=new int[kMaxRecoPFJet];
    
    const int kMaxJetCal = 10000;
    jhcalpt = new float[kMaxJetCal];
    jhcalphi = new float[kMaxJetCal];
    jhcaleta = new float[kMaxJetCal];
    jhcale = new float[kMaxJetCal];
    jhcalemf = new float[kMaxJetCal]; 
    jhcaln90 = new float[kMaxJetCal]; 
    jhcaln90hits = new float[kMaxJetCal]; 
    
    jhcorcalpt = new float[kMaxJetCal]; 
    jhcorcalphi = new float[kMaxJetCal]; 
    jhcorcaleta = new float[kMaxJetCal]; 
    jhcorcale = new float[kMaxJetCal]; 
    jhcorcalemf = new float[kMaxJetCal]; 
    jhcorcaln90 = new float[kMaxJetCal]; 
    jhcorcaln90hits = new float[kMaxJetCal]; 

    jhcorL1L2L3calpt = new float[kMaxJetCal]; 
    jhcorL1L2L3calphi = new float[kMaxJetCal]; 
    jhcorL1L2L3caleta = new float[kMaxJetCal]; 
    jhcorL1L2L3cale = new float[kMaxJetCal]; 
    jhcorL1L2L3calemf = new float[kMaxJetCal]; 
    jhcorL1L2L3caln90 = new float[kMaxJetCal]; 
    jhcorL1L2L3caln90hits = new float[kMaxJetCal]; 
   
    jrcalpt = new float[kMaxJetCal];
    jrcalphi = new float[kMaxJetCal];
    jrcaleta = new float[kMaxJetCal];
    jrcale = new float[kMaxJetCal];
    jrcalemf = new float[kMaxJetCal]; 
    jrcaln90 = new float[kMaxJetCal]; 
    jrcaln90hits = new float[kMaxJetCal]; 

    jrcorcalpt = new float[kMaxJetCal];
    jrcorcalphi = new float[kMaxJetCal];
    jrcorcaleta = new float[kMaxJetCal];
    jrcorcale = new float[kMaxJetCal];
    jrcorcalemf = new float[kMaxJetCal]; 
    jrcorcaln90 = new float[kMaxJetCal]; 
    jrcorcaln90hits = new float[kMaxJetCal]; 

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
    towR45upper = new int[kMaxTower];
    towR45lower = new int[kMaxTower];
    towR45none = new int[kMaxTower];
    const int kMaxTau = 500;
    l2tauPt    = new float[kMaxTau];
    l2tauEta   = new float[kMaxTau];
    l2tauPhi   = new float[kMaxTau];
    l2tauemiso = new float[kMaxTau];
    l25tauPt = new float[kMaxTau];
    l3tautckiso = new int[kMaxTau];
    tauEta = new float[kMaxTau];
    tauPt = new float[kMaxTau];
    tauPhi = new float[kMaxTau];
    
    const int kMaxPFTau = 500;
    ohpfTauEta         =  new float[kMaxPFTau];
    ohpfTauProngs      =  new int[kMaxPFTau];
    ohpfTauPhi         =  new float[kMaxPFTau];
    ohpfTauPt          =  new float[kMaxPFTau];
    ohpfTauJetPt       =  new float[kMaxPFTau];
    ohpfTauLeadTrackPt =  new float[kMaxPFTau];
    ohpfTauLeadTrackVtxZ = new float[kMaxPFTau];
    ohpfTauLeadPionPt  =  new float[kMaxPFTau];
    ohpfTauTrkIso      =  new float[kMaxPFTau];
    ohpfTauGammaIso    =  new float[kMaxPFTau];

    ohpfTauTightConeProngs	=  new int[kMaxPFTau];
    ohpfTauTightConeEta         =  new float[kMaxPFTau];
    ohpfTauTightConePhi         =  new float[kMaxPFTau];
    ohpfTauTightConePt          =  new float[kMaxPFTau];
    ohpfTauTightConeJetPt       =  new float[kMaxPFTau];
    ohpfTauTightConeLeadTrackPt =  new float[kMaxPFTau];
    ohpfTauTightConeLeadPionPt  =  new float[kMaxPFTau];
    ohpfTauTightConeTrkIso      =  new float[kMaxPFTau];
    ohpfTauTightConeGammaIso    =  new float[kMaxPFTau];
    
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
    
    pfHT    = -100.;
    pfMHT   = -100.;    
    const int kMaxPFJet = 500;
    pfJetEta         = new float[kMaxPFJet];
    pfJetPhi         = new float[kMaxPFJet];
    pfJetPt         = new float[kMaxPFJet];
    pfJetE         = new float[kMaxPFJet];
    pfJetneutralHadronEnergyFraction = new float[kMaxPFJet];
    pfJetchargedHadronFraction = new float[kMaxPFJet];
    pfJetneutralMultiplicity = new float[kMaxPFJet];
    pfJetchargedMultiplicity = new float[kMaxPFJet];
    pfJetneutralEMFraction = new float[kMaxPFJet];
    pfJetchargedEMFraction = new float[kMaxPFJet];


    const int kMaxTauIso = 5000;

    // for offlineHPStau 
    signalTrToPFTauMatch = new int[kMaxTauIso];// index of reconstructed tau in tau collection
    recoPFTauSignalTrDz = new float[kMaxTauIso];
    recoPFTauSignalTrPt = new float[kMaxTauIso];

    isoTrToPFTauMatch = new int[kMaxTauIso]; // index of reconstructed tau in tau collection
    recoPFTauIsoTrDz = new float[kMaxTauIso];
    recoPFTauIsoTrPt = new float[kMaxTauIso];

    // HLT pf taus
    hltpftauSignalTrToPFTauMatch = new int[kMaxTauIso]; // index of HLTPF tau in tau collection
    HLTPFTauSignalTrDz = new float[kMaxTauIso];
    HLTPFTauSignalTrPt = new float[kMaxTauIso];

    hltpftauIsoTrToPFTauMatch = new int[kMaxTauIso]; // index of HLTPF tau in tau collection
    HLTPFTauIsoTrDz = new float[kMaxTauIso];
    HLTPFTauIsoTrPt = new float[kMaxTauIso];

    // offline pftau isolation and signal cands
    HltTree->Branch("NoRecoPFTausSignal",&noRecoPFTausSignal,"NoRecoPFTausSignal/I");
    HltTree->Branch("signalTrToPFTauMatch", signalTrToPFTauMatch,"signalTrToPFTauMatch[NoRecoPFTausSignal]/I");
    HltTree->Branch("recoPFTauSignalTrDz", recoPFTauSignalTrDz,"recoPFTauSignalTrDz[NoRecoPFTausSignal]/F");
    HltTree->Branch("recoPFTauSignalTrPt", recoPFTauSignalTrPt,"recoPFTauSignalTrPt[NoRecoPFTausSignal]/F");

    HltTree->Branch("NoRecoPFTausIso",&noRecoPFTausIso,"NoRecoPFTausIso/I");
    HltTree->Branch("isoTrToPFTauMatch", isoTrToPFTauMatch,"isoTrToPFTauMatch[NoRecoPFTausIso]/I");
    HltTree->Branch("recoPFTauIsoTrDz", recoPFTauIsoTrDz,"recoPFTauIsoTrDz[NoRecoPFTausIso]/F");
    HltTree->Branch("recoPFTauIsoTrPt", recoPFTauIsoTrPt,"recoPFTauIsoTrPt[NoRecoPFTausIso]/F");

    // HLT pftau isolation and signal cands
    HltTree->Branch("NoHLTPFTausSignal",&noHLTPFTausSignal,"NoHLTPFTausSignal/I");
    HltTree->Branch("hltpftauSignalTrToPFTauMatch",
                    hltpftauSignalTrToPFTauMatch,"hltpftauSignalTrToPFTauMatch[NoHLTPFTausSignal]/I");
    HltTree->Branch("HLTPFTauSignalTrDz", HLTPFTauSignalTrDz,"HLTPFTauSignalTrDz[NoHLTPFTausSignal]/F");
    HltTree->Branch("HLTPFTauSignalTrPt", HLTPFTauSignalTrPt,"HLTPFTauSignalTrPt[NoHLTPFTausSignal]/F");

    HltTree->Branch("NoHLTPFTausIso",&noHLTPFTausIso,"NoHLTPFTausIso/I");
    HltTree->Branch("hltpftauIsoTrToPFTauMatch",
                    hltpftauIsoTrToPFTauMatch,"hltpftauIsoTrToPFTauMatch[NoHLTPFTausIso]/I");
    HltTree->Branch("HLTPFTauIsoTrDz", HLTPFTauIsoTrDz,"HLTPFTauIsoTrDz[NoHLTPFTausIso]/F");
    HltTree->Branch("HLTPFTauIsoTrPt", HLTPFTauIsoTrPt,"HLTPFTauIsoTrPt[NoHLTPFTausIso]/F");
    
    // Jet- MEt-specific branches of the tree 

    HltTree->Branch("NrecoJetGen",&njetgen,"NrecoJetGen/I");
    HltTree->Branch("NrecoTowCal",&ntowcal,"NrecoTowCal/I");

    //ccla RECO JETs
    HltTree->Branch("NrecoJetCal",&nrjetcal,"NrecoJetCal/I");
    HltTree->Branch("recoJetCalPt",jrcalpt,"recoJetCalPt[NrecoJetCal]/F");
    HltTree->Branch("recoJetCalPhi",jrcalphi,"recoJetCalPhi[NrecoJetCal]/F");
    HltTree->Branch("recoJetCalEta",jrcaleta,"recoJetCalEta[NrecoJetCal]/F");
    HltTree->Branch("recoJetCalE",jrcale,"recoJetCalE[NrecoJetCal]/F");
    HltTree->Branch("recoJetCalEMF",jrcalemf,"recoJetCalEMF[NrecoJetCal]/F");
    HltTree->Branch("recoJetCalN90",jrcaln90,"recoJetCalN90[NrecoJetCal]/F");
    HltTree->Branch("recoJetCalN90hits",jrcaln90hits,"recoJetCalN90hits[NrecoJetCal]/F");

    HltTree->Branch("NrecoJetCorCal",&nrcorjetcal,"NrecoJetCorCal/I"); 
    HltTree->Branch("recoJetCorCalPt",jrcorcalpt,"recoJetCorCalPt[NrecoJetCorCal]/F"); 
    HltTree->Branch("recoJetCorCalPhi",jrcorcalphi,"recoJetCorCalPhi[NrecoJetCorCal]/F"); 
    HltTree->Branch("recoJetCorCalEta",jrcorcaleta,"recoJetCorCalEta[NrecoJetCorCal]/F"); 
    HltTree->Branch("recoJetCorCalE",jrcorcale,"recoJetCorCalE[NrecoJetCorCal]/F"); 
    HltTree->Branch("recoJetCorCalEMF",jrcorcalemf,"recoJetCorCalEMF[NrecoJetCorCal]/F");
    HltTree->Branch("recoJetCorCalN90",jrcorcaln90,"recoJetCorCalN90[NrecoJetCorCal]/F");
    HltTree->Branch("recoJetCorCalN90hits",jrcorcaln90hits,"recoJetCorCalN90hits[NrecoJetCorCal]/F");
 
    //ccla HLTJETS
    HltTree->Branch("NohJetCal",&nhjetcal,"NohJetCal/I");
    HltTree->Branch("ohJetCalPt",jhcalpt,"ohJetCalPt[NohJetCal]/F");
    HltTree->Branch("ohJetCalPhi",jhcalphi,"ohJetCalPhi[NohJetCal]/F");
    HltTree->Branch("ohJetCalEta",jhcaleta,"ohJetCalEta[NohJetCal]/F");
    HltTree->Branch("ohJetCalE",jhcale,"ohJetCalE[NohJetCal]/F");
    HltTree->Branch("ohJetCalEMF",jhcalemf,"ohJetCalEMF[NohJetCal]/F");
    HltTree->Branch("ohJetCalN90",jhcaln90,"ohJetCalN90[NohJetCal]/F");
    HltTree->Branch("ohJetCalN90hits",jhcaln90hits,"ohJetCalN90hits[NohJetCal]/F");

    HltTree->Branch("NohJetCorCal",&nhcorjetcal,"NohJetCorCal/I");
    HltTree->Branch("ohJetCorCalPt",jhcorcalpt,"ohJetCorCalPt[NohJetCorCal]/F");
    HltTree->Branch("ohJetCorCalPhi",jhcorcalphi,"ohJetCorCalPhi[NohJetCorCal]/F");
    HltTree->Branch("ohJetCorCalEta",jhcorcaleta,"ohJetCorCalEta[NohJetCorCal]/F");
    HltTree->Branch("ohJetCorCalE",jhcorcale,"ohJetCorCalE[NohJetCorCal]/F");
    HltTree->Branch("ohJetCorCalEMF",jhcorcalemf,"ohJetCorCalEMF[NohJetCorCal]/F");
    HltTree->Branch("ohJetCorCalN90",jhcorcaln90,"ohJetCorCalN90[NohJetCorCal]/F");
    HltTree->Branch("ohJetCorCalN90hits",jhcorcaln90hits,"ohJetCorCalN90hits[NohJetCorCal]/F");

    HltTree->Branch("NohJetCorL1L2L3Cal",&nhcorL1L2L3jetcal,"NohJetCorL1L2L3Cal/I");
    HltTree->Branch("ohJetCorL1L2L3CalPt",jhcorL1L2L3calpt,"ohJetCorL1L2L3CalPt[NohJetCorL1L2L3Cal]/F");
    HltTree->Branch("ohJetCorL1L2L3CalPhi",jhcorL1L2L3calphi,"ohJetCorL1L2L3CalPhi[NohJetCorL1L2L3Cal]/F");
    HltTree->Branch("ohJetCorL1L2L3CalEta",jhcorL1L2L3caleta,"ohJetCorL1L2L3CalEta[NohJetCorL1L2L3Cal]/F");
    HltTree->Branch("ohJetCorL1L2L3CalE",jhcorL1L2L3cale,"ohJetCorL1L2L3CalE[NohJetCorL1L2L3Cal]/F");
    HltTree->Branch("ohJetCorL1L2L3CalEMF",jhcorL1L2L3calemf,"ohJetCorL1L2L3CalEMF[NohJetCorL1L2L3Cal]/F");
    HltTree->Branch("ohJetCorL1L2L3CalN90",jhcorL1L2L3caln90,"ohJetCorL1L2L3CalN90[NohJetCorL1L2L3Cal]/F");
    HltTree->Branch("ohJetCorL1L2L3CalN90hits",jhcorL1L2L3caln90hits,"ohJetCorL1L2L3CalN90hits[NohJetCorL1L2L3Cal]/F");
    HltTree->Branch("rho",&jrho,"rho/D");
   
    //ccla GenJets
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
    HltTree->Branch("recoTowHCalNoiseR45Upper",towR45upper,"recoTowHCalNoiseR45Upper[NrecoTowCal]/I");
    HltTree->Branch("recoTowHCalNoiseR45Lower",towR45lower,"recoTowHCalNoiseR45Lower[NrecoTowCal]/I");
    HltTree->Branch("recoTowHCalNoiseR45None",towR45none,"recoTowHCalNoiseR45None[NrecoTowCal]/I");

    HltTree->Branch("recoMetCal",&mcalmet,"recoMetCal/F");
    HltTree->Branch("recoMetCalPhi",&mcalphi,"recoMetCalPhi/F");
    HltTree->Branch("recoMetCalSum",&mcalsum,"recoMetCalSum/F");
    HltTree->Branch("recoMetGen",&mgenmet,"recoMetGen/F");
    HltTree->Branch("recoMetGenPhi",&mgenphi,"recoMetGenPhi/F");
    HltTree->Branch("recoMetGenSum",&mgensum,"recoMetGenSum/F");
    HltTree->Branch("recoHTCal",&htcalet,"recoHTCal/F");
    HltTree->Branch("recoHTCalPhi",&htcalphi,"recoHTCalPhi/F");
    HltTree->Branch("recoHTCalSum",&htcalsum,"recoHTCalSum/F");
    HltTree->Branch("recoMetPF", &pfmet, "recoMetPF/F");
    HltTree->Branch("recoMetPFSum", &pfsumet, "recoMetPFSum/F");
    HltTree->Branch("recoMetPFPhi", &pfmetphi, "recoMetPFPhi/F");

    //for(int ieta=0;ieta<NETA;ieta++){std::cout << " ieta " << ieta << " eta min " << CaloTowerEtaBoundries[ieta] <<std::endl;}
    
    
    // Taus
    nohl2tau = 0;
    HltTree->Branch("NohTauL2",&nohl2tau,"NohTauL2/I");
    HltTree->Branch("ohTauL2Pt",l2tauPt,"ohTauL2Pt[NohTauL2]/F");
    HltTree->Branch("ohTauL2Eta",l2tauEta,"ohTauL2Eta[NohTauL2]/F");
    HltTree->Branch("ohTauL2Phi",l2tauPhi,"ohTauL2Phi[NohTauL2]/F");

    nohtau = 0;
    HltTree->Branch("NohTau",&nohtau,"NohTau/I");
    HltTree->Branch("ohTauEta",tauEta,"ohTauEta[NohTau]/F");
    HltTree->Branch("ohTauPhi",tauPhi,"ohTauPhi[NohTau]/F");
    HltTree->Branch("ohTauPt",tauPt,"ohTauPt[NohTau]/F");
    HltTree->Branch("ohTauEiso",l2tauemiso,"ohTauEiso[NohTau]/F");
    HltTree->Branch("ohTauL25Tpt",l25tauPt,"ohTauL25Tpt[NohTau]/F");
    HltTree->Branch("ohTauL3Tiso",l3tautckiso,"ohTauL3Tiso[NohTau]/I");

    //ohpfTaus
    nohPFTau = 0;
    HltTree->Branch("NohpfTau",&nohPFTau,"NohpfTau/I");
    HltTree->Branch("ohpfTauPt",ohpfTauPt,"ohpfTauPt[NohpfTau]/F");
    HltTree->Branch("ohpfTauProngs",ohpfTauProngs,"ohpfTauProngs[NohpfTau]/I");
    HltTree->Branch("ohpfTauEta",ohpfTauEta,"ohpfTauEta[NohpfTau]/F");
    HltTree->Branch("ohpfTauPhi",ohpfTauPhi,"ohpfTauPhi[NohpfTau]/F");
    HltTree->Branch("ohpfTauLeadTrackPt",ohpfTauLeadTrackPt,"ohpfTauLeadTrackPt[NohpfTau]/F");
    HltTree->Branch("ohpfTauLeadTrackVtxZ",ohpfTauLeadTrackVtxZ,"ohpfTauLeadTrackVtxZ[NohpfTau]/F");
    HltTree->Branch("ohpfTauLeadPionPt",ohpfTauLeadPionPt,"ohpfTauLeadPionPt[NohpfTau]/F");
    HltTree->Branch("ohpfTauTrkIso",ohpfTauTrkIso,"ohpfTauTrkIso[NohpfTau]/F");
    HltTree->Branch("ohpfTauGammaIso",ohpfTauGammaIso,"ohpfTauGammaIso[NohpfTau]/F");
    HltTree->Branch("ohpfTauJetPt",ohpfTauJetPt,"ohpfTauJetPt[NohpfTau]/F");    

    //ohpfTaus tight cone
    nohPFTauTightCone = 0;
    HltTree->Branch("NohpfTauTightCone",&nohPFTauTightCone,"NohpfTauTightCone/I");
    HltTree->Branch("ohpfTauTightConePt",ohpfTauTightConePt,"ohpfTauTightConePt[NohpfTauTightCone]/F");
    HltTree->Branch("ohpfTauTightConeProngs",ohpfTauTightConeProngs,"ohpfTauProngs[NohpfTauTightCone]/I");
    HltTree->Branch("ohpfTauTightConeEta",ohpfTauTightConeEta,"ohpfTauEta[NohpfTauTightCone]/F");
    HltTree->Branch("ohpfTauTightConePhi",ohpfTauTightConePhi,"ohpfTauPhi[NohpfTauTightCone]/F");
    HltTree->Branch("ohpfTauTightConeLeadTrackPt",ohpfTauTightConeLeadTrackPt,"ohpfTauTightConeLeadTrackPt[NohpfTauTightCone]/F");
    HltTree->Branch("ohpfTauTightConeLeadPionPt",ohpfTauTightConeLeadPionPt,"ohpfTauTightConeLeadPionPt[NohpfTauTightCone]/F");
    HltTree->Branch("ohpfTauTightConeTrkIso",ohpfTauTightConeTrkIso,"ohpfTauTightConeTrkIso[NohpfTauTightCone]/F");
    HltTree->Branch("ohpfTauTightConeGammaIso",ohpfTauTightConeGammaIso,"ohpfTauTightConeGammaIso[NohpfTauTightCone]/F");
    HltTree->Branch("ohpfTauTightConeJetPt",ohpfTauTightConeJetPt,"ohpfTauTightConeJetPt[NohpfTauTightCone]/F");
   
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
    nohPFJet = 0;
    HltTree->Branch("pfHT",&pfHT,"pfHT/F");
    HltTree->Branch("pfMHT",&pfMHT,"pfMHT/F");
    HltTree->Branch("NohPFJet",&nohPFJet,"NohPFJet/I");
    HltTree->Branch("pfJetPt",pfJetPt,"pfJetPt[NohPFJet]/F");
    HltTree->Branch("pfJetE",pfJetE,"pfJetE[NohPFJet]/F");
    HltTree->Branch("pfJetEta",pfJetEta,"pfJetEta[NohPFJet]/F");
    HltTree->Branch("pfJetPhi",pfJetPhi,"pfJetPhi[NohPFJet]/F");
    HltTree->Branch("pfJetneutralHadronEnergyFraction",pfJetneutralHadronEnergyFraction,"pfJetneutralHadronEnergyFraction[NohPFJet]/F");
    HltTree->Branch("pfJetchargedHadronFraction",pfJetchargedHadronFraction,"pfJetchargedHadronFraction[NohPFJet]/F");
    HltTree->Branch("pfJetneutralMultiplicity",pfJetneutralMultiplicity,"pfJetneutralMultiplicity[NohPFJet]/F");
    HltTree->Branch("pfJetchargedMultiplicity",pfJetchargedMultiplicity,"pfJetchargedMultiplicity[NohPFJet]/F");
    HltTree->Branch("pfJetneutralEMFraction",pfJetneutralEMFraction,"pfJetneutralEMFraction[NohPFJet]/F");
    HltTree->Branch("pfJetchargedEMFraction",pfJetchargedEMFraction,"pfJetchargedEMFraction[NohPFJet]/F");

    //RECO PFJets
    HltTree->Branch("nrpj",&nrpj,"nrpj/I");
    HltTree->Branch("recopfJetpt",                    jpfrecopt,                     "recopfJetpt[nrpj]/F");
    HltTree->Branch("recopfJete",                     jpfrecoe,                      "recopfJete[nrpj]/F");
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
void HLTJets::analyze(edm::Event const& iEvent,
		      const edm::Handle<reco::CaloJetCollection>      & ohcalojets,
                      const edm::Handle<reco::CaloJetCollection>      & ohcalocorjets,
		      const edm::Handle<reco::CaloJetCollection>      & ohcalocorL1L2L3jets,
		      const edm::Handle< double >                     & rho,
		      const edm::Handle<reco::CaloJetCollection>      & rcalojets,
		      const edm::Handle<reco::CaloJetCollection>      & rcalocorjets,
                      const edm::Handle<reco::GenJetCollection>       & genjets,
                      const edm::Handle<reco::CaloMETCollection>      & recmets,
                      const edm::Handle<reco::GenMETCollection>       & genmets,
                      const edm::Handle<reco::METCollection>          & ht,
                      const edm::Handle<reco::CaloJetCollection>      & l2taujets,
                      const edm::Handle<reco::HLTTauCollection>       & taujets,
                      const edm::Handle<reco::PFTauCollection>        & pfTaus,
                      const edm::Handle<reco::PFTauCollection>        & pfTausTightCone,
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
		      const edm::Handle<CaloTowerCollection>          & caloTowersCleanerUpperR45,
		      const edm::Handle<CaloTowerCollection>          & caloTowersCleanerLowerR45,
		      const edm::Handle<CaloTowerCollection>          & caloTowersCleanerNoR45,
              const CaloTowerTopology * cttopo,
		      const edm::Handle<reco::PFMETCollection>        & pfmets, 
                      double thresholdForSavingTowers, 
                      double		    minPtCH,
                      double		   minPtGamma,
                      TTree * HltTree) {
    
    if (_Debug) std::cout << " Beginning HLTJets " << std::endl;
    
    //initialize branch variables
    nhjetcal=0; nhcorjetcal=0;nhcorL1L2L3jetcal=0; njetgen=0;ntowcal=0;
    jrho = 0;
    mcalmet=0.; mcalphi=0.;
    mgenmet=0.; mgenphi=0.;
    htcalet=0.,htcalphi=0.,htcalsum=0.;

    noRecoPFTausSignal = 0; noRecoPFTausIso =0;
    noHLTPFTausSignal = 0; noHLTPFTausIso = 0;



    if (rcalojets.isValid()) {
      reco::CaloJetCollection mycalojets;
      mycalojets=*rcalojets;
      std::sort(mycalojets.begin(),mycalojets.end(),PtGreater());
      typedef reco::CaloJetCollection::const_iterator cjiter;
      int jrcal=0;
      for ( cjiter i=mycalojets.begin(); i!=mycalojets.end(); i++) {
    
    	if (i->pt()>_CalJetMin && i->energy()>0.){
    	  jrcalpt[jrcal] = i->pt();
    	  jrcalphi[jrcal] = i->phi();
    	  jrcaleta[jrcal] = i->eta();
    	  jrcale[jrcal] = i->energy();
    	  jrcalemf[jrcal] = i->emEnergyFraction();
    	  jrcaln90[jrcal] = i->n90();
	  jetID->calculate( iEvent, *i );
	  jrcaln90hits[jrcal] = jetID->n90Hits();
    	  jrcal++;
    	}
      }
      nrjetcal = jrcal;
    }
    else {nrjetcal = 0;}
    
    if (rcalocorjets.isValid()) {
      reco::CaloJetCollection mycalojets;
      mycalojets=*rcalocorjets;
      std::sort(mycalojets.begin(),mycalojets.end(),PtGreater());
      typedef reco::CaloJetCollection::const_iterator cjiter;
      int jrcal=0;
      for ( cjiter i=mycalojets.begin(); i!=mycalojets.end(); i++) {
    
    	if (i->pt()>_CalJetMin && i->energy()>0.){
    	  jrcorcalpt[jrcal] = i->pt();
    	  jrcorcalphi[jrcal] = i->phi();
    	  jrcorcaleta[jrcal] = i->eta();
    	  jrcorcale[jrcal] = i->energy();
    	  jrcorcalemf[jrcal] = i->emEnergyFraction();
    	  jrcorcaln90[jrcal] = i->n90();
	  jetID->calculate( iEvent, *i );
    	  jrcorcaln90hits[jrcal] = jetID->n90Hits();
    	  jrcal++;
    	}
      }
      nrcorjetcal = jrcal;
    }
    else {nrcorjetcal = 0;}
    
    if (ohcalojets.isValid()) {
        reco::CaloJetCollection mycalojets;
        mycalojets=*ohcalojets;
        std::sort(mycalojets.begin(),mycalojets.end(),PtGreater());
        typedef reco::CaloJetCollection::const_iterator cjiter;
        int jhcal=0;
        for ( cjiter i=mycalojets.begin(); i!=mycalojets.end(); i++) {
            
            if (i->pt()>_CalJetMin && i->energy()>0.){
                jhcalpt[jhcal] = i->pt();
                jhcalphi[jhcal] = i->phi();
                jhcaleta[jhcal] = i->eta();
                jhcale[jhcal] = i->energy();
                jhcalemf[jhcal] = i->emEnergyFraction();
                jhcaln90[jhcal] = i->n90();
		jetID->calculate( iEvent, *i );
                jhcaln90hits[jhcal] = jetID->n90Hits();
                jhcal++;
            }
            
        }
        nhjetcal = jhcal;
    }
    else {nhjetcal = 0;}
    
    if (ohcalocorjets.isValid()) {
        reco::CaloJetCollection mycalocorjets;
        mycalocorjets=*ohcalocorjets;
        std::sort(mycalocorjets.begin(),mycalocorjets.end(),PtGreater());
        typedef reco::CaloJetCollection::const_iterator ccorjiter;
        int jhcorcal=0;
        for ( ccorjiter i=mycalocorjets.begin(); i!=mycalocorjets.end(); i++) {
            
            if (i->pt()>_CalJetMin && i->energy()>0.){
                jhcorcalpt[jhcorcal] = i->pt();
                jhcorcalphi[jhcorcal] = i->phi();
                jhcorcaleta[jhcorcal] = i->eta();
                jhcorcale[jhcorcal] = i->energy();
                jhcorcalemf[jhcorcal] = i->emEnergyFraction();
                jhcorcaln90[jhcorcal] = i->n90();
		jetID->calculate( iEvent, *i );
                jhcorcaln90hits[jhcorcal] = jetID->n90Hits();
                jhcorcal++;
            }
            
        }
        nhcorjetcal = jhcorcal;
    }
    else {nhcorjetcal = 0;}
    
 if (ohcalocorL1L2L3jets.isValid()) {
        reco::CaloJetCollection mycalocorL1L2L3jets;
        mycalocorL1L2L3jets=*ohcalocorL1L2L3jets;
        std::sort(mycalocorL1L2L3jets.begin(),mycalocorL1L2L3jets.end(),PtGreater());
        typedef reco::CaloJetCollection::const_iterator ccorL1L2L3jiter;
        int jhcorL1L2L3cal=0;
        for ( ccorL1L2L3jiter i=mycalocorL1L2L3jets.begin(); i!=mycalocorL1L2L3jets.end(); i++) {
            
            if (i->pt()>_CalJetMin && i->energy()>0.){
                jhcorL1L2L3calpt[jhcorL1L2L3cal] = i->pt();
                jhcorL1L2L3calphi[jhcorL1L2L3cal] = i->phi();
                jhcorL1L2L3caleta[jhcorL1L2L3cal] = i->eta();
                jhcorL1L2L3cale[jhcorL1L2L3cal] = i->energy();
                jhcorL1L2L3calemf[jhcorL1L2L3cal] = i->emEnergyFraction();
                jhcorL1L2L3caln90[jhcorL1L2L3cal] = i->n90();
		jetID->calculate( iEvent, *i );
                jhcorL1L2L3caln90hits[jhcorL1L2L3cal] = jetID->n90Hits();
                jhcorL1L2L3cal++;
            }
            
        }
        nhcorL1L2L3jetcal = jhcorL1L2L3cal;
   }
 else {nhcorL1L2L3jetcal = 0;}

 if (rho.isValid()){
   jrho = *rho;
 }
 else {

   if (_Debug) std::cout << "rho not found" << std::endl;
 }

 std::set<unsigned int> towersUpper;
 std::set<unsigned int> towersLower;
 std::set<unsigned int> towersNone;

 bool towersUpperValid=false;
 bool towersLowerValid=false;
 bool towersNoneValid=false;
 if( caloTowersCleanerUpperR45.isValid() ){
   towersUpperValid = true;
   for( CaloTowerCollection::const_iterator tow = caloTowersCleanerUpperR45->begin(); tow != caloTowersCleanerUpperR45->end(); tow++){
     towersUpper.insert(cttopo->denseIndex(tow->id()));
   }
 }
 if( caloTowersCleanerLowerR45.isValid() ){
   towersLowerValid = true;
   for( CaloTowerCollection::const_iterator tow = caloTowersCleanerLowerR45->begin(); tow != caloTowersCleanerLowerR45->end(); tow++){
     towersLower.insert(cttopo->denseIndex(tow->id()));
   }
 }
 if( caloTowersCleanerNoR45.isValid() ){
   towersNoneValid = true;
   for( CaloTowerCollection::const_iterator tow = caloTowersCleanerNoR45->begin(); tow != caloTowersCleanerNoR45->end(); tow++){
     towersNone.insert(cttopo->denseIndex(tow->id()));
   }
 }
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
	 // noise filters: true = no noise, false = noise
	 if(towersUpperValid) {if(towersUpper.find(cttopo->denseIndex(tower->id())) == towersUpper.end()) towR45upper[jtow]=true; else towR45upper[jtow]=false;}
	 if(towersLowerValid) {if(towersLower.find(cttopo->denseIndex(tower->id())) == towersLower.end()) towR45lower[jtow]=true; else towR45lower[jtow]=false;}
	 if(towersNoneValid) {if(towersNone.find(cttopo->denseIndex(tower->id())) == towersNone.end()) towR45none[jtow]=true; else towR45none[jtow]=false;}
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

    if (pfmets.isValid()) {
      typedef reco::PFMETCollection::const_iterator pfmetiter;
      for( pfmetiter i=pfmets->begin(); i!=pfmets->end(); i++) {
	pfmet = i->pt();
	pfsumet = i->sumEt();
	pfmetphi = i->phi();
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
    if (l2taujets.isValid()) {
        nohl2tau = l2taujets->size();
        reco::CaloJetCollection l2taus = *l2taujets;
        std::sort(l2taus.begin(),l2taus.end(),GetPFPtGreater());
        int itau=0;
        for(reco::CaloJetCollection::const_iterator i = l2taus.begin(); 
                                                   i!= l2taus.end(); ++i){
            l2tauPt[itau]  = i->pt();
            l2tauEta[itau] = i->eta();
            l2tauPhi[itau] = i->phi();
            itau++;
        }
    }else{
      nohl2tau = 0;
    }
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

    
    ////////////////Particle Flow Taus - HLT ////////////////////////////////////
    if(pfTaus.isValid()) {
        //float minTrkPt = minPtCH;
        //float minGammaPt = minPtGamma;
        nohPFTau  = pfTaus->size();
        reco::PFTauCollection taus = *pfTaus;
        std::sort(taus.begin(),taus.end(),GetPFPtGreater());
        typedef reco::PFTauCollection::const_iterator pftauit;
        int ipftau=0;
        for(pftauit i=taus.begin(); i!=taus.end(); i++){
            //Ask for Eta,Phi and Et of the tau:
	    ohpfTauProngs[ipftau] = i->signalPFChargedHadrCands().size();
            ohpfTauEta[ipftau] = i->eta();
            ohpfTauPhi[ipftau] = i->phi();
            ohpfTauPt[ipftau] = i->pt();
            ohpfTauJetPt[ipftau] = i->pfTauTagInfoRef()->pfjetRef()->pt();

            
  /*
            if( (i->leadPFCand()).isNonnull())
                pfTauLeadPionPt[ipftau] = i->leadPFCand()->pt();            
*/
            if( (i->leadPFNeutralCand()).isNonnull())
                ohpfTauLeadPionPt[ipftau] = i->leadPFNeutralCand()->pt();        
            else 
                ohpfTauLeadPionPt[ipftau] = -999.0;

            if((i->leadPFChargedHadrCand()).isNonnull()){
                ohpfTauLeadTrackPt[ipftau] = i->leadPFChargedHadrCand()->pt();
                ohpfTauLeadTrackVtxZ[ipftau] = i->leadPFChargedHadrCand()->vertex().z(); 
            }else{
                ohpfTauLeadTrackPt[ipftau] = -999.0;
                ohpfTauLeadTrackVtxZ[ipftau] = -999.0;  
            }

            float maxPtTrkIso = 0;
            for (unsigned int iTrk = 0; iTrk < i->isolationPFChargedHadrCands().size(); iTrk++)
            {
                if(i->isolationPFChargedHadrCands()[iTrk]->pt() > maxPtTrkIso) maxPtTrkIso = i->isolationPFChargedHadrCands()[iTrk]->pt();

                if (i->isolationPFChargedHadrCands()[iTrk]->trackRef().isNonnull()){
                  hltpftauIsoTrToPFTauMatch[noHLTPFTausIso]=ipftau;
                  HLTPFTauIsoTrDz[noHLTPFTausIso]=i->isolationPFChargedHadrCands()[iTrk]->trackRef()->dz(); // dz wrt (0,0,0), to compare offline with HLT 
                  HLTPFTauIsoTrPt[noHLTPFTausIso]=i->isolationPFChargedHadrCands()[iTrk]->pt();
                  /*
                  std::cout << "Adding isocand for hltpftau " << ipftau
                      << " pt " << HLTPFTauIsoTrPt[noHLTPFTausIso]
                      << " dz " << HLTPFTauIsoTrDz[noHLTPFTausIso]
                      << std::endl; // */
                  ++noHLTPFTausIso;
                }

            }
                
            ohpfTauTrkIso[ipftau] = maxPtTrkIso;
            float maxPtGammaIso = 0;
            for (unsigned int iGamma = 0; iGamma < i->isolationPFGammaCands().size(); iGamma++)
            {
                if(i->isolationPFGammaCands()[iGamma]->pt() > maxPtGammaIso) maxPtGammaIso = i->isolationPFGammaCands()[iGamma]->pt();
            }                        



            for (unsigned int iTrk = 0; iTrk < i->signalPFChargedHadrCands().size(); iTrk++)
            {
              if (i->signalPFChargedHadrCands ()[iTrk]->trackRef().isNonnull()){
                hltpftauSignalTrToPFTauMatch[noHLTPFTausSignal]=ipftau;
                HLTPFTauSignalTrDz[noHLTPFTausSignal]=i->signalPFChargedHadrCands()[iTrk]->trackRef()->dz(); // dz wrt (0,0,0), to compare offline with HLT
                HLTPFTauSignalTrPt[noHLTPFTausSignal]=i->signalPFChargedHadrCands()[iTrk]->pt();
                /*
                  std::cout << "Adding sigcand for hltpftau " << ipftau
                      << " pt " << HLTPFTauSignalTrPt[noHLTPFTausSignal]
                      << " dz " << HLTPFTauSignalTrDz[noHLTPFTausSignal]
                      << std::endl; // */
                ++noHLTPFTausSignal;
              }
            }





            ohpfTauGammaIso[ipftau] = maxPtGammaIso;
            ipftau++;
        } 
      
    }

    if(pfTausTightCone.isValid()) {
        //float minTrkPt = minPtCH;
        //float minGammaPt = minPtGamma;
        nohPFTauTightCone = pfTaus->size();
        reco::PFTauCollection taus = *pfTausTightCone;
        std::sort(taus.begin(),taus.end(),GetPFPtGreater());
        typedef reco::PFTauCollection::const_iterator pftauit;
        int ipftau=0;
        for(pftauit i=taus.begin(); i!=taus.end(); i++){
            //Ask for Eta,Phi and Et of the tau:
            ohpfTauTightConeProngs[ipftau] = i->signalPFChargedHadrCands().size();
            ohpfTauTightConeEta[ipftau] = i->eta();
            ohpfTauTightConePhi[ipftau] = i->phi();
            ohpfTauTightConePt[ipftau] = i->pt();
            ohpfTauTightConeJetPt[ipftau] = i->pfTauTagInfoRef()->pfjetRef()->pt();

    
            if( (i->leadPFNeutralCand()).isNonnull())
                ohpfTauTightConeLeadPionPt[ipftau] = i->leadPFNeutralCand()->pt();
            else 
              ohpfTauTightConeLeadPionPt[ipftau] = -999.0;


            if((i->leadPFChargedHadrCand()).isNonnull())
                ohpfTauTightConeLeadTrackPt[ipftau] = i->leadPFChargedHadrCand()->pt();
            else
              ohpfTauTightConeLeadTrackPt[ipftau] = -999.0;

            float maxPtTrkIso = 0;
            for (unsigned int iTrk = 0; iTrk < i->isolationPFChargedHadrCands().size(); iTrk++)
            {
                if(i->isolationPFChargedHadrCands()[iTrk]->pt() > maxPtTrkIso) maxPtTrkIso = i->isolationPFChargedHadrCands()[iTrk]->pt();
            }

            ohpfTauTightConeTrkIso[ipftau] = maxPtTrkIso;
            float maxPtGammaIso = 0;
            for (unsigned int iGamma = 0; iGamma < i->isolationPFGammaCands().size(); iGamma++)
            {
                if(i->isolationPFGammaCands()[iGamma]->pt() > maxPtGammaIso) maxPtGammaIso = i->isolationPFGammaCands()[iGamma]->pt();
            }
            ohpfTauTightConeGammaIso[ipftau] = maxPtGammaIso;
            ipftau++;
        }

    }
    
    ////////////////Reco Particle Flow Taus ////////////////////////////////////
      
    if(recoPfTaus.isValid()) {
        float minTrkPt = minPtCH;
        float minGammaPt = minPtGamma;
        nRecoPFTau  = recoPfTaus->size();
        reco::PFTauCollection taus = *recoPfTaus;

        // disable sorting for proper access to discriminators
        //std::sort(taus.begin(),taus.end(),GetPFPtGreater());
        typedef reco::PFTauCollection::const_iterator pftauit;
        int ipftau=0;
        
        for(pftauit i=taus.begin(); i!=taus.end(); i++){
            //Ask for Eta,Phi and Et of the tau:
            recopfTauEta[ipftau] = i->eta();
            recopfTauPhi[ipftau] = i->phi();
            recopfTauPt[ipftau]  = i->pt();

            if( (i->leadPFNeutralCand()).isNonnull())
                recopfTauLeadPionPt[ipftau] = i->leadPFNeutralCand()->pt();
            else
              recopfTauLeadPionPt[ipftau] = -999.0;  


            if((i->leadPFChargedHadrCand()).isNonnull())
                recopfTauLeadTrackPt[ipftau] = i->leadPFChargedHadrCand()->pt();
            else
              recopfTauLeadTrackPt[ipftau]  = -999.0;

            int myTrks=0;
            for (unsigned int iTrk = 0; iTrk < i->isolationPFChargedHadrCands().size(); iTrk++)
            {
                if(i->isolationPFChargedHadrCands()[iTrk]->pt() > minTrkPt) myTrks++;
                if (i->isolationPFChargedHadrCands()[iTrk]->trackRef().isNonnull()){
                  isoTrToPFTauMatch[noRecoPFTausIso]=ipftau;
                  recoPFTauIsoTrDz[noRecoPFTausIso]=i->isolationPFChargedHadrCands()[iTrk]->trackRef()->dz(); // dz wrt (0,0,0), to compare offline with HLT
                  recoPFTauIsoTrPt[noRecoPFTausIso]=i->isolationPFChargedHadrCands()[iTrk]->pt();
                  /*
                  std::cout << "Adding isocand for tau " << ipftau
                            << " pt " << recoPFTauIsoTrPt[noRecoPFTausIso]
                            << " dz " << recoPFTauIsoTrDz[noRecoPFTausIso]
                            << std::endl;// */
                  ++noRecoPFTausIso;
                }

            }
               
            recopfTauTrkIso[ipftau] = myTrks;
            int myGammas=0;
            for (unsigned int iGamma = 0; iGamma < i->isolationPFGammaCands().size(); iGamma++)
            {
                if(i->isolationPFGammaCands()[iGamma]->pt() > minGammaPt) myGammas++;
            }                        
            recopfTauGammaIso[ipftau] = myGammas;
	    

            for (unsigned int iTrk = 0; iTrk < i->signalPFChargedHadrCands().size(); iTrk++)
            {
                if (i->signalPFChargedHadrCands ()[iTrk]->trackRef().isNonnull()){
                  signalTrToPFTauMatch[noRecoPFTausSignal]=ipftau;
                  recoPFTauSignalTrDz[noRecoPFTausSignal]=i->signalPFChargedHadrCands()[iTrk]->trackRef()->dz(); // dz wrt (0,0,0), to compare offline with HLT
                  recoPFTauSignalTrPt[noRecoPFTausSignal]=i->signalPFChargedHadrCands()[iTrk]->pt();
                  /*
                  std::cout << "Adding sigcand for tau " << ipftau
                            << " pt " << recoPFTauSignalTrPt[noRecoPFTausSignal]
                            << " dz " << recoPFTauSignalTrDz[noRecoPFTausSignal]
                            << std::endl;// */
                  ++noRecoPFTausSignal;
                }
            }

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
        float pfMHTx = 0.;
        float pfMHTy = 0.;
	pfHT         = 0.;

        for(pfJetit i=Jets.begin(); i!=Jets.end(); i++){
            //Ask for Eta,Phi and Et of the Jet:
            pfJetEta[ipfJet] = i->eta();
            pfJetPhi[ipfJet] = i->phi();
            pfJetPt[ipfJet] = i->pt();           
	    pfJetE[ipfJet] = i->energy();
            pfJetneutralHadronEnergyFraction[ipfJet]=i->neutralHadronEnergyFraction();
            pfJetchargedHadronFraction[ipfJet] = i->chargedHadronEnergyFraction ();
            pfJetneutralMultiplicity[ipfJet] = i->neutralMultiplicity ();
            pfJetchargedMultiplicity[ipfJet] = i->chargedMultiplicity ();
            pfJetneutralEMFraction[ipfJet] = i->neutralEmEnergyFraction ();
            pfJetchargedEMFraction[ipfJet] = i->chargedEmEnergyFraction ();
	    //std::cout << "jet pT = " << i->pt() << " ; neutralHadronEnergyFraction = " << i->neutralHadronEnergyFraction() << std::endl;

	    if (i->pt() > 40. && abs(i->eta())<3.0)
	      pfHT  += i -> pt();
	    if (i->pt() > 30.){
	      pfMHTx = pfMHTx + i->px();
	      pfMHTy = pfMHTy + i->py();
	    }
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
		    jpfrecoe[ipfJet] = i->energy();           
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
