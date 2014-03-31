#ifndef HLTJETS_H
#define HLTJETS_H

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TNamed.h"
#include <vector>
#include <map>
#include "TROOT.h"
#include "TChain.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "DataFormats/METReco/interface/PFMETCollection.h"  
#include "DataFormats/METReco/interface/PFMET.h"   

#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/METCollection.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"

#include "DataFormats/TauReco/interface/HLTTau.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

#include "HLTrigger/HLTanalyzers/interface/JetUtil.h"
#include "HLTrigger/HLTanalyzers/interface/CaloTowerBoundries.h"

#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include "RecoJets/JetProducers/interface/JetIDHelper.h"

typedef std::vector<std::string> MyStrings;

/** \class HLTJets
 *  
 * $Date: November 2006
 * $Revision: 
 * \author L. Apanasevich - UIC, P. Bargassa - Rice U.
 */

class GetPtGreater {
public:
    template <typename T> bool operator () (const T& i, const T& j) {
        return (i.getPt() > j.getPt());
    }
};

class GetPFPtGreater {
public:
    template <typename T> bool operator () (const T& i, const T& j) {
        return (i.pt() > j.pt());
    }
};

class HLTJets {
public:
    HLTJets(); 
    
    void setup(const edm::ParameterSet& pSet, TTree* tree);
    
    /** Analyze the Data */
    void analyze(edm::Event const& iEvent,
		 const edm::Handle<reco::CaloJetCollection>      & ohjets,
                 const edm::Handle<reco::CaloJetCollection>      & ohcorjets,
		 const edm::Handle<reco::CaloJetCollection>      & ohcorL1L2L3jets,
		 const edm::Handle<double>                       & rho,
		 const edm::Handle<reco::CaloJetCollection>      & recojets,
		 const edm::Handle<reco::CaloJetCollection>      & recocorjets,
                 const edm::Handle<reco::GenJetCollection>       & gjets,
                 const edm::Handle<reco::CaloMETCollection>      & rmets,
                 const edm::Handle<reco::GenMETCollection>       & gmets,
                 const edm::Handle<reco::METCollection>          & ht,                
                 const edm::Handle<reco::CaloJetCollection>      & myHLTL2Tau,
                 const edm::Handle<reco::HLTTauCollection>       & myHLTTau,
                 const edm::Handle<reco::PFTauCollection>        & myHLTPFTau,
                 const edm::Handle<reco::PFTauCollection>        & myHLTPFTauTightCone,
                 const edm::Handle<reco::PFJetCollection>        & myHLTPFJets,                
		 const edm::Handle<reco::PFTauCollection>	 & myRecoPFTau,
		 const edm::Handle<reco::PFTauDiscriminator>     & theRecoPFTauDiscrByTanCOnePercent,
		 const edm::Handle<reco::PFTauDiscriminator>	 & theRecoPFTauDiscrByTanCHalfPercent,
		 const edm::Handle<reco::PFTauDiscriminator>	 & theRecoPFTauDiscrByTanCQuarterPercent,
		 const edm::Handle<reco::PFTauDiscriminator>	 & theRecoPFTauDiscrByTanCTenthPercent,
		 const edm::Handle<reco::PFTauDiscriminator>	 & theRecoPFTauDiscrByIsolation,
		 const edm::Handle<reco::PFTauDiscriminator>     & theRecoPFTauDiscrAgainstElec,
		 const edm::Handle<reco::PFTauDiscriminator>     & theRecoPFTauDiscrAgainstMuon,
		 const edm::Handle<reco::PFJetCollection>        & recoPFJets,                
		 const edm::Handle<CaloTowerCollection>          & caloTowers,	      
                 const edm::Handle<CaloTowerCollection>          & caloTowersCleanerUpperR45,
                 const edm::Handle<CaloTowerCollection>          & caloTowersCleanerLowerR45,
                 const edm::Handle<CaloTowerCollection>          & caloTowersCleanerNoR45,
         const CaloTowerTopology * cttopo,
		 const edm::Handle<reco::PFMETCollection>        & pfmets,  
                 double thresholdForSavingTowers,
                 double                minPtCH,
                 double                minPtGamma,
                 TTree * tree);
    
private:
    
    // Tree variables
    float *jhcalpt, *jhcalphi, *jhcaleta, *jhcale, *jhcalemf, *jhcaln90, *jhcaln90hits;
    float *jhcorcalpt, *jhcorcalphi, *jhcorcaleta, *jhcorcale, *jhcorcalemf, *jhcorcaln90, *jhcorcaln90hits;
    float *jhcorL1L2L3calpt, *jhcorL1L2L3calphi, *jhcorL1L2L3caleta, *jhcorL1L2L3cale, *jhcorL1L2L3calemf, *jhcorL1L2L3caln90, *jhcorL1L2L3caln90hits;
    double jrho;

    float *jrcalpt, *jrcalphi, *jrcaleta, *jrcale, *jrcalemf, *jrcaln90, *jrcaln90hits;
    float *jrcorcalpt, *jrcorcalphi, *jrcorcaleta, *jrcorcale, *jrcorcalemf, *jrcorcaln90, *jrcorcaln90hits;

    float *jgenpt, *jgenphi, *jgeneta, *jgene;
    float *towet, *toweta, *towphi, *towen, *towem, *towhd, *towoe;
    int *towR45upper, *towR45lower, *towR45none;
    float mcalmet,mcalphi,mcalsum;
    float htcalet,htcalphi,htcalsum;
    float mgenmet,mgenphi,mgensum;

    float pfmet,pfsumet,pfmetphi; 

    int njetgen,ntowcal;
    int nhjetcal,nhcorjetcal,nhcorL1L2L3jetcal;
    int nrjetcal,nrcorjetcal;
    
    // Taus
    float *l2tauPt, *l2tauEta, *l2tauPhi, *l2tauemiso, *l25tauPt;
    int *l3tautckiso;
    int nohl2tau, nohtau;
    float *tauEta, *tauPt, *tauPhi; 
    //PFTau
    int nohPFTau;
    int *ohpfTauProngs;
    float *ohpfTauEta,*ohpfTauPhi,*ohpfTauPt,*ohpfTauJetPt,*ohpfTauLeadTrackPt,*ohpfTauLeadTrackVtxZ,*ohpfTauLeadPionPt;
    float *ohpfTauTrkIso, *ohpfTauGammaIso;
    //PFTau with tight cone
    int nohPFTauTightCone;
    int *ohpfTauTightConeProngs;
    float *ohpfTauTightConeEta,*ohpfTauTightConePhi,*ohpfTauTightConePt,*ohpfTauTightConeJetPt,*ohpfTauTightConeLeadTrackPt,*ohpfTauTightConeLeadPionPt;
    float *ohpfTauTightConeTrkIso, *ohpfTauTightConeGammaIso;
    //PFJets
    float pfHT;
    float pfMHT;    
    int nohPFJet;
    float *pfJetEta, *pfJetPhi, *pfJetPt, *pfJetE, *pfJetneutralHadronEnergyFraction, *pfJetchargedHadronFraction, *pfJetneutralMultiplicity, *pfJetchargedMultiplicity, *pfJetneutralEMFraction, *pfJetchargedEMFraction;
    //Reco PFTau
    int nRecoPFTau;
    float *recopfTauEta,*recopfTauPhi,*recopfTauPt,*recopfTauJetPt,*recopfTauLeadTrackPt,*recopfTauLeadPionPt;
    int   *recopfTauTrkIso, *recopfTauGammaIso;
    float *recopfTauDiscrByTancOnePercent,*recopfTauDiscrByTancHalfPercent, *recopfTauDiscrByTancQuarterPercent, *recopfTauDiscrByTancTenthPercent, *recopfTauDiscrByIso, *recopfTauDiscrAgainstMuon, *recopfTauDiscrAgainstElec;
    //Reco PF jets
    float *jpfrecopt, *jpfrecoe,*jpfrecophi, *jpfrecoeta, *jpfreconeutralHadronFraction, *jpfreconeutralEMFraction, *jpfrecochargedHadronFraction, *jpfrecochargedEMFraction;
    int  *jpfreconeutralMultiplicity, *jpfrecochargedMultiplicity;
    int nrpj;

  
    // isolation/signal cands for recoPFTau and HLTPFtau
    int  noRecoPFTausSignal;
    int *signalTrToPFTauMatch;
    float *recoPFTauSignalTrDz;
    float *recoPFTauSignalTrPt;

    int noRecoPFTausIso;
    int *isoTrToPFTauMatch;
    float *recoPFTauIsoTrDz;
    float *recoPFTauIsoTrPt;

    int  noHLTPFTausSignal;
    int *hltpftauSignalTrToPFTauMatch;
    float *HLTPFTauSignalTrDz;
    float *HLTPFTauSignalTrPt;

    int noHLTPFTausIso;
    int *hltpftauIsoTrToPFTauMatch;
    float *HLTPFTauIsoTrDz;
    float *HLTPFTauIsoTrPt;

    reco::helper::JetIDHelper *jetID;


    // input variables
    bool _Monte,_Debug;
    float _CalJetMin, _GenJetMin;
    
    int evtCounter;
    
    static float etaBarrel() { return 1.4; }
    
    //create maps linking histogram pointers to HCAL Channel hits and digis
    TString gjetpfx, rjetpfx,gmetpfx, rmetpfx,calopfx;
    
};

#endif
