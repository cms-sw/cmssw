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
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/METCollection.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/TauReco/interface/HLTTau.h"

#include "HLTrigger/HLTanalyzers/interface/JetUtil.h"
#include "HLTrigger/HLTanalyzers/interface/CaloTowerBoundries.h"

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


class HLTJets {
public:
  HLTJets(); 

  void setup(const edm::ParameterSet& pSet, TTree* tree);

  /** Analyze the Data */
  void analyze(const edm::Handle<reco::CaloJetCollection>      & recojets,
	       const edm::Handle<reco::CaloJetCollection>      & corjets,
	       const edm::Handle<reco::GenJetCollection>       & gjets,
	       const edm::Handle<reco::CaloMETCollection>      & rmets,
	       const edm::Handle<reco::GenMETCollection>       & gmets,
	       const edm::Handle<reco::METCollection>          & ht,
	       const edm::Handle<reco::HLTTauCollection> & myHLTTau,
	       const edm::Handle<CaloTowerCollection>    & caloTowers,
	       double thresholdForSavingTowers,
	       TTree * tree);

private:

  // Tree variables
  float *jcalpt, *jcalphi, *jcaleta, *jcale, *jcalemf, *jcaln90;
  float *jcorcalpt, *jcorcalphi, *jcorcaleta, *jcorcale, *jcorcalemf, *jcorcaln90;
  float *jgenpt, *jgenphi, *jgeneta, *jgene;
  float *towet, *toweta, *towphi, *towen, *towem, *towhd, *towoe;
  float mcalmet,mcalphi,mcalsum;
  float htcalet,htcalphi,htcalsum;
  float mgenmet,mgenphi,mgensum;
  int njetcal,ncorjetcal,njetgen,ntowcal;

   // Taus
  float *l2tauemiso, *l25tauPt, *l3tauPt;
  int *l25tautckiso, *l3tautckiso;
  int nohtau;
  float *tauEta, *tauPt, *tauPhi; 
  
  // input variables
  bool _Monte,_Debug;
  float _CalJetMin, _GenJetMin;

  int evtCounter;

  const float etaBarrel() {return 1.4;}

  //create maps linking histogram pointers to HCAL Channel hits and digis
  TString gjetpfx, rjetpfx,gmetpfx, rmetpfx,calopfx;

};

#endif
