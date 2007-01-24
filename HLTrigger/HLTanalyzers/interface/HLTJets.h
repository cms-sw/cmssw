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
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetfwd.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetfwd.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "HLTrigger/HLTanalyzers/interface/JetUtil.h"
#include "HLTrigger/HLTanalyzers/interface/CaloTowerBoundries.h"

typedef std::vector<std::string> MyStrings;

/** \class HLTJets
  *  
  * $Date: November 2006
  * $Revision: 
  * \author L. Apanasevich - UIC, P. Bargassa - Rice U.
  */
class HLTJets {
public:
  HLTJets(); 

  void setup(const edm::ParameterSet& pSet, TTree* tree);

  /** Analyze the Data */
  void analyze(const CaloJetCollection& rjets,
	       const GenJetCollection& gjets,
	       const CaloMETCollection& rmets,
	       const GenMETCollection& gmets,
	       const CaloTowerCollection& caloTowers,
	       const CaloGeometry& geom,
	       TTree* tree);

private:

  // Tree variables
  float *jcalpt, *jcalphi, *jcaleta, *jcalet, *jcale;
  float *jgenpt, *jgenphi, *jgeneta, *jgenet, *jgene;
  float *towet, *toweta, *towphi, *towen, *towem, *towhd, *towoe;
  float mcalmet,mcalphi,mcalsum;
  float mgenmet,mgenphi,mgensum;
  int njetcal,njetgen,ntowcal;

  // input variables
  bool _Monte,_Debug;

  int evtCounter;

  const float etaBarrel() {return 1.4;}

  //create maps linking histogram pointers to HCAL Channel hits and digis
  TString gjetpfx, rjetpfx,gmetpfx, rmetpfx,calopfx;

};

#endif
