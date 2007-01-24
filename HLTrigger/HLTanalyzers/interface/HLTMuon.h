#ifndef HLTMUON_H
#define HLTMUON_H

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TNamed.h"
#include <vector>
#include <map>
#include "TROOT.h"
#include "TChain.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "HLTrigger/HLTanalyzers/interface/JetUtil.h"
#include "HLTrigger/HLTanalyzers/interface/CaloTowerBoundries.h"

#include "DataFormats/METReco/interface/CaloMETCollection.h"

typedef std::vector<std::string> MyStrings;

/** \class HLTMuon
  *  
  * $Date: November 2006
  * $Revision: 
  * \author P. Bargassa - Rice U.
  */
class HLTMuon {
public:
  HLTMuon(); 

  void setup(const edm::ParameterSet& pSet, TTree* tree);

  /** Analyze the Data */
  void analyze(const MuonCollection& muon,
	       const CaloGeometry& geom,
	       TTree* tree);

private:

  // Tree variables
  float *muonpt, *muonphi, *muoneta, *muonet, *muone; 
  int nmuon;

  // input variables
  bool _Monte,_Debug;

  int evtCounter;

  const float etaBarrel() {return 1.4;}

};

#endif
