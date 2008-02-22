#ifndef HLTEGAMMA_H
#define HLTEGAMMA_H

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TNamed.h"
#include <vector>
#include <map>
#include "TROOT.h"
#include "TChain.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "HLTrigger/HLTanalyzers/interface/JetUtil.h"
#include "HLTrigger/HLTanalyzers/interface/CaloTowerBoundries.h"

#include "DataFormats/METReco/interface/CaloMETCollection.h"

typedef std::vector<std::string> MyStrings;

/** \class HLTEgamma
  *  
  * $Date: November 2006
  * $Revision: 
  * \author P. Bargassa - Rice U.
  */
class HLTEgamma {
public:
  HLTEgamma(); 

  void setup(const edm::ParameterSet& pSet, TTree* tree);

  /** Analyze the Data */
  void analyze(const ElectronCollection& electron,
	       const PhotonCollection& photon,
	       const CaloGeometry& geom,
	       TTree* tree);

private:

  // Tree variables
  float *elpt, *elphi, *eleta, *elet, *ele; 
  float *photonpt, *photonphi, *photoneta, *photonet, *photone; 
  int nele,nphoton;

  // input variables
  bool _Monte,_Debug;

  int evtCounter;

  const float etaBarrel() {return 1.4;}

};

#endif
