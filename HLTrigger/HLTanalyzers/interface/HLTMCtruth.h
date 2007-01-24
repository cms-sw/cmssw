#ifndef HLTMCTRUTH_H
#define HLTMCTRUTH_H

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TNamed.h"
#include <vector>
#include <map>
#include "TROOT.h"
#include "TChain.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/METReco/interface/CaloMETCollection.h"

typedef std::vector<std::string> MyStrings;

/** \class HLTMCtruth
  *  
  * $Date: November 2006
  * $Revision: 
  * \author P. Bargassa - Rice U.
  */
class HLTMCtruth {
public:
  HLTMCtruth(); 

  void setup(const edm::ParameterSet& pSet, TTree* tree);

  /** Analyze the Data */
  void analyze(const HepMC::GenEvent mctruth,
	       TTree* tree);

private:

  // Tree variables
  float *mcpid, *mcvx, *mcvy, *mcvz, *mcpt;
  int nmcpart;

  // input variables
  bool _Monte,_Debug;

};

#endif
