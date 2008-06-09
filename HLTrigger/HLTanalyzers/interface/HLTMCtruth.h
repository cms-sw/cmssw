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
  void analyze(const CandidateView& mctruth,
	       //const HepMC::GenEvent hepmc,
	       const double pthat,
	       TTree* tree);

private:

  // Tree variables
  float *mcvx, *mcvy, *mcvz, *mcpt, *mceta, *mcphi;
  int *mcpid;
  int nmcpart,nmu3,nab,nbb;
  float pthatf;
  // input variables
  bool _Monte,_Debug;

};

#endif
