#ifndef HLTHIGGSTRUTH_H
#define HLTHIGGSTRUTH_H

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

/** \class HLTHiggsTruth
  *  
  * $Date: November 2006
  * $Revision: 
  * \author P. Bargassa - Rice U.
  */
class HLTHiggsTruth {
public:
  HLTHiggsTruth(); 

  void setup(const edm::ParameterSet& pSet, TTree* tree);

  /** Analyze the Data */
//  void analyzeHWW2l(const HepMC::GenEvent mctruth,
//	       TTree* tree);
//  void analyzeHWW2l(const CandidateCollection& mctruth,
//               const HepMC::GenEvent hepmc,
//	       TTree* tree);
  void analyzeHWW2l(const CandidateCollection& mctruth,
	       TTree* tree);
//  inline bool decision_WW() const {return isvisible_WW;};

  void analyzeHZZ4l(const CandidateCollection& mctruth,TTree* tree);
//  inline bool decision() const {return isvisible_ZZ;};

  void analyzeHgg(const CandidateCollection& mctruth,TTree* tree);
//  inline bool decision() const {return isvisible_gg;};

  void analyzeH2tau(const CandidateCollection& mctruth,TTree* tree);
//  inline bool decision() const {return isvisible_2tau;};

  void analyzeHtaunu(const CandidateCollection& mctruth,TTree* tree);
//  inline bool decision() const {return isvisible_taunu;};

  void analyzeA2mu(const CandidateCollection& mctruth,TTree* tree);
//  inline bool decision() const {return isvisible_2mu;};

  void analyzeHinv(const CandidateCollection& mctruth,TTree* tree);
  inline bool decision() const {return isvisible;};

private:

  // Tree variables
//  float *mcpid, *mcvx, *mcvy, *mcvz, *mcpt;
//  bool isvisible_WW, isvisible_ZZ, isvisible_gg, isvisible_2tau, isvisible_taunu,
//       isvisible_2mu, isvisible_inv,isvisible;
  bool isvisible;

  // input variables
  bool _Monte,_Debug;

};

#endif
