#ifndef HLTALCA_H 
#define HLTALCA_H 
 
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TNamed.h"
#include <vector>
#include <map>
#include "TROOT.h"
#include "TChain.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h" 
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h" 
#include "DataFormats/Candidate/interface/Candidate.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

typedef std::vector<std::string> MyStrings;

/** \class HLTAlCa
  *  
  * $Date: November 2006
  * $Revision: 
  * \author P. Bargassa - Rice U.
  */
class HLTAlCa {
public:
  HLTAlCa(); 

  void setup(const edm::ParameterSet& pSet, TTree* tree);

  /** Analyze the Data */
  void analyze(const edm::Handle<EBRecHitCollection>             & ebrechits,
	       const edm::Handle<EERecHitCollection>             & eerechits,
               const edm::Handle<HBHERecHitCollection>           & hbherechits, 
               const edm::Handle<HORecHitCollection>             & horechits, 
               const edm::Handle<HFRecHitCollection>             & hfrechits, 
	       TTree* tree);


private:

  // Tree variables
  float ohHighestEnergyEERecHit, ohHighestEnergyEBRecHit;
  float ohHighestEnergyHBHERecHit, ohHighestEnergyHORecHit, ohHighestEnergyHFRecHit; 

  // input variables
  bool _Monte,_Debug;

  int evtCounter;

};

#endif
