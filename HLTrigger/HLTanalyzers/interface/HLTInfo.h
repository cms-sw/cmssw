#ifndef HLTINFO_H
#define HLTINFO_H

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TNamed.h"
#include <vector>
#include <map>
#include "TROOT.h"
#include "TChain.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "HLTrigger/HLTanalyzers/interface/JetUtil.h"

#include "DataFormats/METReco/interface/CaloMETCollection.h"

typedef std::vector<std::string> MyStrings;

/** \class HLTInfo
  *  
  * $Date: November 2006
  * $Revision: 
  * \author P. Bargassa - Rice U.
  */
class HLTInfo {
public:
  HLTInfo(); 

  void setup(const edm::ParameterSet& pSet, TTree* tree);

  /** Analyze the Data */
  void analyze(const HLTFilterObjectWithRefs& hltobj,
	       const edm::TriggerResults& hltresults,
	       const l1extra::L1EmParticleCollection& l1extemi,
	       const l1extra::L1EmParticleCollection& l1extemn,
	       const l1extra::L1MuonParticleCollection& l1extmu,
	       const l1extra::L1JetParticleCollection& l1extjetc,
	       const l1extra::L1JetParticleCollection& l1extjetf,
	       const l1extra::L1JetParticleCollection& l1exttaujet,
	       const l1extra::L1EtMissParticle& l1extmet,
	       TTree* tree);

private:

  // Tree variables
  float *hltppt, *hltpeta;
  float *l1extiemet, *l1extieme, *l1extiemeta, *l1extiemphi;
  float *l1extnemet, *l1extneme, *l1extnemeta, *l1extnemphi;
  float *l1extmupt, *l1extmue, *l1extmueta, *l1extmuphi;
  float *l1extjtcet, *l1extjtce, *l1extjtceta, *l1extjtcphi;
  float *l1extjtfet, *l1extjtfe, *l1extjtfeta, *l1extjtfphi;
  float *l1exttauet, *l1exttaue, *l1exttaueta, *l1exttauphi;
  float met, metphi, mettot, methad;
  int evtCount,nhltpart,nl1extiem,nl1extnem,nl1extmu,nl1extjetc,nl1extjetf,nl1extjt,nl1exttau;
  int *trigflag, *l1extmuiso, *l1extmumip;

  // input variables
  bool _Debug;

  // trigger names
  edm::TriggerNames triggerNames_;

};

#endif
