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
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTProducer.h" 

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
/* #include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h" */
/* #include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h" */
/* #include "CondFormats/L1TObjects/interface/L1CaloEtScale.h" */
/* #include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h" */
/* #include "CondFormats/L1TObjects/interface/L1RCTParameters.h" */
/* #include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h" */
/* #include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h" */
/* #include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"  */

#include "HLTrigger/HLTanalyzers/interface/JetUtil.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

// #include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
//#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

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
  void analyze(/*const HLTFilterObjectWithRefs& hltobj,*/
	       const edm::TriggerResults& hltresults,
	       const l1extra::L1EmParticleCollection& l1extemi,
	       const l1extra::L1EmParticleCollection& l1extemn,
	       const l1extra::L1MuonParticleCollection& l1extmu,
	       const l1extra::L1JetParticleCollection& l1extjetc,
	       const l1extra::L1JetParticleCollection& l1extjetf,
	       const l1extra::L1JetParticleCollection& l1exttaujet,
	       const l1extra::L1EtMissParticleCollection& l1extmet,
//	       const l1extra::L1ParticleMapCollection& l1mapcoll,
	       const L1GlobalTriggerReadoutRecord& L1GTRR,
	       const L1GlobalTriggerObjectMapRecord& L1GTOMRec,
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
  int L1EvtCnt,HltEvtCnt,nhltpart,nl1extiem,nl1extnem,nl1extmu,nl1extjetc,nl1extjetf,nl1extjt,nl1exttau;
  int *trigflag, *l1flag, *l1extmuiso, *l1extmumip, *l1extmufor, *l1extmurpc, *l1extmuqul;

  // input variables
  bool _Debug;

  // trigger names
  edm::TriggerNames triggerNames_;

};

#endif
