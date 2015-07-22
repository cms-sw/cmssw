#ifndef HLTINFO_H
#define HLTINFO_H

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TNamed.h"
#include <memory>
#include <vector>
#include <map>
#include "TROOT.h"
#include "TChain.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
# include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHFRingEtSums.h"
# include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHFBitCounts.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTProducer.h" 

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

//ccla
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Common/interface/Provenance.h"

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

#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"

namespace edm {
  class ConsumesCollector;
  class ParameterSet;
}

typedef std::vector<std::string> MyStrings;

/** \class HLTInfo
  *  
  * $Date: November 2006
  * $Revision: 
  * \author P. Bargassa - Rice U.
  */
class HLTInfo {
public:

  template <typename T>
  HLTInfo(edm::ParameterSet const& pset,
          edm::ConsumesCollector&& iC,
          T& module);

  template <typename T>
  HLTInfo(edm::ParameterSet const& pset,
          edm::ConsumesCollector& iC,
          T& module);

  void setup(const edm::ParameterSet& pSet, TTree* tree);
  void beginRun(const edm::Run& , const edm::EventSetup& );

  /** Analyze the Data */
  void analyze(const edm::Handle<edm::TriggerResults>                 & hltresults,
	       const edm::Handle<l1extra::L1EmParticleCollection>     & l1extemi,
	       const edm::Handle<l1extra::L1EmParticleCollection>     & l1extemn,
	       const edm::Handle<l1extra::L1MuonParticleCollection>   & l1extmu,
	       const edm::Handle<l1extra::L1JetParticleCollection>    & l1extjetc,
	       const edm::Handle<l1extra::L1JetParticleCollection>    & l1extjetf,
	       const edm::Handle<l1extra::L1JetParticleCollection>    & l1extjet,
	       const edm::Handle<l1extra::L1JetParticleCollection>    & l1exttaujet,
	       const edm::Handle<l1extra::L1EtMissParticleCollection> & l1extmet,
	       const edm::Handle<l1extra::L1EtMissParticleCollection> & l1extmht,
	       //const edm::Handle<l1extra::L1ParticleMapCollection>    & l1mapcoll,
	       const edm::Handle<L1GlobalTriggerReadoutRecord>        & l1GTRR,
	       const edm::Handle<L1GctHFBitCountsCollection>          & gctBitCounts,
	       const edm::Handle<L1GctHFRingEtSumsCollection>         & gctRingSums,	       
	       edm::EventSetup const& eventSetup,
	       edm::Event const& iEvent,
	       TTree* tree);

private:

  HLTInfo();

  // Tree variables
  float *hltppt, *hltpeta;
  float *l1extiemet, *l1extieme, *l1extiemeta, *l1extiemphi;
  float *l1extnemet, *l1extneme, *l1extnemeta, *l1extnemphi;
  float *l1extmupt, *l1extmue, *l1extmueta, *l1extmuphi;
  int *l1extmuchg;
  float *l1extjtcet, *l1extjtce, *l1extjtceta, *l1extjtcphi;
  float *l1extjtfet, *l1extjtfe, *l1extjtfeta, *l1extjtfphi;
  float *l1extjtet, *l1extjte, *l1extjteta, *l1extjtphi;
  float *l1exttauet, *l1exttaue, *l1exttaueta, *l1exttauphi;
  float met, metphi, ettot;
  float mht, mhtphi, ethad;
  int L1EvtCnt,HltEvtCnt,nhltpart,nl1extiem,nl1extnem,nl1extmu,nl1extjetc,nl1extjetf,nl1extjet,nl1extjt,nl1exttau;
  //int L1EvtCnt,HltEvtCnt,nhltpart,nl1extiem,nl1extnem,nl1extmu,nl1extjetc,nl1extjetf,nl1extjt,nl1exttau;
  int *trigflag, *l1flag, *l1flag5Bx, *l1techflag, *l1techflag5Bx, *l1extmuiso, *l1extmumip, *l1extmufor, *l1extmurpc, *l1extmuqul;
  int *trigPrescl, *l1Prescl, *l1techPrescl; 
  int l1hfRing1EtSumNegativeEta,l1hfRing2EtSumNegativeEta;
  int l1hfRing1EtSumPositiveEta,l1hfRing2EtSumPositiveEta;
  int l1hfTowerCountPositiveEtaRing1,l1hfTowerCountNegativeEtaRing1;
  int l1hfTowerCountPositiveEtaRing2,l1hfTowerCountNegativeEtaRing2;

  TString * algoBitToName;
  TString * techBitToName;
  std::vector<std::string> dummyBranches_;

  std::unique_ptr<HLTPrescaleProvider> hltPrescaleProvider_;
  std::string processName_;

  bool _OR_BXes;
  int UnpackBxInEvent; // save number of BXs unpacked in event

  // input variables
  bool _Debug;
};

template <typename T>
HLTInfo::HLTInfo(edm::ParameterSet const& pset,
                 edm::ConsumesCollector&& iC,
                 T& module) :
  HLTInfo(pset, iC, module) {
}

template <typename T>
HLTInfo::HLTInfo(edm::ParameterSet const& pset,
                 edm::ConsumesCollector& iC,
                 T& module) :
    HLTInfo() {
    hltPrescaleProvider_.reset(new HLTPrescaleProvider(pset, iC, module));
}

#endif
