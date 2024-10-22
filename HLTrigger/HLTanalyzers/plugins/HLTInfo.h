#ifndef HLTINFO_H
#define HLTINFO_H

#include <memory>
#include <vector>
#include <map>

// ROOT
#include <TChain.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TNamed.h>
#include <TROOT.h>

// CMSSW
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "CondFormats/L1TObjects/interface/L1TUtmAlgorithm.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"

#include "JetUtil.h"

namespace edm {
  class ConsumesCollector;
  class ParameterSet;
}  // namespace edm

typedef std::vector<std::string> MyStrings;

/** \class HLTInfo
  *
  * $Date: November 2006
  * $Revision:
  * \author P. Bargassa - Rice U.

  * $Date: April 2016
  * $Revision:
  * \author G. Karapostoli - ULB
  */
class HLTInfo {
public:
  //HLTInfo();

  template <typename T>
  HLTInfo(edm::ParameterSet const& pset, edm::ConsumesCollector&& iC, T& module);

  template <typename T>
  HLTInfo(edm::ParameterSet const& pset, edm::ConsumesCollector& iC, T& module);

  void setup(const edm::ParameterSet& pSet, TTree* tree);
  void beginRun(const edm::Run&, const edm::EventSetup&);

  /** Analyze the Data */
  void analyze(const edm::Handle<edm::TriggerResults>& hltresults,
               const edm::Handle<GlobalAlgBlkBxCollection>& l1results,
               edm::EventSetup const& eventSetup,
               edm::Event const& iEvent,
               TTree* tree);

private:
  HLTInfo();

  edm::ESGetToken<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd> l1tUtmTriggerMenuToken_;

  // Tree variables
  float *hltppt, *hltpeta;
  int L1EvtCnt, HltEvtCnt, nhltpart;

  int *trigflag, *l1flag, *l1flag5Bx, *l1techflag;
  double* trigPrescl;
  int *l1Prescl, *l1techPrescl;

  TString* algoBitToName;
  TString* techBitToName;
  std::vector<std::string> dummyBranches_;

  //HLTConfigProvider hltConfig_;
  //L1GtUtils m_l1GtUtils;
  std::unique_ptr<HLTPrescaleProvider> hltPrescaleProvider_;
  std::string processName_;

  bool _OR_BXes;
  int UnpackBxInEvent;  // save number of BXs unpacked in event

  // input variables

  // L1 uGT menu
  unsigned long long cache_id_;

  /*
  edm::ESHandle<L1TUtmTriggerMenu> menu;
  //std::map<std::string, L1TUtmAlgorithm> const & algorithmMap_;
  const std::map<std::string, L1TUtmAlgorithm>* algorithmMap_;
  */
  bool _Debug;
};

template <typename T>
HLTInfo::HLTInfo(edm::ParameterSet const& pset, edm::ConsumesCollector&& iC, T& module) : HLTInfo(pset, iC, module) {}

template <typename T>
HLTInfo::HLTInfo(edm::ParameterSet const& pset, edm::ConsumesCollector& iC, T& module) : HLTInfo() {
  l1tUtmTriggerMenuToken_ = iC.esConsumes<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd>();
  hltPrescaleProvider_ = std::make_unique<HLTPrescaleProvider>(pset, iC, module);
}

#endif
