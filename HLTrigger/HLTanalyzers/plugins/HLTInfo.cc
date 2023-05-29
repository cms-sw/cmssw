#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>
#include <cstdlib>
#include <cstring>

#include "HLTInfo.h"
#include "FWCore/Common/interface/TriggerNames.h"

// L1 related
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

//static const bool useL1EventSetup(true);
//static const bool useL1GtTriggerMenuLite(false);

HLTInfo::HLTInfo() {
  //set parameter defaults
  _Debug = false;
  _OR_BXes = false;
  UnpackBxInEvent = 1;
}

void HLTInfo::beginRun(const edm::Run& run, const edm::EventSetup& c) {
  bool changed(true);
  if (hltPrescaleProvider_->init(run, c, processName_, changed)) {
    // if init returns TRUE, initialisation has succeeded!
    if (changed) {
      // The HLT config has actually changed wrt the previous Run, hence rebook your
      // histograms or do anything else dependent on the revised HLT config
      std::cout << "Initalizing HLTConfigProvider" << std::endl;
    }
  } else {
    // if init returns FALSE, initialisation has NOT succeeded, which indicates a problem
    // with the file and/or code and needs to be investigated!
    std::cout << " HLT config extraction failure with process name " << processName_ << std::endl;
    // In this case, all access methods will return empty values!
  }
}

/*  Setup the analysis to put the branch-variables into the tree. */
void HLTInfo::setup(const edm::ParameterSet& pSet, TTree* HltTree) {
  processName_ = pSet.getParameter<std::string>("HLTProcessName");

  edm::ParameterSet myHltParams = pSet.getParameter<edm::ParameterSet>("RunParameters");
  std::vector<std::string> parameterNames = myHltParams.getParameterNames();

  for (auto& parameterName : parameterNames) {
    if (parameterName == "Debug")
      _Debug = myHltParams.getParameter<bool>(parameterName);
  }

  dummyBranches_ = pSet.getUntrackedParameter<std::vector<std::string> >("dummyBranches", std::vector<std::string>(0));

  HltEvtCnt = 0;
  const int kMaxTrigFlag = 10000;
  trigflag = new int[kMaxTrigFlag];
  trigPrescl = new double[kMaxTrigFlag];

  L1EvtCnt = 0;
  const int kMaxL1Flag = 10000;
  l1flag = new int[kMaxL1Flag];
  l1flag5Bx = new int[kMaxTrigFlag];
  l1Prescl = new int[kMaxL1Flag];

  l1techflag = new int[kMaxL1Flag];
  //  l1techflag5Bx = new int[kMaxTrigFlag];
  l1techPrescl = new int[kMaxTrigFlag];

  const int kMaxHLTPart = 10000;
  hltppt = new float[kMaxHLTPart];
  hltpeta = new float[kMaxHLTPart];

  algoBitToName = new TString[512];
  techBitToName = new TString[512];
}

/* **Analyze the event** */
void HLTInfo::analyze(const edm::Handle<edm::TriggerResults>& hltresults,
                      const edm::Handle<GlobalAlgBlkBxCollection>& l1results,
                      edm::EventSetup const& eventSetup,
                      edm::Event const& iEvent,
                      TTree* HltTree) {
  //   std::cout << " Beginning HLTInfo " << std::endl;

  /////////// Analyzing HLT Trigger Results (TriggerResults) //////////
  if (hltresults.isValid()) {
    int ntrigs = hltresults->size();
    if (ntrigs == 0) {
      std::cout << "%HLTInfo -- No trigger name given in TriggerResults of the input " << std::endl;
    }

    edm::TriggerNames const& triggerNames = iEvent.triggerNames(*hltresults);

    // 1st event : Book as many branches as trigger paths provided in the input...
    if (HltEvtCnt == 0) {
      for (int itrig = 0; itrig != ntrigs; ++itrig) {
        TString trigName = triggerNames.triggerName(itrig);
        HltTree->Branch(trigName, trigflag + itrig, trigName + "/I");
        HltTree->Branch(trigName + "_Prescl", trigPrescl + itrig, trigName + "_Prescl/I");
      }

      int itdum = ntrigs;
      for (auto& dummyBranche : dummyBranches_) {
        TString trigName(dummyBranche.data());
        bool addThisBranch = true;
        for (int itrig = 0; itrig != ntrigs; ++itrig) {
          TString realTrigName = triggerNames.triggerName(itrig);
          if (trigName == realTrigName)
            addThisBranch = false;
        }
        if (addThisBranch) {
          HltTree->Branch(trigName, trigflag + itdum, trigName + "/I");
          HltTree->Branch(trigName + "_Prescl", trigPrescl + itdum, trigName + "_Prescl/I");
          trigflag[itdum] = 0;
          trigPrescl[itdum] = 0;
          ++itdum;
        }
      }

      HltEvtCnt++;
    }
    // ...Fill the corresponding accepts in branch-variables

    //std::cout << "Number of prescale sets: " << hltConfig_.prescaleSize() << std::endl;
    //std::cout << "Number of HLT paths: " << hltConfig_.size() << std::endl;
    //int presclSet = hltConfig_.prescaleSet(iEvent, eventSetup);
    //std::cout<<"\tPrescale set number: "<< presclSet <<std::endl;

    for (int itrig = 0; itrig != ntrigs; ++itrig) {
      const std::string& trigName = triggerNames.triggerName(itrig);
      bool accept = hltresults->accept(itrig);

      trigPrescl[itrig] = hltPrescaleProvider_->prescaleValue<double>(iEvent, eventSetup, trigName);

      if (accept) {
        trigflag[itrig] = 1;
      } else {
        trigflag[itrig] = 0;
      }

      if (_Debug) {
        if (_Debug)
          std::cout << "%HLTInfo --  Number of HLT Triggers: " << ntrigs << std::endl;
        std::cout << "%HLTInfo --  HLTTrigger(" << itrig << "): " << trigName << " = " << accept << std::endl;
      }
    }
  } else {
    if (_Debug)
      std::cout << "%HLTInfo -- No Trigger Result" << std::endl;
  }

  //==============L1 information=======================================

  // L1 Triggers from Menu
  L1GtUtils const& l1GtUtils = hltPrescaleProvider_->l1GtUtils();

  //  m_l1GtUtils.retrieveL1EventSetup(eventSetup);
  //m_l1GtUtils.getL1GtRunCache(iEvent,eventSetup,useL1EventSetup,useL1GtTriggerMenuLite);
  /*
  unsigned long long id = eventSetup.get<L1TUtmTriggerMenuRcd>().cacheIdentifier();
  
  if (id != cache_id_) {
    cache_id_ = id; 
  */
  auto const& menu = eventSetup.getHandle(l1tUtmTriggerMenuToken_);

  //std::map<std::string, L1TUtmAlgorithm> const & algorithmMap_ = &(menu->getAlgorithmMap());
  /*
  // get the bit/name association
  for (auto const & keyval: menu->getAlgorithmMap()) {
    std::string const & name  = keyval.second.getName();
    unsigned int        index = keyval.second.getIndex();
    std::cerr << "bit: " << index << "\tname: " << name << std::endl;
  }
  */
  //} // end get menu

  int iErrorCode = -1;
  L1GtUtils::TriggerCategory trigCategory = L1GtUtils::AlgorithmTrigger;
  const int pfSetIndexAlgorithmTrigger = l1GtUtils.prescaleFactorSetIndex(iEvent, trigCategory, iErrorCode);
  if (iErrorCode == 0) {
    if (_Debug)
      std::cout << "%Prescale set index: " << pfSetIndexAlgorithmTrigger << std::endl;
  } else {
    std::cout << "%Could not extract Prescale set index from event record. Error code: " << iErrorCode << std::endl;
  }

  // 1st event : Book as many branches as trigger paths provided in the input...
  if (l1results.isValid()) {
    int ntrigs = l1results->size();
    if (ntrigs == 0) {
      std::cout << "%L1Results -- No trigger name given in TriggerResults of the input " << std::endl;
    }
    /*
    edm::TriggerNames const& triggerNames = iEvent.triggerNames(&results);
    // 1st event : Book as many branches as trigger paths provided in the input...
    */
    if (L1EvtCnt == 0) {
      // get the bit/name association
      for (auto const& keyval : menu->getAlgorithmMap()) {
        std::string const& trigName = keyval.second.getName();
        unsigned int index = keyval.second.getIndex();
        if (_Debug)
          std::cerr << "bit: " << index << "\tname: " << trigName << std::endl;

        int itrig = index;
        algoBitToName[itrig] = TString(trigName);

        TString l1trigName = static_cast<const char*>(algoBitToName[itrig]);
        std::string l1triggername = static_cast<const char*>(algoBitToName[itrig]);

        HltTree->Branch(l1trigName, l1flag + itrig, l1trigName + "/I");
        HltTree->Branch(l1trigName + "_Prescl", l1Prescl + itrig, l1trigName + "_Prescl/I");

      }  // end algo Map

      L1EvtCnt++;
    }  // end l1evtCnt=0

    GlobalAlgBlk const& result = l1results->at(0, 0);

    // get the individual decisions from the GlobalAlgBlk
    for (unsigned int itrig = 0; itrig < result.maxPhysicsTriggers; ++itrig) {
      //      std::cerr << "bit: " << itrig << "\tresult: " << results.getAlgoDecisionFinal(itrig) << std::endl;

      bool myflag = result.getAlgoDecisionFinal(itrig);
      if (myflag) {
        l1flag[itrig] = 1;
      } else {
        l1flag[itrig] = 0;
      }

      std::string l1triggername = static_cast<const char*>(algoBitToName[itrig]);
      l1Prescl[itrig] = l1GtUtils.prescaleFactor(iEvent, l1triggername, iErrorCode);

      if (_Debug)
        std::cout << "L1 TD: " << itrig << " " << algoBitToName[itrig] << " " << l1flag[itrig] << " " << l1Prescl[itrig]
                  << std::endl;
    }

    //    L1EvtCnt++;
    if (_Debug)
      std::cout << "%L1Info -- Done with routine" << std::endl;

  }  // l1results.isValid
  else {
    if (_Debug)
      std::cout << "%L1Results -- No Trigger Result" << std::endl;
  }
}
