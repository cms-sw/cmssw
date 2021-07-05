// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      pat::PATTriggerEventProducer
//
//
/**
  \class    pat::PATTriggerEventProducer PATTriggerEventProducer.h "PhysicsTools/PatAlgos/plugins/PATTriggerEventProducer.h"
  \brief    Produces the central entry point to full PAT trigger information

   This producer extract general trigger and conditions information from
   - the edm::TriggerResults written by the HLT process,
   - the ConditionsInEdm products,
   - the process history and
   - the GlobalTrigger information in the event and the event setup
   and writes it together with links to the full PAT trigger information collections and PAT trigger match results to
   - the pat::TriggerEvent

   For me information, s.
   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATTrigger

  \author   Volker Adler
  \version  $Id: PATTriggerEventProducer.h,v 1.11 2010/11/27 15:16:20 vadler Exp $
*/

#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Framework/interface/InputTagMatch.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/Common/interface/AssociativeIterator.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/PatCandidates/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/ConditionsInEdm.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include <cassert>
#include <string>
#include <vector>

namespace pat {

  class PATTriggerEventProducer : public edm::stream::EDProducer<> {
  public:
    explicit PATTriggerEventProducer(const edm::ParameterSet& iConfig);
    ~PATTriggerEventProducer() override{};

  private:
    void beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
    void beginLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup) override;
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    std::string nameProcess_;  // configuration
    bool autoProcessName_;
    edm::InputTag tagTriggerProducer_;  // configuration (optional with default)
    edm::EDGetTokenT<TriggerAlgorithmCollection> triggerAlgorithmCollectionToken_;
    edm::EDGetTokenT<TriggerConditionCollection> triggerConditionCollectionToken_;
    edm::EDGetTokenT<TriggerPathCollection> triggerPathCollectionToken_;
    edm::EDGetTokenT<TriggerFilterCollection> triggerFilterCollectionToken_;
    edm::EDGetTokenT<TriggerObjectCollection> triggerObjectCollectionToken_;
    std::vector<edm::InputTag> tagsTriggerMatcher_;  // configuration (optional)
    std::vector<edm::EDGetTokenT<TriggerObjectStandAloneMatch> > triggerMatcherTokens_;
    // L1
    edm::InputTag tagL1Gt_;  // configuration (optional with default)
    edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> l1GtToken_;
    // HLT
    HLTConfigProvider hltConfig_;
    bool hltConfigInit_;
    edm::InputTag tagTriggerResults_;  // configuration (optional with default)
    edm::GetterOfProducts<edm::TriggerResults> triggerResultsGetter_;
    edm::InputTag tagTriggerEvent_;  // configuration (optional with default)
    // Conditions
    edm::InputTag tagCondGt_;  // configuration (optional with default)
    edm::EDGetTokenT<edm::ConditionsInRunBlock> tagCondGtRunToken_;
    edm::EDGetTokenT<edm::ConditionsInLumiBlock> tagCondGtLumiToken_;
    edm::EDGetTokenT<edm::ConditionsInEventBlock> tagCondGtEventToken_;
    edm::ConditionsInRunBlock condRun_;
    edm::ConditionsInLumiBlock condLumi_;
    bool gtCondRunInit_;
    bool gtCondLumiInit_;
  };

}  // namespace pat

using namespace pat;
using namespace edm;

PATTriggerEventProducer::PATTriggerEventProducer(const ParameterSet& iConfig)
    : nameProcess_(iConfig.getParameter<std::string>("processName")),
      autoProcessName_(nameProcess_ == "*"),
      tagTriggerProducer_("patTrigger"),
      tagsTriggerMatcher_(),
      // L1 configuration parameters
      tagL1Gt_(),
      // HLTConfigProvider
      hltConfigInit_(false),
      // HLT configuration parameters
      tagTriggerResults_("TriggerResults"),
      tagTriggerEvent_("hltTriggerSummaryAOD"),
      // Conditions configuration parameters
      tagCondGt_(),
      // Conditions
      condRun_(),
      condLumi_(),
      gtCondRunInit_(false),
      gtCondLumiInit_(false) {
  if (iConfig.exists("triggerResults"))
    tagTriggerResults_ = iConfig.getParameter<InputTag>("triggerResults");
  triggerResultsGetter_ =
      GetterOfProducts<TriggerResults>(InputTagMatch(InputTag(tagTriggerResults_.label(),
                                                              tagTriggerResults_.instance(),
                                                              autoProcessName_ ? std::string("") : nameProcess_)),
                                       this);
  if (iConfig.exists("triggerEvent"))
    tagTriggerEvent_ = iConfig.getParameter<InputTag>("triggerEvent");
  if (iConfig.exists("patTriggerProducer"))
    tagTriggerProducer_ = iConfig.getParameter<InputTag>("patTriggerProducer");
  triggerAlgorithmCollectionToken_ = mayConsume<TriggerAlgorithmCollection>(tagTriggerProducer_);
  triggerConditionCollectionToken_ = mayConsume<TriggerConditionCollection>(tagTriggerProducer_);
  triggerPathCollectionToken_ = mayConsume<TriggerPathCollection>(tagTriggerProducer_);
  triggerFilterCollectionToken_ = mayConsume<TriggerFilterCollection>(tagTriggerProducer_);
  triggerObjectCollectionToken_ = mayConsume<TriggerObjectCollection>(tagTriggerProducer_);
  if (iConfig.exists("condGtTag")) {
    tagCondGt_ = iConfig.getParameter<InputTag>("condGtTag");
    tagCondGtRunToken_ = mayConsume<ConditionsInRunBlock, InRun>(tagCondGt_);
    tagCondGtLumiToken_ = mayConsume<ConditionsInLumiBlock, InLumi>(tagCondGt_);
    tagCondGtEventToken_ = mayConsume<ConditionsInEventBlock>(tagCondGt_);
  }
  if (iConfig.exists("l1GtTag"))
    tagL1Gt_ = iConfig.getParameter<InputTag>("l1GtTag");
  l1GtToken_ = mayConsume<L1GlobalTriggerReadoutRecord>(tagL1Gt_);
  if (iConfig.exists("patTriggerMatches"))
    tagsTriggerMatcher_ = iConfig.getParameter<std::vector<InputTag> >("patTriggerMatches");
  triggerMatcherTokens_ = vector_transform(
      tagsTriggerMatcher_, [this](InputTag const& tag) { return mayConsume<TriggerObjectStandAloneMatch>(tag); });

  callWhenNewProductsRegistered([this](BranchDescription const& bd) {
    if (not(this->autoProcessName_ and bd.processName() == this->moduleDescription().processName())) {
      triggerResultsGetter_(bd);
    }
  });

  for (size_t iMatch = 0; iMatch < tagsTriggerMatcher_.size(); ++iMatch) {
    produces<TriggerObjectMatch>(tagsTriggerMatcher_.at(iMatch).label());
  }
  produces<TriggerEvent>();
}

void PATTriggerEventProducer::beginRun(const Run& iRun, const EventSetup& iSetup) {
  // Initialize process name
  if (autoProcessName_) {
    // reset
    nameProcess_ = "*";
    // determine process name from last run TriggerSummaryProducerAOD module in process history of input
    const ProcessHistory& processHistory(iRun.processHistory());
    ProcessConfiguration processConfiguration;
    ParameterSet processPSet;
    // unbroken loop, which relies on time ordering (accepts the last found entry)
    for (ProcessHistory::const_iterator iHist = processHistory.begin(); iHist != processHistory.end(); ++iHist) {
      if (processHistory.getConfigurationForProcess(iHist->processName(), processConfiguration) &&
          pset::Registry::instance()->getMapped(processConfiguration.parameterSetID(), processPSet) &&
          processPSet.exists(tagTriggerEvent_.label())) {
        nameProcess_ = iHist->processName();
        LogDebug("autoProcessName") << "HLT process name '" << nameProcess_ << "' discovered";
      }
    }
    // terminate, if nothing is found
    if (nameProcess_ == "*") {
      LogError("autoProcessName") << "trigger::TriggerEvent product with label '" << tagTriggerEvent_.label()
                                  << "' not produced according to process history of input data\n"
                                  << "No trigger information produced.";
      return;
    }
    LogInfo("autoProcessName") << "HLT process name " << nameProcess_ << " used for PAT trigger information";
  }
  // adapt configuration of used input tags
  if (tagTriggerResults_.process().empty() || tagTriggerResults_.process() == "*") {
    tagTriggerResults_ = InputTag(tagTriggerResults_.label(), tagTriggerResults_.instance(), nameProcess_);
  } else if (tagTriggerResults_.process() != nameProcess_) {
    LogWarning("triggerResultsTag") << "TriggerResults process name '" << tagTriggerResults_.process()
                                    << "' differs from HLT process name '" << nameProcess_ << "'";
  }
  if (tagTriggerEvent_.process().empty() || tagTriggerEvent_.process() == "*") {
    tagTriggerEvent_ = InputTag(tagTriggerEvent_.label(), tagTriggerEvent_.instance(), nameProcess_);
  } else if (tagTriggerEvent_.process() != nameProcess_) {
    LogWarning("triggerEventTag") << "TriggerEvent process name '" << tagTriggerEvent_.process()
                                  << "' differs from HLT process name '" << nameProcess_ << "'";
  }

  gtCondRunInit_ = false;
  if (!tagCondGt_.label().empty()) {
    Handle<ConditionsInRunBlock> condRunBlock;
    iRun.getByToken(tagCondGtRunToken_, condRunBlock);
    if (condRunBlock.isValid()) {
      condRun_ = *condRunBlock;
      gtCondRunInit_ = true;
    } else {
      LogError("conditionsInEdm") << "ConditionsInRunBlock product with InputTag '" << tagCondGt_.encode()
                                  << "' not in run";
    }
  }

  // Initialize HLTConfigProvider
  hltConfigInit_ = false;
  bool changed(true);
  if (!hltConfig_.init(iRun, iSetup, nameProcess_, changed)) {
    LogError("hltConfigExtraction") << "HLT config extraction error with process name '" << nameProcess_ << "'";
  } else if (hltConfig_.size() <= 0) {
    LogError("hltConfigSize") << "HLT config size error";
  } else
    hltConfigInit_ = true;
}

void PATTriggerEventProducer::beginLuminosityBlock(const LuminosityBlock& iLuminosityBlock, const EventSetup& iSetup) {
  // Terminate, if auto process name determination failed
  if (nameProcess_ == "*")
    return;

  gtCondLumiInit_ = false;
  if (!tagCondGt_.label().empty()) {
    Handle<ConditionsInLumiBlock> condLumiBlock;
    iLuminosityBlock.getByToken(tagCondGtLumiToken_, condLumiBlock);
    if (condLumiBlock.isValid()) {
      condLumi_ = *condLumiBlock;
      gtCondLumiInit_ = true;
    } else {
      LogError("conditionsInEdm") << "ConditionsInLumiBlock product with InputTag '" << tagCondGt_.encode()
                                  << "' not in lumi";
    }
  }
}

void PATTriggerEventProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  // Terminate, if auto process name determination failed
  if (nameProcess_ == "*")
    return;

  if (!hltConfigInit_)
    return;

  ESHandle<L1GtTriggerMenu> handleL1GtTriggerMenu;
  iSetup.get<L1GtTriggerMenuRcd>().get(handleL1GtTriggerMenu);
  Handle<TriggerResults> handleTriggerResults;
  iEvent.getByLabel(tagTriggerResults_, handleTriggerResults);
  //   iEvent.getByToken( triggerResultsToken_, handleTriggerResults );
  if (!handleTriggerResults.isValid()) {
    LogError("triggerResultsValid") << "TriggerResults product with InputTag '" << tagTriggerResults_.encode()
                                    << "' not in event\n"
                                    << "No trigger information produced";
    return;
  }
  Handle<TriggerAlgorithmCollection> handleTriggerAlgorithms;
  iEvent.getByToken(triggerAlgorithmCollectionToken_, handleTriggerAlgorithms);
  Handle<TriggerConditionCollection> handleTriggerConditions;
  iEvent.getByToken(triggerConditionCollectionToken_, handleTriggerConditions);
  Handle<TriggerPathCollection> handleTriggerPaths;
  iEvent.getByToken(triggerPathCollectionToken_, handleTriggerPaths);
  Handle<TriggerFilterCollection> handleTriggerFilters;
  iEvent.getByToken(triggerFilterCollectionToken_, handleTriggerFilters);
  Handle<TriggerObjectCollection> handleTriggerObjects;
  iEvent.getByToken(triggerObjectCollectionToken_, handleTriggerObjects);

  bool physDecl(false);
  if (iEvent.isRealData() && !tagL1Gt_.label().empty()) {
    Handle<L1GlobalTriggerReadoutRecord> handleL1GlobalTriggerReadoutRecord;
    iEvent.getByToken(l1GtToken_, handleL1GlobalTriggerReadoutRecord);
    if (handleL1GlobalTriggerReadoutRecord.isValid()) {
      L1GtFdlWord fdlWord = handleL1GlobalTriggerReadoutRecord->gtFdlWord();
      if (fdlWord.physicsDeclared() == 1) {
        physDecl = true;
      }
    } else {
      LogError("l1GlobalTriggerReadoutRecordValid")
          << "L1GlobalTriggerReadoutRecord product with InputTag '" << tagL1Gt_.encode() << "' not in event";
    }
  } else {
    physDecl = true;
  }

  // produce trigger event

  auto triggerEvent = std::make_unique<TriggerEvent>(handleL1GtTriggerMenu->gtTriggerMenuName(),
                                                     std::string(hltConfig_.tableName()),
                                                     handleTriggerResults->wasrun(),
                                                     handleTriggerResults->accept(),
                                                     handleTriggerResults->error(),
                                                     physDecl);
  // set product references to trigger collections
  if (handleTriggerAlgorithms.isValid()) {
    triggerEvent->setAlgorithms(handleTriggerAlgorithms);
  } else {
    LogError("triggerAlgorithmsValid") << "pat::TriggerAlgorithmCollection product with InputTag '"
                                       << tagTriggerProducer_.encode() << "' not in event";
  }
  if (handleTriggerConditions.isValid()) {
    triggerEvent->setConditions(handleTriggerConditions);
  } else {
    LogError("triggerConditionsValid") << "pat::TriggerConditionCollection product with InputTag '"
                                       << tagTriggerProducer_.encode() << "' not in event";
  }
  if (handleTriggerPaths.isValid()) {
    triggerEvent->setPaths(handleTriggerPaths);
  } else {
    LogError("triggerPathsValid") << "pat::TriggerPathCollection product with InputTag '"
                                  << tagTriggerProducer_.encode() << "' not in event";
  }
  if (handleTriggerFilters.isValid()) {
    triggerEvent->setFilters(handleTriggerFilters);
  } else {
    LogError("triggerFiltersValid") << "pat::TriggerFilterCollection product with InputTag '"
                                    << tagTriggerProducer_.encode() << "' not in event";
  }
  if (handleTriggerObjects.isValid()) {
    triggerEvent->setObjects(handleTriggerObjects);
  } else {
    LogError("triggerObjectsValid") << "pat::TriggerObjectCollection product with InputTag '"
                                    << tagTriggerProducer_.encode() << "' not in event";
  }
  if (gtCondRunInit_) {
    triggerEvent->setLhcFill(condRun_.lhcFillNumber);
    triggerEvent->setBeamMode(condRun_.beamMode);
    triggerEvent->setBeamMomentum(condRun_.beamMomentum);
    triggerEvent->setBCurrentStart(condRun_.BStartCurrent);
    triggerEvent->setBCurrentStop(condRun_.BStopCurrent);
    triggerEvent->setBCurrentAvg(condRun_.BAvgCurrent);
  }
  if (gtCondLumiInit_) {
    triggerEvent->setIntensityBeam1(condLumi_.totalIntensityBeam1);
    triggerEvent->setIntensityBeam2(condLumi_.totalIntensityBeam2);
  }
  if (!tagCondGt_.label().empty()) {
    Handle<ConditionsInEventBlock> condEventBlock;
    iEvent.getByToken(tagCondGtEventToken_, condEventBlock);
    if (condEventBlock.isValid()) {
      triggerEvent->setBstMasterStatus(condEventBlock->bstMasterStatus);
      triggerEvent->setTurnCount(condEventBlock->turnCountNumber);
    } else {
      LogError("conditionsInEdm") << "ConditionsInEventBlock product with InputTag '" << tagCondGt_.encode()
                                  << "' not in event";
    }
  }

  // produce trigger match association and set references
  if (handleTriggerObjects.isValid()) {
    for (size_t iMatch = 0; iMatch < tagsTriggerMatcher_.size(); ++iMatch) {
      const std::string labelTriggerObjectMatcher(tagsTriggerMatcher_.at(iMatch).label());
      // copy trigger match association using TriggerObjectStandAlone to those using TriggerObject
      // relying on the fact, that only one candidate collection is present in the association
      Handle<TriggerObjectStandAloneMatch> handleTriggerObjectStandAloneMatch;
      iEvent.getByToken(triggerMatcherTokens_.at(iMatch), handleTriggerObjectStandAloneMatch);
      if (!handleTriggerObjectStandAloneMatch.isValid()) {
        LogError("triggerMatchValid") << "pat::TriggerObjectStandAloneMatch product with InputTag '"
                                      << labelTriggerObjectMatcher << "' not in event";
        continue;
      }
      auto it = makeAssociativeIterator<reco::CandidateBaseRef>(*handleTriggerObjectStandAloneMatch, iEvent);
      auto itEnd = it.end();
      Handle<reco::CandidateView> handleCands;
      if (it != itEnd)
        iEvent.get(it->first.id(), handleCands);
      std::vector<int> indices;
      while (it != itEnd) {
        indices.push_back(it->second.key());
        ++it;
      }
      auto triggerObjectMatch = std::make_unique<TriggerObjectMatch>(handleTriggerObjects);
      TriggerObjectMatch::Filler matchFiller(*triggerObjectMatch);
      if (handleCands.isValid()) {
        matchFiller.insert(handleCands, indices.begin(), indices.end());
      }
      matchFiller.fill();
      OrphanHandle<TriggerObjectMatch> handleTriggerObjectMatch(
          iEvent.put(std::move(triggerObjectMatch), labelTriggerObjectMatcher));
      // set product reference to trigger match association
      if (!handleTriggerObjectMatch.isValid()) {
        LogError("triggerMatchValid") << "pat::TriggerObjectMatch product with InputTag '" << labelTriggerObjectMatcher
                                      << "' not in event";
        continue;
      }
      if (!(triggerEvent->addObjectMatchResult(handleTriggerObjectMatch, labelTriggerObjectMatcher))) {
        LogWarning("triggerObjectMatchReplication")
            << "pat::TriggerEvent contains already a pat::TriggerObjectMatch from matcher module '"
            << labelTriggerObjectMatcher << "'";
      }
    }
  }

  iEvent.put(std::move(triggerEvent));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATTriggerEventProducer);
