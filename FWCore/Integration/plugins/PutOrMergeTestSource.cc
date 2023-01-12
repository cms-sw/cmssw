//
//  PutOrMergeTestSource.cc
//  CMSSW
//
//  Created by Chris Jones on 3/23/21.
//

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/RunPrincipal.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"

#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/ThingWithMerge.h"
#include "DataFormats/TestObjects/interface/ThingWithIsEqual.h"

#include <cassert>
using namespace edm;

namespace edmtest {
  class PutOrMergeTestSource : public InputSource {
  public:
    PutOrMergeTestSource(ParameterSet const&, InputSourceDescription const&);

    /// Register any produced products
    void registerProducts() final;

  private:
    ItemType getNextItemType() final;
    std::shared_ptr<RunAuxiliary> readRunAuxiliary_() final;
    std::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() final;
    std::shared_ptr<FileBlock> readFile_() final;
    void readRun_(RunPrincipal& runPrincipal) final;
    void readEvent_(EventPrincipal& eventPrincipal) final;

    int stage_;
    ParameterSet const dummyPSet_;
    BranchDescription thingDesc_;
    BranchDescription thingWithMergeDesc_;
    BranchDescription thingWithEqualDesc_;
    ProcessHistoryID historyID_;
  };
}  // namespace edmtest

using namespace edmtest;

PutOrMergeTestSource::PutOrMergeTestSource(ParameterSet const& iPS, InputSourceDescription const& iISD)
    : InputSource(iPS, iISD),
      stage_(0),
      dummyPSet_([]() {
        ParameterSet dummy;
        dummy.registerIt();
        return dummy;
      }()),
      thingDesc_(InRun,
                 "thingWithMergeProducer",
                 "PROD",
                 "edmtest::Thing",
                 "edmtestThing",
                 "endRun",
                 "PutOrMergeTestSource",
                 dummyPSet_.id(),
                 edm::TypeWithDict(typeid(edmtest::Thing)),
                 false,
                 true),
      thingWithMergeDesc_(InRun,
                          "thingWithMergeProducer",
                          "PROD",
                          "edmtest::ThingWithMerge",
                          "edmtestThingWithMerge",
                          "endRun",
                          "PutOrMergeTestSource",
                          dummyPSet_.id(),
                          edm::TypeWithDict(typeid(edmtest::ThingWithMerge)),
                          false,
                          true),
      thingWithEqualDesc_(InRun,
                          "thingWithMergeProducer",
                          "PROD",
                          "edmtest::ThingWithIsEqual",
                          "edmtestThingWithIsEqual",
                          "endRun",
                          "PutOrMergeTestSource",
                          dummyPSet_.id(),
                          edm::TypeWithDict(typeid(edmtest::ThingWithIsEqual)),
                          false,
                          true) {
  edm::ParameterSet dummyPset;
  dummyPset.registerIt();

  ProcessHistory history;
  history.emplace_back("PROD", dummyPset.id(), "RELVERSION", "PASSID");
  processHistoryRegistry().registerProcessHistory(history);
  historyID_ = history.id();
}

void PutOrMergeTestSource::registerProducts() {
  edm::ParameterSet dummyPset;
  dummyPset.registerIt();

  thingDesc_.setIsProvenanceSetOnRead();
  thingWithMergeDesc_.setIsProvenanceSetOnRead();
  productRegistryUpdate().copyProduct(thingDesc_);
  productRegistryUpdate().copyProduct(thingWithMergeDesc_);
  productRegistryUpdate().copyProduct(thingWithEqualDesc_);
}

InputSource::ItemType PutOrMergeTestSource::getNextItemType() {
  switch (stage_) {
    case 0: {
      return IsFile;
    }
    case 1: {
      return IsRun;
    }
    case 2: {
      return IsRun;
    }
    default:
      return IsStop;
  }
  return IsInvalid;
}

std::shared_ptr<RunAuxiliary> PutOrMergeTestSource::readRunAuxiliary_() {
  auto id = std::make_shared<RunAuxiliary>(1, Timestamp::beginOfTime(), Timestamp::endOfTime());
  id->setProcessHistoryID(historyID_);
  return id;
}
std::shared_ptr<LuminosityBlockAuxiliary> PutOrMergeTestSource::readLuminosityBlockAuxiliary_() { return {}; }
std::shared_ptr<FileBlock> PutOrMergeTestSource::readFile_() {
  ++stage_;
  return std::make_shared<FileBlock>();
}
void PutOrMergeTestSource::readRun_(RunPrincipal& runPrincipal) {
  runAuxiliary()->setProcessHistoryID(historyID_);
  runPrincipal.fillRunPrincipal(processHistoryRegistry());
  ++stage_;
  runPrincipal.putOrMerge(thingDesc_, std::make_unique<Wrapper<edmtest::Thing>>(WrapperBase::Emplace{}, 100001));
  runPrincipal.putOrMerge(thingWithMergeDesc_,
                          std::make_unique<Wrapper<edmtest::ThingWithMerge>>(WrapperBase::Emplace{}, 100002));
  runPrincipal.putOrMerge(thingWithEqualDesc_,
                          std::make_unique<Wrapper<edmtest::ThingWithIsEqual>>(WrapperBase::Emplace{}, 100003));
}
void PutOrMergeTestSource::readEvent_(EventPrincipal& eventPrincipal) { assert(false); }

DEFINE_FWK_INPUT_SOURCE(PutOrMergeTestSource);
