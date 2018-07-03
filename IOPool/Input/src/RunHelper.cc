/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "RunHelper.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <cassert>

namespace edm {

  std::unique_ptr<RunHelperBase>
  makeRunHelper(ParameterSet const& pset) {
    if(pset.exists("setRunNumber")) {
      RunNumber_t run = pset.getUntrackedParameter<unsigned int>("setRunNumber");
      if(run != 0U) {
        return std::make_unique<SetRunHelper>(pset);
      }
    } 
    if(pset.exists("setRunNumberForEachLumi")) {
      std::vector<RunNumber_t> runs = pset.getUntrackedParameter<std::vector<unsigned int> >("setRunNumberForEachLumi");
      if(!runs.empty()) {
        return std::make_unique<SetRunForEachLumiHelper>(pset);
      }
    }
    return std::make_unique<DefaultRunHelper>();
  }

  RunHelperBase::~RunHelperBase() {}

  void
  RunHelperBase::checkLumiConsistency(LuminosityBlockNumber_t lumi, LuminosityBlockNumber_t originalLumi) const {
    assert(lumi == originalLumi);
  }

  void
  RunHelperBase::checkRunConsistency(RunNumber_t run, RunNumber_t originalRun) const {
    assert(run == originalRun);
  }

  DefaultRunHelper::~DefaultRunHelper() {}

  SetRunHelper::SetRunHelper(ParameterSet const& pset) :
        RunHelperBase(),
        setRun_(pset.getUntrackedParameter<unsigned int>("setRunNumber")),
        forcedRunOffset_(0),
        firstTime_(true) {
  }

  SetRunHelper::~SetRunHelper() {}
 
  void
  SetRunHelper::setForcedRunOffset(RunNumber_t firstRun) {
    if(firstTime_ && setRun_ != 0) {
      forcedRunOffset_  = setRun_ - firstRun;
      if(forcedRunOffset_ < 0) {
        throw Exception(errors::Configuration)
          << "The value of the 'setRunNumber' parameter must not be\n"
          << "less than the first run number in the first input file.\n"
          << "'setRunNumber' was " << setRun_ <<", while the first run was "
          << firstRun << ".\n";
      }
    }
    firstTime_ = false;
  }

  void
  SetRunHelper::overrideRunNumber(RunID& id) {
    id = RunID(id.run() + forcedRunOffset_);
    if(id < RunID::firstValidRun()) id = RunID::firstValidRun();
  }

  void
  SetRunHelper::overrideRunNumber(LuminosityBlockID& id) {
    id = LuminosityBlockID(id.run() + forcedRunOffset_, id.luminosityBlock());
    if(RunID(id.run()) < RunID::firstValidRun()) id = LuminosityBlockID(RunID::firstValidRun().run(), id.luminosityBlock());
  }

  void
  SetRunHelper::overrideRunNumber(EventID& id, bool isRealData) {
    if(isRealData) {
      throw Exception(errors::Configuration, "SetRunHelper::overrideRunNumber()")
        << "The 'setRunNumber' parameter of PoolSource cannot be used with real data.\n";
    }
    id = EventID(id.run() + forcedRunOffset_, id.luminosityBlock(), id.event());
    if(RunID(id.run()) < RunID::firstValidRun()) {
      id = EventID(RunID::firstValidRun().run(), LuminosityBlockID::firstValidLuminosityBlock().luminosityBlock(), id.event());
    }
  }

  void
  SetRunHelper::checkRunConsistency(RunNumber_t run, RunNumber_t originalRun) const {
    assert(run == originalRun + forcedRunOffset_);
  }

  SetRunForEachLumiHelper::SetRunForEachLumiHelper(ParameterSet const& pset) :
        RunHelperBase(),
        setRunNumberForEachLumi_(pset.getUntrackedParameter<std::vector<unsigned int> >("setRunNumberForEachLumi")),
        indexOfNextRunNumber_(0),
        realRunNumber_(0),
        fakeNewRun_(false),
        firstTime_(true) {
  }

  SetRunForEachLumiHelper::~SetRunForEachLumiHelper() {}
 
  InputSource::ItemType
  SetRunForEachLumiHelper::nextItemType(InputSource::ItemType const& previousItemType, InputSource::ItemType const& newItemType) {
    if(newItemType == InputSource::IsRun || (newItemType == InputSource::IsLumi && previousItemType != InputSource::IsRun)) {
      if(firstTime_) {
        firstTime_ = false;
      } else {
        ++indexOfNextRunNumber_;
      }
      if(indexOfNextRunNumber_ == setRunNumberForEachLumi_.size()) {
        throw Exception(errors::MismatchedInputFiles, "PoolSource::getNextItemType")
          << " Parameter 'setRunNumberForEachLumi' has " << setRunNumberForEachLumi_.size() << " entries\n"
          << "but this job is processing more luminosity blocks than this.\n";
      }
      RunNumber_t run = setRunNumberForEachLumi_[indexOfNextRunNumber_];
      if(run == 0) {
        throw Exception(errors::Configuration, "PoolSource") <<
          "'setRunNumberForEachLumi' contains an illegal run number of '0'.\n";
      }
      bool sameRunNumber = (indexOfNextRunNumber_ != 0U && run == setRunNumberForEachLumi_[indexOfNextRunNumber_ - 1]);
      if(!sameRunNumber) {
        fakeNewRun_ = (newItemType != InputSource::IsRun);
        return InputSource::IsRun;
      }
    }
    return newItemType;
  }

  RunNumber_t
  SetRunForEachLumiHelper::runNumberToUseForThisLumi() const {
    return setRunNumberForEachLumi_.at(indexOfNextRunNumber_);
  }

  void
  SetRunForEachLumiHelper::checkForNewRun(RunNumber_t run) {
    if(realRunNumber_ != 0 && run != realRunNumber_) {
      throw Exception(errors::MismatchedInputFiles, "PoolSource::checkForNewRun")
        << " Parameter 'setRunNumberForEachLumi' can only process a single run.\n"
        << "but this job is processing more than one run.\n";
    }
    realRunNumber_ = run;
  }

  void
  SetRunForEachLumiHelper::overrideRunNumber(RunID& id) {
    id = RunID(runNumberToUseForThisLumi());
  }

  void
  SetRunForEachLumiHelper::overrideRunNumber(LuminosityBlockID& id) {
    id = LuminosityBlockID(runNumberToUseForThisLumi(), id.luminosityBlock());
  }

  void
  SetRunForEachLumiHelper::overrideRunNumber(EventID& id, bool isRealData) {
    if(isRealData) {
      throw Exception(errors::Configuration, "SetRunForEachLumiHelper::overrideRunNumber()")
        << "The 'setRunNumberForEachLumi' parameter of PoolSource cannot be used with real data.\n";
    }
    id = EventID(runNumberToUseForThisLumi(), id.luminosityBlock(), id.event());
  }

  void
  SetRunForEachLumiHelper::checkRunConsistency(RunNumber_t run, RunNumber_t originalRun) const {
    assert(run == runNumberToUseForThisLumi());
  }

  void
  RunHelperBase::fillDescription(ParameterSetDescription& desc) {
    desc.addOptionalNode(ParameterDescription<unsigned int>("setRunNumber", 0U, false) xor
         ParameterDescription<std::vector<unsigned int> >("setRunNumberForEachLumi", std::vector<unsigned int>(), false), true)
         ->setComment("If 'setRun' is non-zero, change number of first run to this number. Apply same offset to all runs." \
         "If 'setRunNumberForEachLumi' is non-empty, use these as run numbers for each lumi respectively." \
         "''setRun' and 'setRunNumberForEachLumi' are mutually exclusive and allowed only for simulation.");
  }
}
