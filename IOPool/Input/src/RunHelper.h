#ifndef IOPool_Input_RunHelper_h
#define IOPool_Input_RunHelper_h

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputSource.h"

#include <memory>
#include <vector>

namespace edm {
  class IndexIntoFile;
  class ParameterSetDescription;

  class RunHelperBase {
  public: 
    explicit RunHelperBase() = default;
    virtual ~RunHelperBase();

    virtual InputSource::ItemType nextItemType(
      InputSource::ItemType const& previousItemType,
      InputSource::ItemType const& newItemType) {return newItemType;}
    virtual RunNumber_t runNumberToUseForThisLumi() const {return 0;}
    virtual bool fakeNewRun() const {return false;} 
    virtual void setForcedRunOffset(RunNumber_t firstRun) {}
    virtual void checkForNewRun(RunNumber_t run) {}

    virtual void checkRunConsistency(RunNumber_t run, RunNumber_t origninalRun) const;
    virtual void checkLumiConsistency(LuminosityBlockNumber_t lumi, LuminosityBlockNumber_t origninalLumi) const; 
    virtual void overrideRunNumber(EventID& event, bool isRealData) {}
    virtual void overrideRunNumber(RunID& run) {}
    virtual void overrideRunNumber(LuminosityBlockID& lumi) {}

    static void fillDescription(ParameterSetDescription& desc);
  };

  class DefaultRunHelper : public RunHelperBase {
  public:
    explicit DefaultRunHelper() = default;
    virtual ~DefaultRunHelper();
  };

  class SetRunHelper : public RunHelperBase {
  public: 
    explicit SetRunHelper(ParameterSet const& pset);
    virtual ~SetRunHelper();

    virtual void setForcedRunOffset(RunNumber_t firstRun) override;

    virtual void checkRunConsistency(RunNumber_t run, RunNumber_t origninalRun) const override;
    virtual void overrideRunNumber(EventID& event, bool isRealData) override;
    virtual void overrideRunNumber(RunID& run) override;
    virtual void overrideRunNumber(LuminosityBlockID& lumi) override;

  private:
    RunNumber_t setRun_;
    int forcedRunOffset_;
    bool firstTime_;
  };

  class SetRunForEachLumiHelper : public RunHelperBase {
  public: 
    explicit SetRunForEachLumiHelper(ParameterSet const& pset);
    virtual ~SetRunForEachLumiHelper();

    virtual InputSource::ItemType nextItemType(
      InputSource::ItemType const& previousItemType,
      InputSource::ItemType const& newIemType) override;
    virtual RunNumber_t runNumberToUseForThisLumi() const override;
    virtual bool fakeNewRun() const override {return fakeNewRun_;}
    virtual void checkForNewRun(RunNumber_t run) override;

    virtual void checkRunConsistency(RunNumber_t run, RunNumber_t origninalRun) const override; 
    virtual void overrideRunNumber(EventID& event, bool isRealData) override;
    virtual void overrideRunNumber(RunID& run) override;
    virtual void overrideRunNumber(LuminosityBlockID& lumi) override;

  private:
    std::vector<RunNumber_t> setRunNumberForEachLumi_;
    size_t indexOfNextRunNumber_;
    RunNumber_t realRunNumber_;
    bool fakeNewRun_;
    bool firstTime_;
  };

  std::unique_ptr<RunHelperBase>
  makeRunHelper(ParameterSet const& pset);
}

#endif
