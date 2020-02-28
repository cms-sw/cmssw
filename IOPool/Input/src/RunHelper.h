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

    virtual InputSource::ItemType nextItemType(InputSource::ItemType const& previousItemType,
                                               InputSource::ItemType const& newItemType,
                                               RunNumber_t,
                                               LuminosityBlockNumber_t,
                                               EventNumber_t) {
      return newItemType;
    }
    virtual RunNumber_t runNumberToUseForThisLumi() const { return 0; }
    virtual bool fakeNewRun() const { return false; }
    virtual void setForcedRunOffset(RunNumber_t firstRun) {}
    virtual void checkForNewRun(RunNumber_t run, LuminosityBlockNumber_t nextLumi) {}

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
    ~DefaultRunHelper() override;
  };

  class SetRunHelper : public RunHelperBase {
  public:
    explicit SetRunHelper(ParameterSet const& pset);
    ~SetRunHelper() override;

    void setForcedRunOffset(RunNumber_t firstRun) override;

    void checkRunConsistency(RunNumber_t run, RunNumber_t origninalRun) const override;
    void overrideRunNumber(EventID& event, bool isRealData) override;
    void overrideRunNumber(RunID& run) override;
    void overrideRunNumber(LuminosityBlockID& lumi) override;

  private:
    RunNumber_t setRun_;
    int forcedRunOffset_;
    bool firstTime_;
  };

  class SetRunForEachLumiHelper : public RunHelperBase {
  public:
    explicit SetRunForEachLumiHelper(ParameterSet const& pset);
    ~SetRunForEachLumiHelper() override;

    InputSource::ItemType nextItemType(InputSource::ItemType const& previousItemType,
                                       InputSource::ItemType const& newIemType,
                                       RunNumber_t,
                                       LuminosityBlockNumber_t,
                                       EventNumber_t) override;
    RunNumber_t runNumberToUseForThisLumi() const override;
    bool fakeNewRun() const override { return fakeNewRun_; }
    void checkForNewRun(RunNumber_t run, LuminosityBlockNumber_t nextLumi) override;

    void checkRunConsistency(RunNumber_t run, RunNumber_t origninalRun) const override;
    void overrideRunNumber(EventID& event, bool isRealData) override;
    void overrideRunNumber(RunID& run) override;
    void overrideRunNumber(LuminosityBlockID& lumi) override;

  private:
    std::vector<RunNumber_t> setRunNumberForEachLumi_;
    size_t indexOfNextRunNumber_;
    RunNumber_t realRunNumber_;
    bool fakeNewRun_;
    bool firstTime_;
  };

  class FirstLuminosityBlockForEachRunHelper : public RunHelperBase {
  public:
    explicit FirstLuminosityBlockForEachRunHelper(ParameterSet const& pset);

    InputSource::ItemType nextItemType(InputSource::ItemType const& previousItemType,
                                       InputSource::ItemType const& newIemType,
                                       RunNumber_t,
                                       LuminosityBlockNumber_t,
                                       EventNumber_t) override;
    RunNumber_t runNumberToUseForThisLumi() const override;
    bool fakeNewRun() const override { return fakeNewRun_; }
    void checkForNewRun(RunNumber_t run, LuminosityBlockNumber_t nextLumi) override;

    void checkRunConsistency(RunNumber_t run, RunNumber_t originalRun) const override;
    void overrideRunNumber(EventID& event, bool isRealData) override;
    void overrideRunNumber(RunID& run) override;
    void overrideRunNumber(LuminosityBlockID& lumi) override;

  private:
    RunNumber_t findRunFromLumi(LuminosityBlockNumber_t) const;
    std::vector<edm::LuminosityBlockID> const lumiToRun_;
    RunNumber_t realRunNumber_;
    RunNumber_t lastUsedRunNumber_;
    bool fakeNewRun_;
  };
  std::unique_ptr<RunHelperBase> makeRunHelper(ParameterSet const& pset);
}  // namespace edm

#endif
