#ifndef Framework_Sources_IDGeneratorSourceBase_h
#define Framework_Sources_IDGeneratorSourceBase_h

/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"

#include <memory>
#include <vector>

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
  template <typename BASE>
  class IDGeneratorSourceBase : public BASE {
  public:
    explicit IDGeneratorSourceBase(ParameterSet const& pset, InputSourceDescription const& desc, bool realData);
    ~IDGeneratorSourceBase() noexcept(false) override;

    unsigned int numberEventsInRun() const { return numberEventsInRun_; }
    unsigned int numberEventsInLumi() const { return numberEventsInLumi_; }
    TimeValue_t presentTime() const { return presentTime_; }
    unsigned int timeBetweenEvents() const { return timeBetweenEvents_; }
    unsigned int eventCreationDelay() const { return eventCreationDelay_; }
    unsigned int numberEventsInThisRun() const { return numberEventsInThisRun_; }
    unsigned int numberEventsInThisLumi() const { return numberEventsInThisLumi_; }
    EventID const& eventID() const { return eventID_; }
    RunNumber_t run() const { return eventID_.run(); }
    EventNumber_t event() const { return eventID_.event(); }
    LuminosityBlockNumber_t luminosityBlock() const { return eventID_.luminosityBlock(); }

    static void fillDescription(ParameterSetDescription& desc);

  protected:
    template <typename F>
    void doReadEvent(EventPrincipal& eventPrincipal, F&& f) {
      assert(BASE::eventCached() || BASE::processingMode() != BASE::RunsLumisAndEvents);
      EventAuxiliary aux(eventID_, BASE::processGUID(), Timestamp(presentTime_), isRealData_, eType_);
      auto history = BASE::processHistoryRegistry().getMapped(aux.processHistoryID());
      eventPrincipal.fillEventPrincipal(aux, history);
      f(eventPrincipal);
      BASE::resetEventCached();
    }

  private:
    typename BASE::ItemType getNextItemType() final;
    virtual void initialize(EventID& id, TimeValue_t& time, TimeValue_t& interval);
    virtual bool setRunAndEventInfo(EventID& id, TimeValue_t& time, EventAuxiliary::ExperimentType& etype) = 0;
    virtual bool noFiles() const;
    virtual size_t fileIndex() const;
    void beginJob() override;

    std::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() override;
    std::shared_ptr<RunAuxiliary> readRunAuxiliary_() override;
    void skip(int offset) override;
    void rewind_() override;

    void advanceToNext(EventID& eventID, TimeValue_t& time);
    void retreatToPrevious(EventID& eventID, TimeValue_t& time);
    RunNumber_t runForLumi(LuminosityBlockNumber_t) const;

    std::vector<edm::LuminosityBlockID> firstLumiForRuns_;
    unsigned int numberEventsInRun_;
    unsigned int numberEventsInLumi_;
    TimeValue_t presentTime_;
    TimeValue_t origTime_;
    TimeValue_t timeBetweenEvents_;
    unsigned int eventCreationDelay_; /* microseconds */

    unsigned int numberEventsInThisRun_;
    unsigned int numberEventsInThisLumi_;
    EventNumber_t const zerothEvent_;
    EventID eventID_;
    EventID origEventID_;
    bool isRealData_;
    EventAuxiliary::ExperimentType eType_;
  };
}  // namespace edm
#endif
