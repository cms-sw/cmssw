#ifndef Framework_ConfigurableInputSource_h
#define Framework_ConfigurableInputSource_h

/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/RunID.h"

#include "boost/shared_ptr.hpp"

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
  class ConfigurableInputSource : public InputSource {
  public:
    explicit ConfigurableInputSource(ParameterSet const& pset, InputSourceDescription const& desc, bool realData = true);
    virtual ~ConfigurableInputSource();

    unsigned int numberEventsInRun() const {return numberEventsInRun_;} 
    unsigned int numberEventsInLumi() const {return numberEventsInLumi_;} 
    TimeValue_t presentTime() const {return presentTime_;}
    unsigned int timeBetweenEvents() const {return timeBetweenEvents_;}
    unsigned int eventCreationDelay() const {return eventCreationDelay_;}
    unsigned int numberEventsInThisRun() const {return numberEventsInThisRun_;}
    unsigned int numberEventsInThisLumi() const {return numberEventsInThisLumi_;}
    RunNumber_t run() const {return eventID_.run();}
    EventNumber_t event() const {return eventID_.event();}
    LuminosityBlockNumber_t luminosityBlock() const {return eventID_.luminosityBlock();}
    static void fillDescription(ParameterSetDescription& desc);

  protected:

    void setEventNumber(EventNumber_t e) {
      RunNumber_t r = run();
      LuminosityBlockNumber_t lb = luminosityBlock();
      eventID_ = EventID(r, lb, e);
      eventSet_ = true;
    } 
    void setTime(TimeValue_t t) {presentTime_ = t;}
    void reallyReadEvent();

  private:
    virtual ItemType getNextItemType();
    virtual void setRunAndEventInfo();
    virtual bool produce(Event& e) = 0;
    virtual void beginRun(Run&);
    virtual void endRun(Run&);
    virtual void beginLuminosityBlock(LuminosityBlock&);
    virtual void endLuminosityBlock(LuminosityBlock&);
    virtual EventPrincipal* readEvent_(EventPrincipal& eventPrincipal);
    virtual boost::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    virtual boost::shared_ptr<RunAuxiliary> readRunAuxiliary_();
    virtual void skip(int offset);
    virtual void setRun(RunNumber_t r);
    virtual void setLumi(LuminosityBlockNumber_t lb);
    virtual void rewind_();

    virtual void postForkReacquireResources(boost::shared_ptr<edm::multicore::MessageReceiverForSource>);
    void advanceToNext() ;
    void retreatToPrevious();

    unsigned int numberEventsInRun_;
    unsigned int numberEventsInLumi_;
    TimeValue_t presentTime_;
    TimeValue_t origTime_;
    TimeValue_t timeBetweenEvents_;
    unsigned int eventCreationDelay_;  /* microseconds */

    unsigned int numberEventsInThisRun_;
    unsigned int numberEventsInThisLumi_;
    unsigned int const zerothEvent_;
    EventID eventID_;
    EventID origEventID_;
    bool lumiSet_;
    bool eventSet_;
    bool isRealData_;
    EventAuxiliary::ExperimentType eType_;
     
    //used when process has been forked
    boost::shared_ptr<edm::multicore::MessageReceiverForSource> receiver_;
    unsigned int numberOfEventsBeforeBigSkip_;
  };
}
#endif
