#ifndef Framework_Sources_ProducerSourceBase_h
#define Framework_Sources_ProducerSourceBase_h

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
  class ProducerSourceBase : public InputSource {
  public:
    explicit ProducerSourceBase(ParameterSet const& pset, InputSourceDescription const& desc, bool realData);
    virtual ~ProducerSourceBase();

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

  private:
    virtual ItemType getNextItemType();
    virtual bool setRunAndEventInfo(EventID& id, TimeValue_t& time) = 0;
    virtual void produce(Event& e) = 0;
    virtual void beginRun(Run&);
    virtual void endRun(Run&);
    virtual void beginLuminosityBlock(LuminosityBlock&);
    virtual void endLuminosityBlock(LuminosityBlock&);
    virtual EventPrincipal* readEvent_(EventPrincipal& eventPrincipal);
    virtual boost::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    virtual boost::shared_ptr<RunAuxiliary> readRunAuxiliary_();
    virtual void skip(int offset);
    virtual void rewind_();

    void advanceToNext(EventID& eventID, TimeValue_t& time);
    void retreatToPrevious(EventID& eventID, TimeValue_t& time);

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
    bool isRealData_;
    EventAuxiliary::ExperimentType eType_;
  };
}
#endif
