#ifndef DaqSource_DaqSource_H
#define DaqSource_DaqSource_H

/** \class DaqSource
 *  An input service for raw data. 
 *  The actual source can be the real DAQ, a file, a random generator, etc.
 *
 *  $Date: 2012/11/29 01:34:39 $
 *  $Revision: 1.25 $
 *  \author N. Amapane - S. Argiro'
 */

#include <memory>
#include "boost/shared_ptr.hpp"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/DaqProvenanceHelper.h"
#include "EventFilter/Utilities/interface/ModuleWeb.h"

#include "xdata/UnsignedInteger32.h"
#include "xdata/Boolean.h"

#include <pthread.h>

class DaqBaseReader;
class FEDRawDataCollection;

namespace edm {
  class ParameterSet;
  class Timestamp;
  class InputSourceDescription;
  class EventPrincipal;
  class LuminosityBlockAuxiliary;


  class DaqSource : public InputSource, private evf::ModuleWeb {

   public:
    explicit DaqSource(const ParameterSet& pset, 
  		     const InputSourceDescription& desc);
  
    virtual ~DaqSource();
    
   private:

    void defaultWebPage(xgi::Input *in, xgi::Output *out); 
    virtual EventPrincipal* readEvent_(EventPrincipal& eventPrincipal);
    virtual boost::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    virtual boost::shared_ptr<RunAuxiliary> readRunAuxiliary_();
    virtual EventPrincipal* readIt(EventID const& eventID);
    virtual void skip(int offset);
    virtual void setLumi(LuminosityBlockNumber_t lb);
    virtual void setRun(RunNumber_t r);
    //virtual void doBeginRun(edm::EventPrincipal &rp);
    virtual ItemType getNextItemType();
  

    int doMyBeginRun();
    virtual void publish(xdata::InfoSpace *);
    virtual void publishToXmas(xdata::InfoSpace *);
    virtual void publishForkInfo(evf::moduleweb::ForkInfoObj * forkInfoObj);
    virtual void openBackDoor(unsigned int,bool*);
    virtual void closeBackDoor();
    virtual void signalWaitingThreadAndBlock();

    DaqBaseReader*  reader_;
    unsigned int    lumiSegmentSizeInEvents_; //temporary kludge, LS# will come from L1 Global record
    unsigned int    lumiSegmentSizeInSeconds_; //temporary kludge, LS# will come from L1 Global record
    bool            useEventCounter_;
    bool            useTimer_;
    unsigned int    eventCounter_;
    bool            keepUsingPsidFromTrigger_;
    bool            fakeLSid_;
  
    RunNumber_t runNumber_;
    LuminosityBlockNumber_t luminosityBlockNumber_;
    DaqProvenanceHelper daqProvenanceHelper_;
    ProcessHistoryID phid_;
    bool noMoreEvents_;
    bool alignLsToLast_;
    
    pthread_mutex_t mutex_;
    pthread_mutex_t signal_lock_;
    pthread_cond_t cond_;
    xdata::UnsignedInteger32        *lumiSectionIndex_;
    xdata::UnsignedInteger32        *prescaleSetIndex_;
    xdata::UnsignedInteger32        *lastLumiPrescaleIndex_;
    xdata::UnsignedInteger32        *lastLumiUsingEol_;
    xdata::Boolean                  *lsTimedOut_;
    xdata::Boolean                  *lsToBeRecovered_;
    xdata::InfoSpace                *is_;
    xdata::InfoSpace                *mis_;
    int                              count;
    unsigned int                     thisEventLSid;
    bool                             goToStopping;
    struct timeval                   startOfLastLumi; 
    bool                             immediateStop;
    evf::moduleweb::ForkInfoObj      *forkInfo_;
    bool                             runFork_;
    timeval                          tvStat_;
    bool                             beginRunTiming_;
    int                              bunchCrossing_;
    int                              orbitNumber_;
    EventAuxiliary::ExperimentType   evttype_; 
    EventID                          eventID_;
    FEDRawDataCollection*            fedCollection_;
  };
  
}
  
#endif
