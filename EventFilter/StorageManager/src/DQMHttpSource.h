// $Id: DQMHttpSource.h,v 1.14 2012/10/17 02:03:00 wmtan Exp $
/// @file: DQMHttpSource.h

#ifndef StorageManager_DQMHttpSource_h
#define StorageManager_DQMHttpSource_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "EventFilter/StorageManager/interface/DQMEventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/EventServerProxy.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Sources/interface/RawInputSource.h"
#include "IOPool/Streamer/interface/DQMEventMessage.h"

#include <boost/thread/mutex.hpp>

#include <memory>


namespace edm
{
  /**
    An input source for DQM consumers using cmsRun that connect to
    the StorageManager or SMProxyServer to get DQM (histogram) data.
    
    $Author: wmtan $
    $Revision: 1.14 $
    $Date: 2012/10/17 02:03:00 $
  */

  class DQMHttpSource : public edm::RawInputSource
  {
  public:
    DQMHttpSource
    (
      const edm::ParameterSet&, 
      const edm::InputSourceDescription&
    );
    virtual ~DQMHttpSource() {};

    static void addEventToDQMBackend
    (
      DQMStore*,
      const DQMEventMsgView&,
      const bool overwrite
    );


  private:
    EventAuxiliary const& eventAuxiliary() const {
      return *eventAuxiliary_;
    }
    void setEventAuxiliary(std::unique_ptr<EventAuxiliary> aux) {
      eventAuxiliary_ = std::move(aux);
    }
    virtual EventPrincipal* read(EventPrincipal& eventPrincipal);
    virtual bool checkNextEvent();
    void initializeDQMStore();

    std::unique_ptr<EventAuxiliary> eventAuxiliary_;
    stor::EventServerProxy<stor::DQMEventConsumerRegistrationInfo> dqmEventServerProxy_;
    DQMStore* dqmStore_;

    static boost::mutex mutex_;
  };

} // namespace edm

#endif // StorageManager_DQMHttpSource_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
