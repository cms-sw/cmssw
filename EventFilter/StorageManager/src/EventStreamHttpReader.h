// $Id: EventStreamHttpReader.h,v 1.30 2012/10/31 17:09:27 wmtan Exp $
/// @file: EventStreamHttpReader.h

#ifndef StorageManager_EventStreamHttpReader_h
#define StorageManager_EventStreamHttpReader_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/EventServerProxy.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "IOPool/Streamer/interface/StreamerInputSource.h"


namespace edm
{
  /**
    Input source for event consumers that will get events from the
    Storage Manager Event Server. This does uses a HTTP get using the
    CURL library. The Storage Manager Event Server responses with
    a binary octet-stream.  The product registry is also obtained
    through a HTTP get.

    There is currently no test of the product registry against
    the consumer client product registry within the code. It should
    already be done if this was inherenting from the standard
    framework input source. Currently we inherit from InputSource.

    $Author: wmtan $
    $Revision: 1.30 $
    $Date: 2012/10/31 17:09:27 $
  */

  class EventStreamHttpReader : public edm::StreamerInputSource
  {
  public:
    EventStreamHttpReader
    (
      edm::ParameterSet const&,
      edm::InputSourceDescription const&
    );
    virtual ~EventStreamHttpReader() {};

  private:
    virtual bool checkNextEvent();

    void readHeader();
    void initializeDQMStore();
    
    stor::EventServerProxy<stor::EventConsumerRegistrationInfo> eventServerProxy_;
    DQMStore* dqmStore_;
    bool dqmStoreAvailabiltyChecked_;

    const bool dropOldLumisectionEvents_;
    const std::string consumerName_;
    unsigned int totalDroppedEvents_;
    unsigned int lastLS_;

  };

} // namespace edm

#endif // StorageManager_EventStreamHttpReader_h

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
