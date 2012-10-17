// $Id: DQMHttpSource.h,v 1.13 2011/04/04 16:05:37 mommsen Exp $
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
    
    $Author: mommsen $
    $Revision: 1.13 $
    $Date: 2011/04/04 16:05:37 $
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
    virtual EventPrincipal* read();
    void initializeDQMStore();

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
