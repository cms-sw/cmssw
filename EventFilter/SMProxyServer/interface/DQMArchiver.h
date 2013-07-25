// $Id: DQMArchiver.h,v 1.2 2011/03/07 15:41:54 mommsen Exp $
/// @file: DQMArchiver.h 

#ifndef EventFilter_SMProxyServer_DQMArchiver_h
#define EventFilter_SMProxyServer_DQMArchiver_h

#include "EventFilter/SMProxyServer/interface/Configuration.h"
#include "EventFilter/SMProxyServer/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/ConsumerID.h"
#include "EventFilter/StorageManager/interface/DQMEventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/DQMEventQueueCollection.h"
#include "IOPool/Streamer/interface/DQMEventMessage.h"

#include <boost/scoped_ptr.hpp>
#include <boost/thread/thread.hpp>

#include <map>


namespace smproxy {

  /**
   * Archive DQM histograms
   *
   * $Author: mommsen $
   * $Revision: 1.2 $
   * $Date: 2011/03/07 15:41:54 $
   */
  
  class DQMArchiver
  {
  public:

    DQMArchiver(StateMachine*);

    ~DQMArchiver();

    /**
     * Return the registration pointer
     */
    const stor::DQMEventConsRegPtr& getRegPtr() const
    { return regPtr_; }
    
    
  private:

    void activity();
    void doIt();
    void handleDQMEvent(const stor::DQMTopLevelFolder::Record&);
    void updateLastRecord(const stor::DQMTopLevelFolder::Record&);
    void writeDQMEventToFile(const DQMEventMsgView&, const bool endRun) const;
    void createRegistration();

    StateMachine* stateMachine_;
    const DQMArchivingParams dqmArchivingParams_;
    stor::DQMEventQueueCollectionPtr dqmEventQueueCollection_;
    stor::DQMEventConsRegPtr regPtr_;

    boost::scoped_ptr<boost::thread> thread_;

    typedef std::map<std::string,stor::DQMTopLevelFolder::Record> Records;
    Records lastUpdateForFolders_;
  };
  
} // namespace smproxy

#endif // EventFilter_SMProxyServer_DQMArchiver_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
