// $Id: DataManager.cc,v 1.3 2011/03/24 17:26:25 mommsen Exp $
/// @file: DataManager.cc

#include "EventFilter/SMProxyServer/interface/DataManager.h"
#include "EventFilter/SMProxyServer/interface/DQMArchiver.h"
#include "EventFilter/SMProxyServer/interface/Exception.h"
#include "EventFilter/SMProxyServer/interface/StateMachine.h"
#include "EventFilter/SMProxyServer/src/EventRetriever.icc"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"

#include <boost/foreach.hpp>
#include <boost/pointer_cast.hpp>


namespace smproxy
{
  DataManager::DataManager
  (
    StateMachine* stateMachine
  ) :
  stateMachine_(stateMachine),
  registrationQueue_(stateMachine->getRegistrationQueue())
  {
    watchDogThread_.reset(
      new boost::thread( boost::bind( &DataManager::checkForStaleConsumers, this) )
    );
  }

  DataManager::~DataManager()
  {
    stop();
    watchDogThread_->interrupt();
    watchDogThread_->join();
  }
  
  
  void DataManager::start(DataRetrieverParams const& drp)
  {
    dataRetrieverParams_ = drp;
    dataEventRetrievers_.clear();
    dqmEventRetrievers_.clear();
    edm::shutdown_flag = false;
    thread_.reset(
      new boost::thread( boost::bind( &DataManager::doIt, this) )
    );
  }
  
  
  void DataManager::stop()
  {
    // enqueue a dummy RegistrationInfoBase to tell the thread to stop
    registrationQueue_->enqWait( stor::RegPtr() );
    thread_->join();

    edm::shutdown_flag = true;

    BOOST_FOREACH(
      const DataEventRetrieverMap::value_type& pair,
      dataEventRetrievers_
    ) pair.second->stop();

    BOOST_FOREACH(
      const DQMEventRetrieverMap::value_type& pair,
      dqmEventRetrievers_
    ) pair.second->stop();
  }
  
  
  bool DataManager::getQueueIDsFromDataEventRetrievers
  (
    stor::EventConsRegPtr eventConsumer,
    stor::QueueIDs& queueIDs
  ) const
  {
    if ( ! eventConsumer ) return false;
    
    DataEventRetrieverMap::const_iterator pos =
      dataEventRetrievers_.find(eventConsumer);
    if ( pos == dataEventRetrievers_.end() ) return false;
    
    queueIDs = pos->second->getQueueIDs();
    return true;
  }
  
  
  bool DataManager::getQueueIDsFromDQMEventRetrievers
  (
    stor::DQMEventConsRegPtr dqmEventConsumer,
    stor::QueueIDs& queueIDs
  ) const
  {
    if ( ! dqmEventConsumer ) return false;
    
    DQMEventRetrieverMap::const_iterator pos =
      dqmEventRetrievers_.find(dqmEventConsumer);
    if ( pos == dqmEventRetrievers_.end() ) return false;
    
    queueIDs = pos->second->getQueueIDs();
    return true;
  }
  
  
  void DataManager::activity()
  {
    try
    {
      doIt();
    }
    catch(xcept::Exception &e)
    {
      stateMachine_->moveToFailedState(e);
    }
    catch(std::exception &e)
    {
      XCEPT_DECLARE(exception::Exception,
        sentinelException, e.what());
      stateMachine_->moveToFailedState(sentinelException);
    }
    catch(...)
    {
      std::string errorMsg = "Unknown exception in watch dog";
      XCEPT_DECLARE(exception::Exception,
        sentinelException, errorMsg);
      stateMachine_->moveToFailedState(sentinelException);
    }
  }
  
  
  void DataManager::doIt()
  {
    stor::RegPtr regPtr;
    bool process(true);

    DQMArchiver dqmArchiver(stateMachine_);
    addDQMEventConsumer(dqmArchiver.getRegPtr());

    do
    {
      registrationQueue_->deqWait(regPtr);

      if ( ! (addEventConsumer(regPtr) || addDQMEventConsumer(regPtr)) )
      {
        // base type received, signalling the end of the run
        process = false;
      }
    } while (process);
  }
  
  
  bool DataManager::addEventConsumer(stor::RegPtr regPtr)
  {
    stor::EventConsRegPtr eventConsumer =
      boost::dynamic_pointer_cast<stor::EventConsumerRegistrationInfo>(regPtr);
    
    if ( ! eventConsumer ) return false;

    DataEventRetrieverMap::iterator pos = dataEventRetrievers_.lower_bound(eventConsumer);
    if (
      pos == dataEventRetrievers_.end() ||
      (dataEventRetrievers_.key_comp()(eventConsumer, pos->first))
    )
    {
      // no retriever found for this event requests
      DataEventRetrieverPtr dataEventRetriever(
        new DataEventRetriever(stateMachine_, eventConsumer)
      );
      dataEventRetrievers_.insert(pos,
        DataEventRetrieverMap::value_type(eventConsumer, dataEventRetriever));
    }
    else
    {
      pos->second->addConsumer( eventConsumer );
    }

    return true;
  }
  
  
  bool DataManager::addDQMEventConsumer(stor::RegPtr regPtr)
  {
    stor::DQMEventConsRegPtr dqmEventConsumer =
      boost::dynamic_pointer_cast<stor::DQMEventConsumerRegistrationInfo>(regPtr);
    
    if ( ! dqmEventConsumer ) return false;

    DQMEventRetrieverMap::iterator pos =
      dqmEventRetrievers_.lower_bound(dqmEventConsumer);
    if (
      pos == dqmEventRetrievers_.end() ||
      (dqmEventRetrievers_.key_comp()(dqmEventConsumer, pos->first)) )
    {
      // no retriever found for this DQM event requests
      DQMEventRetrieverPtr dqmEventRetriever(
        new DQMEventRetriever(stateMachine_, dqmEventConsumer)
      );
      dqmEventRetrievers_.insert(pos,
        DQMEventRetrieverMap::value_type(dqmEventConsumer, dqmEventRetriever));
    }
    else
    {
      pos->second->addConsumer( dqmEventConsumer );
    }
    
    return true;
  }
  
  
  void DataManager::watchDog()
  {
    try
    {
      checkForStaleConsumers();
    }
    catch(boost::thread_interrupted)
    {
      // thread was interrupted.
    }
    catch(xcept::Exception &e)
    {
      stateMachine_->moveToFailedState(e);
    }
    catch(std::exception &e)
    {
      XCEPT_DECLARE(exception::Exception,
        sentinelException, e.what());
      stateMachine_->moveToFailedState(sentinelException);
    }
    catch(...)
    {
      std::string errorMsg = "Unknown exception in watch dog";
      XCEPT_DECLARE(exception::Exception,
        sentinelException, errorMsg);
      stateMachine_->moveToFailedState(sentinelException);
    }
  }
  
  
  void DataManager::checkForStaleConsumers()
  {
    EventQueueCollectionPtr eventQueueCollection =
      stateMachine_->getEventQueueCollection();
    stor::DQMEventQueueCollectionPtr dqmEventQueueCollection =
      stateMachine_->getDQMEventQueueCollection();
    
    while (true)
    {
      boost::this_thread::sleep(boost::posix_time::seconds(1));
      stor::utils::TimePoint_t now = stor::utils::getCurrentTime();
      eventQueueCollection->clearStaleQueues(now);
      dqmEventQueueCollection->clearStaleQueues(now);
    }
  }

} // namespace smproxy
  
/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
