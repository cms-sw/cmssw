// $Id: QueueCollection.h,v 1.12 2011/03/07 15:31:32 mommsen Exp $
/// @file: QueueCollection.h 

#ifndef EventFilter_StorageManager_QueueCollection_h
#define EventFilter_StorageManager_QueueCollection_h

#include <vector>
#include <limits>

#include "boost/bind.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/shared_ptr.hpp"

#include "FWCore/Utilities/interface/Algorithms.h"

#include "EventFilter/StorageManager/interface/ConcurrentQueue.h"
#include "EventFilter/StorageManager/interface/ConsumerID.h"
#include "EventFilter/StorageManager/interface/EnquingPolicyTag.h"
#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/ExpirableQueue.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/RegistrationInfoBase.h"
#include "EventFilter/StorageManager/interface/Utils.h"
#include "EventFilter/StorageManager/interface/ConsumerMonitorCollection.h"


namespace stor {

  /**
   * Class template QueueCollection provides a collection of 
   * ConcurrentQueue<T>.
   *
   * The class T must implement a method getEventConsumerTags() const,
   * returning a std::vector<QueueID> which gives the list
   * of QueueIDs of queues the class should be added.
   *
   * $Author: mommsen $
   * $Revision: 1.12 $
   * $Date: 2011/03/07 15:31:32 $
   */

  template <class T>
  class QueueCollection
  {
  public:
    typedef typename ExpirableQueue<T, RejectNewest<T> >::SizeType SizeType;
    typedef typename ExpirableQueue<T, RejectNewest<T> >::ValueType ValueType;

    /**
       A default-constructed QueueCollection contains no queues
     */
    QueueCollection(ConsumerMonitorCollection&);

    /**
       Set or get the time in seconds that the queue with the given id
       can be unused (by a consumer) before becoming stale.
     */
    void setExpirationInterval(const QueueID&, const utils::Duration_t&);
    utils::Duration_t getExpirationInterval(const QueueID& id) const;

    /**
      Create a new contained queue, with the given policy and given
      maximum size. It returns a unique identifier to later identify
      requests originating from this consumer.
    */
    QueueID createQueue
    (
      const EventConsRegPtr,
      const utils::TimePoint_t& now = utils::getCurrentTime()
    );
    QueueID createQueue
    (
      const RegPtr,
      const utils::TimePoint_t& now = utils::getCurrentTime()
    );
    
    /**
       Remove all contained queues. Note that this has the effect of
       clearing all the queues as well.
    */
    void removeQueues();

    /**
       Return the number of queues in the collection.
      */
    SizeType size() const;
    
    /**
       Add an event to all queues matching the specifications.
     */
    void addEvent(T const&);

    /**
      Remove and return an event from the queue for the consumer with
      the given id. If there is no event in that queue, an empty
      event is returned.
     */
    ValueType popEvent(const QueueID&);

    /**
      Remove and return an event from the queue for the consumer with
      the given ConsumerID. If there is no event in that queue, an
      empty event is returned.
     */
    ValueType popEvent(const ConsumerID&);

    /**
       Clear the queue with the given QueueID.
     */
    void clearQueue(const QueueID&);

    /**
       Clear all queues which are stale at the specified point in time.
     */
    bool clearStaleQueues(const utils::TimePoint_t&);

    /**
       Clear all the contained queues.
    */
    void clearQueues();

    /**
       Test to see if the queue with the given QueueID is empty.
    */
    bool empty(const QueueID&) const;

    /**
       Test to see if the queue with the given QueueID is full.
     */
    bool full(const QueueID&) const;

    /**
       Test to see if the queue with the given QueueID is stale
       at the given time.
     */
    bool stale(const QueueID&, const utils::TimePoint_t&) const;

    /**
       Returns true if all queues are stale at the given time.
     */
    bool allQueuesStale(const utils::TimePoint_t&) const;

    /**
       Get number of elements in queue
     */
    SizeType size(const QueueID&) const;


  private:
    typedef ExpirableQueue<T, RejectNewest<T> > ExpirableDiscardNewQueue_t;
    typedef ExpirableQueue<T, KeepNewest<T> > ExpirableDiscardOldQueue_t;

    typedef boost::shared_ptr<ExpirableDiscardNewQueue_t> 
            ExpirableDiscardNewQueuePtr;
    typedef boost::shared_ptr<ExpirableDiscardOldQueue_t> 
            ExpirableDiscardOldQueuePtr;

    // These typedefs need to be changed when we move to Boost 1.38
    typedef boost::mutex::scoped_lock ReadLock_t;
    typedef boost::mutex::scoped_lock WriteLock_t;
    typedef boost::mutex ReadWriteMutex_t;

    // It is possible that one mutex would be better than these
    // three. Only profiling the application will tell for sure.
    mutable ReadWriteMutex_t protectDiscardNewQueues_;
    mutable ReadWriteMutex_t protectDiscardOldQueues_;
    mutable ReadWriteMutex_t protectLookup_;

    typedef std::vector<ExpirableDiscardNewQueuePtr> DiscardNewQueues_t;
    DiscardNewQueues_t discardNewQueues_;
    typedef std::vector<ExpirableDiscardOldQueuePtr> DiscardOldQueues_t;
    DiscardOldQueues_t discardOldQueues_;

    typedef std::map<ConsumerID, QueueID> IDLookup_t;
    IDLookup_t queueIdLookup_;
    typedef std::map<EventConsRegPtr, QueueID,
                     utils::ptrComp<EventConsumerRegistrationInfo>
                     > ReginfoLookup_t;
    ReginfoLookup_t queueReginfoLookup_;
    ConsumerMonitorCollection& consumerMonitorCollection_;

    /*
      These functions are declared private and not implemented to
      prevent their use.
    */
    QueueCollection(QueueCollection const&);
    QueueCollection& operator=(QueueCollection const&);

    /*
      These are helper functions used in the implementation.
    */
    
    SizeType enqueueEvent_(QueueID const&, T const&, utils::TimePoint_t const&);
    QueueID getQueue(const RegPtr, const utils::TimePoint_t&);

  };

  //------------------------------------------------------------------
  // Implementation follows
  //------------------------------------------------------------------

  /**
   N.B.: To avoid deadlock, in any member function that must obtain a
   lock on both the discardNewQueues and on the discardOldQueues,
   always do the locking in that order.
  */

  namespace
  {
    void throwUnknownQueueid(const QueueID& id)
    {
      std::ostringstream msg;
      msg << "Unable to retrieve queue with signature: ";
      msg << id;
      XCEPT_RAISE(exception::UnknownQueueId, msg.str());
    }
  } // anonymous namespace
  
  template <class T>
  QueueCollection<T>::QueueCollection(ConsumerMonitorCollection& ccp ) :
  protectDiscardNewQueues_(),
  protectDiscardOldQueues_(),
  protectLookup_(),
  discardNewQueues_(),
  discardOldQueues_(),
  queueIdLookup_(),
  consumerMonitorCollection_( ccp )
  { }
  
  template <class T>
  void
  QueueCollection<T>::setExpirationInterval(
    const QueueID& id,
    const utils::Duration_t& interval
  )
  {
    switch (id.policy()) 
    {
      case enquing_policy::DiscardNew:
      {
        ReadLock_t lock(protectDiscardNewQueues_);
        try
        {
          discardNewQueues_.at(id.index())->setStalenessInterval(interval);
        }
        catch(std::out_of_range)
        {
          throwUnknownQueueid(id);
        }
        break;
      }
      case enquing_policy::DiscardOld:
      {
        ReadLock_t lock(protectDiscardOldQueues_);
        try
        {
          discardOldQueues_.at(id.index())->setStalenessInterval(interval);
        }
        catch(std::out_of_range)
        {
          throwUnknownQueueid(id);
        }
        break;
      }
      default:
      {
        throwUnknownQueueid(id);
        // does not return, no break needed
      }
    }
  }
  
  template <class T>
  utils::Duration_t
  QueueCollection<T>::getExpirationInterval(const QueueID& id) const
  {
    utils::Duration_t result = boost::posix_time::seconds(0);
    switch (id.policy()) 
    {
      case enquing_policy::DiscardNew:
      {
        ReadLock_t lock(protectDiscardNewQueues_);
        try
        {
          result = discardNewQueues_.at(id.index())->stalenessInterval();
        }
        catch(std::out_of_range)
        {
          throwUnknownQueueid(id);
        }
        break;
      }
      case enquing_policy::DiscardOld:
      {
        ReadLock_t lock(protectDiscardOldQueues_);
        try
        {
          result = discardOldQueues_.at(id.index())->stalenessInterval();
        }
        catch(std::out_of_range)
        {
          throwUnknownQueueid(id);
        }
        break;
      }
      default:
      {
        throwUnknownQueueid(id);
        // does not return, no break needed
      }
    }
    return result;
  }
  
  template <class T>
  QueueID
  QueueCollection<T>::createQueue
  (
    const EventConsRegPtr reginfo,
    const utils::TimePoint_t& now
  )
  {
    QueueID qid;
    const ConsumerID& cid = reginfo->consumerId();
    
    // We don't proceed if the given ConsumerID is invalid, or if
    // we've already seen that value before.
    if (!cid.isValid()) return qid;
    WriteLock_t lockLookup(protectLookup_);
    if (queueIdLookup_.find(cid) != queueIdLookup_.end()) return qid;
    
    if ( reginfo->uniqueEvents() )
    {
      // another consumer wants to share the
      // queue to get unique events.
      ReginfoLookup_t::const_iterator it =
        queueReginfoLookup_.find(reginfo);
      if ( it != queueReginfoLookup_.end() )
      {
        qid = it->second;
        queueIdLookup_[cid] = qid;
        return qid;
      }
    }
    
    qid = getQueue(reginfo, now);
    queueIdLookup_[cid] = qid;
    queueReginfoLookup_[reginfo] = qid;
    return qid;
  }
  
  template <class T>
  QueueID 
  QueueCollection<T>::createQueue
  (
    const RegPtr reginfo,
    const utils::TimePoint_t& now
  )
  {
    QueueID qid;
    const ConsumerID& cid = reginfo->consumerId();

    // We don't proceed if the given ConsumerID is invalid, or if
    // we've already seen that value before.
    if (!cid.isValid()) return qid;
    WriteLock_t lockLookup(protectLookup_);
    if (queueIdLookup_.find(cid) != queueIdLookup_.end()) return qid;
    qid = getQueue(reginfo, now);
    queueIdLookup_[cid] = qid;
    return qid;
  }
  
  template <class T>
  QueueID
  QueueCollection<T>::getQueue
  (
    const RegPtr reginfo,
    const utils::TimePoint_t& now
  )
  {
    if (reginfo->queuePolicy() == enquing_policy::DiscardNew)
    {
      WriteLock_t lock(protectDiscardNewQueues_);
      ExpirableDiscardNewQueuePtr newborn(
        new ExpirableDiscardNewQueue_t(
          reginfo->queueSize(),
          reginfo->secondsToStale(),
          now
        )
      );
      discardNewQueues_.push_back(newborn);
      return QueueID(
        enquing_policy::DiscardNew,
        discardNewQueues_.size()-1
      );
    }
    else if (reginfo->queuePolicy() == enquing_policy::DiscardOld)
    {
      WriteLock_t lock(protectDiscardOldQueues_);
      ExpirableDiscardOldQueuePtr newborn(
        new ExpirableDiscardOldQueue_t(
          reginfo->queueSize(),
          reginfo->secondsToStale(),
          now
        )
      );
      discardOldQueues_.push_back(newborn);
      return QueueID(
        enquing_policy::DiscardOld,
        discardOldQueues_.size()-1
      );
    }
    return QueueID();
  }
  
  template <class T>
  void
  QueueCollection<T>::removeQueues()
  {
    clearQueues();
    
    WriteLock_t lockDiscardNew(protectDiscardNewQueues_);
    WriteLock_t lockDiscardOld(protectDiscardOldQueues_);
    discardNewQueues_.clear();
    discardOldQueues_.clear();    

    WriteLock_t lockLookup(protectLookup_);
    queueIdLookup_.clear();
  }
  
  template <class T>
  typename QueueCollection<T>::SizeType
  QueueCollection<T>::size() const
  {
    // We obtain locks not because it is unsafe to read the sizes
    // without locking, but because we want consistent values.
    ReadLock_t lockDiscardNew(protectDiscardNewQueues_);
    ReadLock_t lockDiscardOld(protectDiscardOldQueues_);
    return discardNewQueues_.size() + discardOldQueues_.size();
  }
  
  template <class T>
  void 
  QueueCollection<T>::addEvent(T const& event)
  {
    ReadLock_t lockDiscardNew(protectDiscardNewQueues_);
    ReadLock_t lockDiscardOld(protectDiscardOldQueues_);
    
    utils::TimePoint_t now = utils::getCurrentTime();
    QueueIDs routes = event.getEventConsumerTags();
    
    for( QueueIDs::const_iterator it = routes.begin(), itEnd = routes.end();
         it != itEnd; ++it )
    {
      const SizeType droppedEvents = enqueueEvent_( *it, event, now );
      if ( droppedEvents > 0 )
        consumerMonitorCollection_.addDroppedEvents( *it, droppedEvents );
      else
        consumerMonitorCollection_.addQueuedEventSample( *it, event.totalDataSize() );
    }
  }
  
  template <class T>
  typename QueueCollection<T>::ValueType
  QueueCollection<T>::popEvent(const QueueID& id)
  {
    ValueType result;
    switch (id.policy()) 
    {
      case enquing_policy::DiscardNew:
      {
        ReadLock_t lock(protectDiscardNewQueues_);
        try
        {
          if ( discardNewQueues_.at(id.index())->deqNowait(result) )
            consumerMonitorCollection_.addServedEventSample(id,
              result.first.totalDataSize());
        }
        catch(std::out_of_range)
        {
          throwUnknownQueueid(id);
        }
        break;
      }
      case enquing_policy::DiscardOld:
      {
        ReadLock_t lock(protectDiscardOldQueues_);
        try
        {
          if ( discardOldQueues_.at(id.index())->deqNowait(result) )
            consumerMonitorCollection_.addServedEventSample(id,
              result.first.totalDataSize());
        }
        catch(std::out_of_range)
        {
          throwUnknownQueueid(id);
        }
        break;
      }
      default:
      {
        throwUnknownQueueid(id);
        // does not return, no break needed
      }
    }
    return result;
  }
  
  template <class T>
  typename QueueCollection<T>::ValueType
  QueueCollection<T>::popEvent(const ConsumerID& cid)
  {
    ValueType result;
    if (!cid.isValid()) return result;
    QueueID id;
    {
      // Scope to control lifetime of lock.
      ReadLock_t lock(protectLookup_);
      IDLookup_t::const_iterator i = queueIdLookup_.find(cid);
      if (i == queueIdLookup_.end()) return result;
      id = i->second;
    }
    return popEvent(id);
  }
  
  template <class T>
  void
  QueueCollection<T>::clearQueue(const QueueID& id)
  {
    switch (id.policy()) 
    {
      case enquing_policy::DiscardNew:
      {
        ReadLock_t lock(protectDiscardNewQueues_);
        try
        {
          const SizeType clearedEvents =
            discardNewQueues_.at(id.index())->clear();
          
          consumerMonitorCollection_.addDroppedEvents(
            id, clearedEvents);
        }
        catch(std::out_of_range)
        {
          throwUnknownQueueid(id);
        }
        break;
      }
      case enquing_policy::DiscardOld:
      {
        ReadLock_t lock(protectDiscardOldQueues_);
        try
        {
          const SizeType clearedEvents = 
            discardOldQueues_.at(id.index())->clear();
          
          consumerMonitorCollection_.addDroppedEvents(
            id, clearedEvents);
        }
        catch(std::out_of_range)
        {
          throwUnknownQueueid(id);
        }
        break;
      }
      default:
      {
        throwUnknownQueueid(id);
        // does not return, no break needed
      }
    }
  }
  
  template <class T>
  bool
  QueueCollection<T>::clearStaleQueues(const utils::TimePoint_t& now)
  {
    bool result(false);
    SizeType clearedEvents;
    
    {
      ReadLock_t lock(protectDiscardNewQueues_);
      const SizeType numQueues = discardNewQueues_.size();
      for (SizeType i = 0; i < numQueues; ++i)
      {
        if ( discardNewQueues_[i]->clearIfStale(now, clearedEvents) )
        {
          consumerMonitorCollection_.addDroppedEvents(
            QueueID(enquing_policy::DiscardNew, i),
            clearedEvents
          );
          result = true;
        }
      }
    }
    {
      ReadLock_t lock(protectDiscardOldQueues_);
      const SizeType numQueues = discardOldQueues_.size();
      for (SizeType i = 0; i < numQueues; ++i)
      {
        if ( discardOldQueues_[i]->clearIfStale(now, clearedEvents) )
        {
          consumerMonitorCollection_.addDroppedEvents(
            QueueID(enquing_policy::DiscardOld, i),
            clearedEvents
          );
          result = true;
        }
      }
    }
    return result;
  }
  
  template <class T>
  void
  QueueCollection<T>::clearQueues()
  {
    {
      ReadLock_t lock(protectDiscardNewQueues_);
      const SizeType numQueues = discardNewQueues_.size();
      for (SizeType i = 0; i < numQueues; ++i)
      {
        const SizeType clearedEvents =
          discardNewQueues_[i]->clear();

        consumerMonitorCollection_.addDroppedEvents(
          QueueID(enquing_policy::DiscardNew, i),
          clearedEvents
        );
      }
    }
    {
      ReadLock_t lock(protectDiscardOldQueues_);
      const SizeType numQueues = discardOldQueues_.size();
      for (SizeType i = 0; i < numQueues; ++i)
      {
        const SizeType clearedEvents =
          discardOldQueues_[i]->clear();

        consumerMonitorCollection_.addDroppedEvents(
          QueueID(enquing_policy::DiscardOld, i),
          clearedEvents
        );
      }
    }
  }
  
  template <class T>
  bool
  QueueCollection<T>::empty(const QueueID& id) const
  {
    bool result(true);
    switch (id.policy()) 
    {
      case enquing_policy::DiscardNew:
      {
        ReadLock_t lock(protectDiscardNewQueues_);
        try
        {
          result = discardNewQueues_.at(id.index())->empty();
        }
        catch(std::out_of_range)
        {
          throwUnknownQueueid(id);
        }
        break;
      }
      case enquing_policy::DiscardOld:
      {
        ReadLock_t lock(protectDiscardOldQueues_);
        try
        {
          result = discardOldQueues_.at(id.index())->empty();
        }
        catch(std::out_of_range)
        {
          throwUnknownQueueid(id);
        }
        break;
      }
      default:
      {
        throwUnknownQueueid(id);
        // does not return, no break needed
      }
    }
    return result;
  }
  
  template <class T>
  bool
  QueueCollection<T>::full(const QueueID& id) const
  {
    bool result(true);
    switch (id.policy()) 
    {
      case enquing_policy::DiscardNew:
      {
        ReadLock_t lock(protectDiscardNewQueues_);
        try
        {
          result = discardNewQueues_.at(id.index())->full();
        }
        catch(std::out_of_range)
        {
          throwUnknownQueueid(id);
        }
        break;
      }
      case enquing_policy::DiscardOld:
      {
        ReadLock_t lock(protectDiscardOldQueues_);
        try
        {
          result = discardOldQueues_.at(id.index())->full();
        }
        catch(std::out_of_range)
        {
          throwUnknownQueueid(id);
        }
        break;
      }
      default:
      {
        throwUnknownQueueid(id);
        // does not return, no break needed
      }
    }
    return result;
  }
  
  template <class T>
  bool
  QueueCollection<T>::stale(const QueueID& id, const utils::TimePoint_t& now) const
  {
    bool result(true);
    switch (id.policy()) 
    {
      case enquing_policy::DiscardNew:
      {
        ReadLock_t lock(protectDiscardNewQueues_);
        try
        {
          result = discardNewQueues_.at(id.index())->stale(now);
        }
        catch(std::out_of_range)
        {
          throwUnknownQueueid(id);
        }
        break;
      }
      case enquing_policy::DiscardOld:
      {
        ReadLock_t lock(protectDiscardOldQueues_);
        try
        {
          result = discardOldQueues_.at(id.index())->stale(now);
        }
        catch(std::out_of_range)
        {
          throwUnknownQueueid(id);
        }
        break;
      }
      default:
      {
        throwUnknownQueueid(id);
        // does not return, no break needed
      }
    }
    return result;
  }
  
  template <class T>
  bool
  QueueCollection<T>::allQueuesStale(const utils::TimePoint_t& now) const
  {
    {
      ReadLock_t lock(protectDiscardNewQueues_);
      const SizeType numQueues = discardNewQueues_.size();
      for (SizeType i = 0; i < numQueues; ++i)
      {
        if ( ! discardNewQueues_[i]->stale(now) ) return false;
      }
    }
    {
      ReadLock_t lock(protectDiscardOldQueues_);
      const SizeType numQueues = discardOldQueues_.size();
      for (SizeType i = 0; i < numQueues; ++i)
      {
        if ( ! discardOldQueues_[i]->stale(now) ) return false;
      }
    }
    return true;
  }
  
  template <class T>
  typename QueueCollection<T>::SizeType
  QueueCollection<T>::size(const QueueID& id) const
  {
    SizeType result = 0;
    switch (id.policy()) 
    {
      case enquing_policy::DiscardNew:
      {
        ReadLock_t lock(protectDiscardNewQueues_);
        try
        {
          result = discardNewQueues_.at(id.index())->size();
        }
        catch(std::out_of_range)
        {
          throwUnknownQueueid(id);
        }
        break;
      }
      case enquing_policy::DiscardOld:
      {
        ReadLock_t lock(protectDiscardOldQueues_);
        try
        {
          result = discardOldQueues_.at(id.index())->size();
        }
        catch(std::out_of_range)
        {
          throwUnknownQueueid(id);
        }
        break;
      }
      default:
      {
        throwUnknownQueueid( id );
        // does not return, no break needed
      }
    }
    return result;
  }
  
  template <class T>
  typename QueueCollection<T>::SizeType
  QueueCollection<T>::enqueueEvent_
  (
    QueueID const& id, 
    T const& event,
    utils::TimePoint_t const& now
  )
  {
    switch (id.policy())
    {
      case enquing_policy::DiscardNew:
      {
        try
        {
          return discardNewQueues_.at(id.index())->enqNowait(event,now);
        }
        catch(std::out_of_range)
        {
          throwUnknownQueueid(id);
        }
        break;
      }
      case enquing_policy::DiscardOld:
      {
        try
        {
          return discardOldQueues_.at(id.index())->enqNowait(event,now);
        }
        catch(std::out_of_range)
        {
          throwUnknownQueueid(id);
        }
        break;
      }
      default:
      {
        throwUnknownQueueid(id);
        // does not return, no break needed
      }
    }
    return 1; // event could not be entered
  }
  
} // namespace stor

#endif // EventFilter_StorageManager_QueueCollection_h 

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
