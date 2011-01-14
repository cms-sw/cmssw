// $Id: QueueCollection.h,v 1.10 2010/12/20 16:33:21 mommsen Exp $
/// @file: QueueCollection.h 

#ifndef StorageManager_QueueCollection_h
#define StorageManager_QueueCollection_h

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
   * $Revision: 1.10 $
   * $Date: 2010/12/20 16:33:21 $
   */

  template <class T>
  class QueueCollection
  {
  public:
    typedef typename ExpirableQueue<T, RejectNewest<T> >::size_type size_type;

    /**
       A default-constructed QueueCollection contains no queues
     */
    QueueCollection(ConsumerMonitorCollection&);

    /**
       Set or get the time in seconds that the queue with the given id
       can be unused (by a consumer) before becoming stale.
     */
    void setExpirationInterval(QueueID id, utils::duration_t interval);
    utils::duration_t getExpirationInterval(QueueID id) const;

    /**
      Create a new contained queue, with the given policy and given
      maximum size. It returns a unique identifier to later identify
      requests originating from this consumer.
    */
    QueueID createQueue
    (
      const EventConsRegPtr,
      const utils::time_point_t& now = utils::getCurrentTime()
    );
    QueueID createQueue
    (
      const RegPtr,
      const utils::time_point_t& now = utils::getCurrentTime()
    );
    
    /**
       Remove all contained queues. Note that this has the effect of
       clearing all the queues as well.
    */
    void removeQueues();

    /**
       Return the number of queues in the collection.
      */
    size_type size() const;
    
    /**
       Add an event to all queues matching the specifications.
     */
    void addEvent(T const&);

    /**
      Remove and return an event from the queue for the consumer with
      the given id. If there is no event in that queue, an empty
      event is returned.
     */
    T popEvent(QueueID);

    /**
      Remove and return an event from the queue for the consumer with
      the given ConsumerID. If there is no event in that queue, an
      empty event is returned.
     */
    T popEvent(ConsumerID);

    /**
       Clear the queue with the given QueueID.
     */
    void clearQueue(QueueID);

    /**
       Clear all the contained queues.
    */
    void clearQueues();

    /**
       Test to see if the queue with the given QueueID is empty.
    */
    bool empty(QueueID) const;

    /**
       Test to see if the queue with the given QueueID is full.
     */
    bool full(QueueID) const;

    /**
       Get number of elements in queue
    */
    size_type size(QueueID) const;

    /**
       Clear queues which are 'stale'; a queue is stale if it hasn't
      been requested by a consumer within its 'staleness
      interval. Return the QueueID for each queue that is stale (not
      merely those that have become stale recently, but all that are
      stale) in the output argument 'stale_queues'.
     */
    void clearStaleQueues(std::vector<QueueID>& stale_queues);


  private:
    typedef ExpirableQueue<T, RejectNewest<T> > expirable_discard_new_queue_t;
    typedef ExpirableQueue<T, KeepNewest<T> > expirable_discard_old_queue_t;


    typedef boost::shared_ptr<expirable_discard_new_queue_t> 
            expirable_discard_new_queue_ptr;
    typedef boost::shared_ptr<expirable_discard_old_queue_t> 
            expirable_discard_old_queue_ptr;

    // These typedefs need to be changed when we move to Boost 1.38
    typedef boost::mutex::scoped_lock read_lock_t;
    typedef boost::mutex::scoped_lock write_lock_t;
    typedef boost::mutex  read_write_mutex;

    // It is possible that one mutex would be better than these
    // three. Only profiling the application will tell for sure.
    mutable read_write_mutex  _protect_discard_new_queues;
    mutable read_write_mutex  _protect_discard_old_queues;
    mutable read_write_mutex  _protect_lookup;

    typedef std::vector<expirable_discard_new_queue_ptr> discard_new_queues_t;
    discard_new_queues_t _discard_new_queues;
    typedef std::vector<expirable_discard_old_queue_ptr> discard_old_queues_t;
    discard_old_queues_t _discard_old_queues;

    typedef std::map<ConsumerID, QueueID>        id_lookup_t;
    id_lookup_t                                  _queue_id_lookup;
    typedef std::map<EventConsRegPtr, QueueID,
                     utils::ptr_comp<EventConsumerRegistrationInfo> > reginfo_lookup_t;
    reginfo_lookup_t                             _queue_reginfo_lookup;
    ConsumerMonitorCollection& _consumer_monitor_collection;

    /*
      These functions are declared private and not implemented to
      prevent their use.
    */
    QueueCollection(QueueCollection const&);
    QueueCollection& operator=(QueueCollection const&);

    /*
      These are helper functions used in the implementation.
    */
    
    size_type _enqueue_event(QueueID const& id, T const& event);
    //QueueID get_queue(const EventConsRegPtr, const utils::time_point_t&);
    QueueID get_queue(const RegPtr, const utils::time_point_t&);

  };

  //------------------------------------------------------------------
  // Implementation follows
  //------------------------------------------------------------------

  /**
   N.B.: To avoid deadlock, in any member function that must obtain a
   lock on both the discard_new_queues and on the discard_old_queues,
   always do the locking in that order.
  */

  namespace
  {
    void throw_unknown_queueid(QueueID id)
    {
      std::ostringstream msg;
      msg << "Unable to retrieve queue with signature: ";
      msg << id;
      XCEPT_RAISE(exception::UnknownQueueId, msg.str());
    }
  } // anonymous namespace

  template <class T>
  QueueCollection<T>::QueueCollection(ConsumerMonitorCollection& ccp ) :
    _protect_discard_new_queues(),
    _protect_discard_old_queues(),
    _protect_lookup(),
    _discard_new_queues(),
    _discard_old_queues(),
    _queue_id_lookup(),
    _consumer_monitor_collection( ccp )
  { }

  template <class T>
  void
  QueueCollection<T>::setExpirationInterval(QueueID id,
                                         utils::duration_t interval)
  {
    switch (id.policy()) 
      {
      case enquing_policy::DiscardNew:
        {
          read_lock_t lock(_protect_discard_new_queues);
          if (id.index() < _discard_new_queues.size())
            _discard_new_queues[id.index()]->set_staleness_interval(interval);
          break;
        }
      case enquing_policy::DiscardOld:
        {
          read_lock_t lock(_protect_discard_old_queues);
          if (id.index() < _discard_old_queues.size())
            _discard_old_queues[id.index()]->set_staleness_interval(interval);
          break;
        }
      default:
        {
          throw_unknown_queueid(id);
          // does not return, no break needed
        }
      }
  }

  template <class T>
  utils::duration_t
  QueueCollection<T>::getExpirationInterval(QueueID id) const
  {
    utils::duration_t result = boost::posix_time::seconds(0);
    switch (id.policy()) 
      {
      case enquing_policy::DiscardNew:
        {
          read_lock_t lock(_protect_discard_new_queues);
          if (id.index() < _discard_new_queues.size())
            result = _discard_new_queues[id.index()]->staleness_interval();
          break;
        }
      case enquing_policy::DiscardOld:
        {
          read_lock_t lock(_protect_discard_old_queues);
          if (id.index() < _discard_old_queues.size())
            result = _discard_old_queues[id.index()]->staleness_interval();
          break;
        }
      default:
        {
          throw_unknown_queueid(id);
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
    const utils::time_point_t& now
  )
  {
    QueueID qid;
    const ConsumerID& cid = reginfo->consumerId();
    
    // We don't proceed if the given ConsumerID is invalid, or if
    // we've already seen that value before.
    if (!cid.isValid()) return qid;
    write_lock_t lock_lookup(_protect_lookup);
    if (_queue_id_lookup.find(cid) != _queue_id_lookup.end()) return qid;

    if ( reginfo->uniqueEvents() )
    {
      // another consumer wants to share the
      // queue to get unique events.
      reginfo_lookup_t::const_iterator it =
        _queue_reginfo_lookup.find(reginfo);
      if ( it != _queue_reginfo_lookup.end() )
      {
        qid = it->second;
        _queue_id_lookup[cid] = qid;
        return qid;
      }
    }

    qid = get_queue(reginfo, now);
    _queue_id_lookup[cid] = qid;
    _queue_reginfo_lookup[reginfo] = qid;
    return qid;
  }

  template <class T>
  QueueID 
  QueueCollection<T>::createQueue
  (
    const RegPtr reginfo,
    const utils::time_point_t& now
  )
  {
    QueueID qid;
    const ConsumerID& cid = reginfo->consumerId();

    // We don't proceed if the given ConsumerID is invalid, or if
    // we've already seen that value before.
    if (!cid.isValid()) return qid;
    write_lock_t lock_lookup(_protect_lookup);
    if (_queue_id_lookup.find(cid) != _queue_id_lookup.end()) return qid;
    qid = get_queue(reginfo, now);
    _queue_id_lookup[cid] = qid;
    return qid;
  }

  template <class T>
  QueueID
  QueueCollection<T>::get_queue
  (
    const RegPtr reginfo,
    const utils::time_point_t& now
  )
  {
    if (reginfo->queuePolicy() == enquing_policy::DiscardNew)
    {
      write_lock_t lock(_protect_discard_new_queues);
      expirable_discard_new_queue_ptr newborn(
        new expirable_discard_new_queue_t(
        reginfo->queueSize(),
        reginfo->secondsToStale(),
        now
        )
      );
      _discard_new_queues.push_back(newborn);
      return QueueID(
        enquing_policy::DiscardNew,
        _discard_new_queues.size()-1
      );
    }
    else if (reginfo->queuePolicy() == enquing_policy::DiscardOld)
    {
      write_lock_t lock(_protect_discard_old_queues);
      expirable_discard_old_queue_ptr newborn(
        new expirable_discard_old_queue_t(
          reginfo->queueSize(),
          reginfo->secondsToStale(),
          now
        )
      );
      _discard_old_queues.push_back(newborn);
      return QueueID(
        enquing_policy::DiscardOld,
        _discard_old_queues.size()-1
      );
    }
    return QueueID();
  }

  template <class T>
  void
  QueueCollection<T>::removeQueues()
  {
    clearQueues();

    write_lock_t lock_discard_new(_protect_discard_new_queues);
    write_lock_t lock_discard_old(_protect_discard_old_queues);
    _discard_new_queues.clear();
    _discard_old_queues.clear();    
  }

  template <class T>
  typename QueueCollection<T>::size_type
  QueueCollection<T>::size() const
  {
    // We obtain locks not because it is unsafe to read the sizes
    // without locking, but because we want consistent values.
    read_lock_t lock_discard_new(_protect_discard_new_queues);
    read_lock_t lock_discard_old(_protect_discard_old_queues);
    return _discard_new_queues.size() + _discard_old_queues.size();
  }

  template <class T>
  void 
  QueueCollection<T>::addEvent(T const& event)
  {

    read_lock_t lock_discard_new(_protect_discard_new_queues);
    read_lock_t lock_discard_old(_protect_discard_old_queues);

    std::vector<QueueID> routes = event.getEventConsumerTags();

    for( std::vector<QueueID>::const_iterator it = routes.begin(), itEnd = routes.end();
         it != itEnd; ++it )
      {
        const size_type discardedEvents = _enqueue_event( *it, event );
        _consumer_monitor_collection.addQueuedEventSample( *it, event.totalDataSize() );
        _consumer_monitor_collection.addDiscardedEvents( *it, discardedEvents );
      }

  }

  template <class T>
  T
  QueueCollection<T>::popEvent(QueueID id)
  {
    T result;
    switch (id.policy()) 
      {
      case enquing_policy::DiscardNew:
        {
          read_lock_t lock(_protect_discard_new_queues);
          if (id.index() < _discard_new_queues.size())
            _discard_new_queues[id.index()]->deq_nowait(result);
          break;
        }
      case enquing_policy::DiscardOld:
        {
          read_lock_t lock(_protect_discard_old_queues);
          if (id.index() < _discard_old_queues.size())
            _discard_old_queues[id.index()]->deq_nowait(result);
          break;
        }
      default:
        {
          throw_unknown_queueid(id);
          // does not return, no break needed
        }
      }

    if (!result.empty())
      {
        _consumer_monitor_collection.addServedEventSample( id, result.totalDataSize() );
      }

    return result;
  }

  template <class T>
  T
  QueueCollection<T>::popEvent(ConsumerID cid)
  {
    T result;
    if (!cid.isValid()) return result;
    QueueID id;
    {
      // Scope to control lifetime of lock.
      read_lock_t lock(_protect_lookup);
      id_lookup_t::const_iterator i = _queue_id_lookup.find(cid);
      if (i == _queue_id_lookup.end()) return result;
      id = i->second;
    }
    return popEvent(id);
  }


  template <class T>
  void
  QueueCollection<T>::clearQueue(QueueID id)
  {
    switch (id.policy()) 
      {
      case enquing_policy::DiscardNew:
        {
          read_lock_t lock(_protect_discard_new_queues);
          if (id.index() < _discard_new_queues.size())
          {
            _consumer_monitor_collection.addDiscardedEvents(
              id, _discard_new_queues[id.index()]->size() );
            _discard_new_queues[id.index()]->clear();
          }
          break;
        }
      case enquing_policy::DiscardOld:
        {
          read_lock_t lock(_protect_discard_old_queues);
          if (id.index() < _discard_old_queues.size())
          {
            _consumer_monitor_collection.addDiscardedEvents(
              id, _discard_old_queues[id.index()]->size() );
            _discard_old_queues[id.index()]->clear();
          }
          break;
        }
      default:
        {
          throw_unknown_queueid(id);
          // does not return, no break needed
        }
      }
  }

  template <class T>
  void
  QueueCollection<T>::clearQueues()
  {
    {
      read_lock_t lock_discard_new(_protect_discard_new_queues);
      const size_type num_queues = _discard_new_queues.size();
      for (size_type i = 0; i < num_queues; ++i)
      {
        _consumer_monitor_collection.addDiscardedEvents(
          QueueID(enquing_policy::DiscardNew, i),
          _discard_new_queues[i]->size()
        );
        _discard_new_queues[i]->clear();
      }
    }
    {
      read_lock_t lock_discard_old(_protect_discard_old_queues);
      const size_type num_queues = _discard_old_queues.size();
      for (size_type i = 0; i < num_queues; ++i)
      {
        _consumer_monitor_collection.addDiscardedEvents(
          QueueID(enquing_policy::DiscardOld, i),
          _discard_old_queues[i]->size()
        );
        _discard_old_queues[i]->clear();
      }

    }
  }

  template <class T>
  bool
  QueueCollection<T>::empty(QueueID id) const
  {
    bool result(true);
    switch (id.policy()) 
      {
      case enquing_policy::DiscardNew:
        {
          read_lock_t lock(_protect_discard_new_queues);
          if (id.index() < _discard_new_queues.size())
            result = _discard_new_queues[id.index()]->empty();
          break;
        }
      case enquing_policy::DiscardOld:
        {
          read_lock_t lock(_protect_discard_old_queues);
          if (id.index() < _discard_old_queues.size())
            result = _discard_old_queues[id.index()]->empty();
          break;
        }
      default:
        {
          throw_unknown_queueid(id);
          // does not return, no break needed
        }
      }
    return result;
  }

  template <class T>
  bool
  QueueCollection<T>::full(QueueID id) const
  {
    bool result(true);
    switch (id.policy()) 
      {
      case enquing_policy::DiscardNew:
        {
          read_lock_t lock(_protect_discard_new_queues);
          if (id.index() < _discard_new_queues.size())
            result = _discard_new_queues[id.index()]->full();
          break;
        }
      case enquing_policy::DiscardOld:
        {
          read_lock_t lock(_protect_discard_old_queues);
          if (id.index() < _discard_old_queues.size())
            result = _discard_old_queues[id.index()]->full();
          break;
        }
      default:
        {
          throw_unknown_queueid(id);
          // does not return, no break needed
        }
      }
    return result;
  }

  template <class T>
  typename QueueCollection<T>::size_type
  QueueCollection<T>::size( QueueID id ) const
  {
    size_type result = 0;
    switch (id.policy()) 
      {
      case enquing_policy::DiscardNew:
        {
          read_lock_t lock( _protect_discard_new_queues );
          if( id.index() < _discard_new_queues.size() )
            {
              result = _discard_new_queues[ id.index() ]->size();
            }
          else
            {
              throw_unknown_queueid( id );
            }
          break;
        }
      case enquing_policy::DiscardOld:
        {
          read_lock_t lock( _protect_discard_old_queues );
          if( id.index() < _discard_old_queues.size() )
            {
              result = _discard_old_queues[ id.index() ]->size();
            }
          else
            {
              throw_unknown_queueid( id );
            }
          break;
        }
      default:
        {
          throw_unknown_queueid( id );
          // does not return, no break needed
        }
      }
    return result;
  }


  template <class T>
  void 
  QueueCollection<T>::clearStaleQueues(std::vector<QueueID>& result)
  {
    result.clear();
    utils::time_point_t now = utils::getCurrentTime();

    {
      read_lock_t lock_discard_new(_protect_discard_new_queues);
    
      const size_type num_queues = _discard_new_queues.size();
      size_type clearedEvents;
      for (size_type i = 0; i < num_queues; ++i)
      {
        if ( _discard_new_queues[i]->clearIfStale(now, clearedEvents))
        {
          const QueueID id(enquing_policy::DiscardNew, i);
          _consumer_monitor_collection.addDiscardedEvents(id, clearedEvents);
          result.push_back(id);
        }
      }
    }

    {
      read_lock_t lock_discard_old(_protect_discard_old_queues);
      const size_type num_queues = _discard_old_queues.size();
      size_type clearedEvents;
       for (size_type i = 0; i < num_queues; ++i)
      {
        if ( _discard_old_queues[i]->clearIfStale(now, clearedEvents))
        {
          const QueueID id(enquing_policy::DiscardOld, i);
          _consumer_monitor_collection.addDiscardedEvents(id, clearedEvents);
          result.push_back(id);
        }
      }
    }
  }

  template <class T>
  typename QueueCollection<T>::size_type
  QueueCollection<T>::_enqueue_event(QueueID const& id, 
                                     T const& event)
  {
    switch (id.policy())
      {
      case enquing_policy::DiscardNew:
        {
          if (id.index() < _discard_new_queues.size())
          {
            if ( _discard_new_queues[id.index()]->enq_nowait(event) )
              return 0; // event was put into the queue
          }
          break;
        }
      case enquing_policy::DiscardOld:
        {
          if (id.index() < _discard_old_queues.size())
          {
            return _discard_old_queues[id.index()]->enq_nowait(event);
            // returns number of discarded events to make room for new one
          }
          break;
        }
      default:
        {
          throw_unknown_queueid(id);
          // does not return, no break needed
        }
      }
    return 1; // event could not be entered
  }
  
} // namespace stor

#endif // StorageManager_QueueCollection_h 

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
