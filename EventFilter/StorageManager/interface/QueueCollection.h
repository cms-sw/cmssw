// $Id: QueueCollection.h,v 1.3 2009/06/19 13:49:12 dshpakov Exp $
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
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/ExpirableQueue.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
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
   * $Author: dshpakov $
   * $Revision: 1.3 $
   * $Date: 2009/06/19 13:49:12 $
   */

  template <class T>
  class QueueCollection
  {
  public:

    typedef boost::shared_ptr<ConsumerMonitorCollection> ConsCollPtr;

    /**
       A default-constructed QueueCollection contains no queues
     */
    QueueCollection( ConsCollPtr );

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
    QueueID createQueue(ConsumerID cid,
                        enquing_policy::PolicyTag policy,
                        size_t max = std::numeric_limits<size_t>::max(),
                        utils::duration_t interval = 120.0,
                        utils::time_point_t now = utils::getCurrentTime());

    /**
       Remove all contained queues. Note that this has the effect of
       clearing all the queues as well.
    */
    void removeQueues();

    /**
       Return the number of queues in the collection.
      */
    size_t size() const;
    
    /**
       Add an event to all queues matching the specifications.
     */
    void addEvent(T const& event);

    /**
      Remove and return an event from the queue for the consumer with
      the given id. If there is no event in that queue, an empty
      event is returned.
     */
    T popEvent(QueueID id);

    /**
      Remove and return an event from the queue for the consumer with
      the given ConsumerID. If there is no event in that queue, an
      empty event is returned.
     */
    T popEvent(ConsumerID id);

    /**
       Clear the queue with the given QueueID.
     */
    void clearQueue(QueueID id);

    /**
       Clear all the contained queues.
    */
    void clearQueues();

    /**
       Test to see if the queue with the given QueueID is empty.
    */
    bool empty(QueueID id) const;

    /**
       Test to see if the queue with the given QueueID is full.
     */
    bool full(QueueID id) const;

    /**
       Get number of elements in queue
    */
    size_t size( QueueID id ) const;

    /**
       Clear queues which are 'stale'; a queue is stale if it hasn't
      been requested by a consumer within its 'staleness
      interval. Return the QueueID for each queue that is stale (not
      merely those that have become stale recently, but all that are
      stale) in the output argument 'stale_queues'.
     */
    void clearStaleQueues(std::vector<QueueID>& stale_queues);

    /**
       Get consumer monitor collection
    */
    ConsCollPtr consumerMonitorCollection();

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

    std::vector<expirable_discard_new_queue_ptr> _discard_new_queues;
    std::vector<expirable_discard_old_queue_ptr> _discard_old_queues;
    typedef std::map<ConsumerID, QueueID>        map_type;
    map_type                                     _queue_id_lookup;
    ConsCollPtr _consumer_monitor_collection;

    /*
      These functions are declared private and not implemented to
      prevent their use.
    */
    QueueCollection(QueueCollection const&);
    QueueCollection& operator=(QueueCollection const&);

    /*
      These are helper functions used in the implementation.
    */
    
    void _enqueue_event(QueueID const& id, T const& event);
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
    const utils::duration_t DEFAULT_STALENESS_INTERVAL = 120.0;

    void throw_unknown_queueid(QueueID id)
    {
      std::ostringstream msg;
      msg << "Unable to retrieve queue with signature: ";
      msg << id;
      XCEPT_RAISE(exception::UnknownQueueId, msg.str());
    }
  } // anonymous namespace

  template <class T>
  QueueCollection<T>::QueueCollection( ConsCollPtr ccp ) :
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
    utils::duration_t result(0.0);
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
  QueueCollection<T>::createQueue(ConsumerID cid,
                                    enquing_policy::PolicyTag policy,
				    size_t max,
                                    utils::duration_t interval,
                                    utils::time_point_t now)
  {
    QueueID result;

    // We don't proceed if the given ConsumerID is invalid, or if
    // we've already seen that value before.
    if (!cid.isValid()) return result;
    write_lock_t lock_lookup(_protect_lookup);
    if (_queue_id_lookup.find(cid) != _queue_id_lookup.end()) return result;

    if (policy == enquing_policy::DiscardNew)
      {
 	write_lock_t lock(_protect_discard_new_queues);
        expirable_discard_new_queue_ptr newborn(new expirable_discard_new_queue_t(max,
                                                                                  interval,
                                                                                  now));
        _discard_new_queues.push_back(newborn);
        result = QueueID(enquing_policy::DiscardNew,
 			 _discard_new_queues.size()-1);
      }
    else if (policy == enquing_policy::DiscardOld)
      {
	write_lock_t lock(_protect_discard_old_queues);
        expirable_discard_old_queue_ptr newborn(new expirable_discard_old_queue_t(max,
                                                                                  interval,
                                                                                  now));
	_discard_old_queues.push_back(newborn);
	result = QueueID(enquing_policy::DiscardOld,
			 _discard_old_queues.size()-1);

      }
    _queue_id_lookup[cid] = result;
    return result;
  }

  template <class T>
  void
  QueueCollection<T>::removeQueues()
  {
    write_lock_t lock_discard_new(_protect_discard_new_queues);
    write_lock_t lock_discard_old(_protect_discard_old_queues);
    _discard_new_queues.clear();
    _discard_old_queues.clear();    
  }

  template <class T>
  size_t
  QueueCollection<T>::size() const
  {
    // We obtain locks not because it is unsafe to read the sizes
    // without locking, but because we want consistent values.
    read_lock_t lock_discard_old(_protect_discard_new_queues);
    read_lock_t lock_discard_new(_protect_discard_old_queues);
    return _discard_new_queues.size() + _discard_old_queues.size();
  }

  template <class T>
  void 
  QueueCollection<T>::addEvent(T const& event)
  {

    read_lock_t lock_discard_old(_protect_discard_new_queues);
    read_lock_t lock_discard_new(_protect_discard_old_queues);

    std::vector<QueueID> routes = event.getEventConsumerTags();
    edm::for_all(routes,
                 boost::bind(&QueueCollection<T>::_enqueue_event, 
                             this, _1, event));

    for( std::vector<QueueID>::iterator i = routes.begin(); i != routes.end(); ++i )
      {
        _consumer_monitor_collection->addQueuedEventSample( *i, event.totalDataSize() );
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
        _consumer_monitor_collection->addServedEventSample( id, result.totalDataSize() );
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
      map_type::const_iterator i = _queue_id_lookup.find(cid);
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
            _discard_new_queues[id.index()]->clear();
          break;
        }
      case enquing_policy::DiscardOld:
        {
          read_lock_t lock(_protect_discard_old_queues);
          if (id.index() < _discard_old_queues.size())
            _discard_old_queues[id.index()]->clear();
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
    read_lock_t lock_discard_new(_protect_discard_new_queues);
    read_lock_t lock_discard_old(_protect_discard_old_queues);
    edm::for_all(_discard_new_queues, 
                 boost::bind(&expirable_discard_new_queue_t::clear, _1));
    edm::for_all(_discard_old_queues, 
                 boost::bind(&expirable_discard_old_queue_t::clear, _1));

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
  size_t
  QueueCollection<T>::size( QueueID id ) const
  {
    size_t result = 0;
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
    read_lock_t lock_discard_old(_protect_discard_new_queues);
    read_lock_t lock_discard_new(_protect_discard_old_queues);
    
    size_t num_queues = _discard_new_queues.size();
    for (size_t i = 0; i < num_queues; ++i)
      {
        if ( _discard_new_queues[i]->clearIfStale(now))
          result.push_back(QueueID(enquing_policy::DiscardNew, i));
      }

    num_queues = _discard_old_queues.size();
    for (size_t i = 0; i < num_queues; ++i)
      {
        if ( _discard_old_queues[i]->clearIfStale(now))
          result.push_back(QueueID(enquing_policy::DiscardOld, i));
      }
  }

  template <class T>
  void
  QueueCollection<T>::_enqueue_event(QueueID const& id, 
                                  T const& event)
  {
    switch (id.policy())
      {
      case enquing_policy::DiscardNew:
        {
          if (id.index() < _discard_new_queues.size())
            _discard_new_queues[id.index()]->enq_nowait(event);
          break;
        }
      case enquing_policy::DiscardOld:
        {
          if (id.index() < _discard_old_queues.size())
            _discard_old_queues[id.index()]->enq_nowait(event);
          break;
        }
      default:
        {
          throw_unknown_queueid(id);
          // does not return, no break needed
        }
      }
  }
  
} // namespace stor

#endif // StorageManager_QueueCollection_h 

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
