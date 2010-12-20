// $Id: ExpirableQueue.h,v 1.6 2010/12/14 12:56:51 mommsen Exp $
/// @file: ExpirableQueue.h 


#ifndef EventFilter_StorageManager_ExpirableQueue_h
#define EventFilter_StorageManager_ExpirableQueue_h

#include "EventFilter/StorageManager/interface/ConcurrentQueue.h"
#include "EventFilter/StorageManager/interface/EnquingPolicyTag.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/Utils.h"

namespace stor
{
  /**
     Class template ExpirableQueue encapsulates a Queue (held
     through a shared_ptr, for inexpensive copying) and timing
     information. It keeps track of when the most recent called to deq
     was made.
   
     $Author: mommsen $
     $Revision: 1.6 $
     $Date: 2010/12/14 12:56:51 $
   */

  template <class T, class Policy>
  class ExpirableQueue
  {
  public:
    typedef Policy policy_type; // publish template parameter
    typedef typename ConcurrentQueue<T, Policy>::size_type size_type;

    /**
       Create an ExpirableQueue with the given maximum size and
       given "time to stale", specified in seconds.
     */
    explicit ExpirableQueue(size_type maxsize=std::numeric_limits<size_type>::max(),
                            utils::duration_t staleness_interval = boost::posix_time::seconds(120),
                            utils::time_point_t now = utils::getCurrentTime());
    /**
      Try to remove an event from the queue, without blocking.
      If an event is available, return 'true' and set the output
      argument 'event'.
      If no event is available, return 'false'.
      In either case, update the staleness time to reflect this
      attempt to get an event.
    */
    bool deq_nowait(T& event);

    /**
       Put an event onto the queue, respecting the Policy of this
       queue that controls what is done in the case of a full queue.
       This does not affect the staleness time of this queue.
     */
    typename Policy::return_type enq_nowait(T const& event);

    /**
       Set the staleness interval.
     */
    void set_staleness_interval(utils::duration_t staleness_interval);

    /**
       Get the staleness interval.
    */
    utils::duration_t staleness_interval() const;    

    /**
       Clear the queue. Return the number of elements removed.
     */
    size_type clear();

    /**
       Return true if the queue is empty, and false otherwise.
     */
    bool empty() const;

    /**
       Return true if the queue is full, and false otherwise.
    */
    bool full() const;

    /**
       Get number of entries in queue
    */
    size_type size() const;

    /**
       Return true if the queue is stale, and false if it is not. The
       queue is stale if its staleness_time is before the given
       time. If the queue is stale, we also clear it and return the
       number of cleared events.
    */
    bool clearIfStale(utils::time_point_t now, size_type& clearedEvents);

  private:
    typedef ConcurrentQueue<T, Policy> queue_t;

    queue_t      _events;
    /**  Time in seconds it takes for this queue to become stale. */
    utils::duration_t   _staleness_interval;
    /** Point in time at which this queue will become stale. */
    utils::time_point_t _staleness_time;

    /*
      The following are not implemented, to prevent copying and
      assignment.
     */
    ExpirableQueue(ExpirableQueue&);
    ExpirableQueue& operator=(ExpirableQueue&);
  };

  
  template <class T, class Policy>
  ExpirableQueue<T, Policy>::ExpirableQueue(size_type maxsize,
                                            utils::duration_t staleness_interval,
                                            utils::time_point_t now) :
    _events(maxsize),
    _staleness_interval(staleness_interval),
    _staleness_time(now+_staleness_interval)
  {
  }

  template <class T, class Policy>
  bool
  ExpirableQueue<T, Policy>::deq_nowait(T& event)
  {
    _staleness_time = utils::getCurrentTime() + _staleness_interval;
    return _events.deq_nowait(event);
  }

  template <class T, class Policy>
  typename Policy::return_type
  ExpirableQueue<T, Policy>::enq_nowait(T const& event)
  {
    return _events.enq_nowait(event);
  }  

  template <class T, class Policy>
  inline
  void
  ExpirableQueue<T, Policy>::set_staleness_interval(utils::duration_t t)
  {
    _staleness_interval = t;
  }

  template <class T, class Policy>
  inline
  utils::duration_t
  ExpirableQueue<T, Policy>::staleness_interval() const
  {
    return _staleness_interval;
  }

  template <class T, class Policy>
  inline
  typename ExpirableQueue<T, Policy>::size_type
  ExpirableQueue<T, Policy>::clear()
  {
    const size_type entries = size();
    _events.clear();
    return entries;
  }  

  template <class T, class Policy>
  inline
  bool
  ExpirableQueue<T, Policy>::empty() const
  {
    return _events.empty();
  }

  template <class T, class Policy>
  inline
  typename ExpirableQueue<T, Policy>::size_type
  ExpirableQueue<T, Policy>::size() const
  {
    return _events.size();
  }

  template <class T, class Policy>
  inline
  bool
  ExpirableQueue<T, Policy>::full() const
  {
    return _events.full();
  }

  template <class T, class Policy>
  inline
  bool
  ExpirableQueue<T, Policy>::clearIfStale(utils::time_point_t now, size_type& clearedEvents)
  {
    if (_staleness_time < now)
    {
      clearedEvents = clear();
      return true;
    }
    else
    {
      clearedEvents = 0;
      return false;
    }
  }

} // namespace stor
  

#endif

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

