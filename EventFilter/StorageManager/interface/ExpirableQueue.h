// $Id: ExpirableQueue.h,v 1.8 2011/03/07 15:31:31 mommsen Exp $
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
     $Revision: 1.8 $
     $Date: 2011/03/07 15:31:31 $
   */

  template <class T, class Policy>
  class ExpirableQueue
  {
  public:
    typedef Policy PolicyType; // publish template parameter
    typedef typename ConcurrentQueue<T, Policy>::SizeType SizeType;
    typedef typename Policy::ValueType ValueType;

    /**
       Create an ExpirableQueue with the given maximum size and
       given "time to stale", specified in seconds.
     */
    explicit ExpirableQueue
    (
      SizeType maxsize=std::numeric_limits<SizeType>::max(),
      utils::Duration_t stalenessInterval = boost::posix_time::seconds(120),
      utils::TimePoint_t now = utils::getCurrentTime()
    );
    /**
      Try to remove an event from the queue, without blocking.
      If an event is available, return 'true' and set the output
      argument 'event'.
      If no event is available, return 'false'.
      In either case, update the staleness time to reflect this
      attempt to get an event.
    */
    bool deqNowait(ValueType&);

    /**
       Put an event onto the queue, if the queue is not stale at the
       given time. It respects the Policy of this queue that controls
       what is done in the case of a full queue.
       This does not affect the staleness time of this queue.
       Returns the number of dropped events.
     */
    SizeType enqNowait
    (
      T const&,
      const utils::TimePoint_t& now = utils::getCurrentTime()
    );

    /**
       Set the staleness interval.
     */
    void setStalenessInterval(const utils::Duration_t&);

    /**
       Get the staleness interval.
    */
    utils::Duration_t stalenessInterval() const;    

    /**
       Clear the queue. Return the number of elements removed.
     */
    SizeType clear();

    /**
       Return true if the queue is empty, and false otherwise.
     */
    bool empty() const;

    /**
       Return true if the queue is full, and false otherwise.
    */
    bool full() const;

    /**
       Return true if the queue is stale at the given time, and false otherwise.
    */
    bool stale(const utils::TimePoint_t&) const;

    /**
       Get number of entries in queue
    */
    SizeType size() const;

    /**
       Return true if the queue is stale, and false if it is not. The
       queue is stale if its stalenessTime is before the given
       time. If the queue is stale, we also clear it and return the
       number of cleared events.
    */
    bool clearIfStale(const utils::TimePoint_t&, SizeType& clearedEvents);

  private:
    typedef ConcurrentQueue<T, Policy> queue_t;

    queue_t events_;
    /**  Time in seconds it takes for this queue to become stale. */
    utils::Duration_t stalenessInterval_;
    /** Point in time at which this queue will become stale. */
    utils::TimePoint_t stalenessTime_;

    /*
      The following are not implemented, to prevent copying and
      assignment.
     */
    ExpirableQueue(ExpirableQueue&);
    ExpirableQueue& operator=(ExpirableQueue&);
  };

  
  template <class T, class Policy>
  ExpirableQueue<T, Policy>::ExpirableQueue
  (
    SizeType maxsize,
    utils::Duration_t stalenessInterval,
    utils::TimePoint_t now
  ) :
    events_(maxsize),
    stalenessInterval_(stalenessInterval),
    stalenessTime_(now+stalenessInterval_)
  {
  }

  template <class T, class Policy>
  bool
  ExpirableQueue<T, Policy>::deqNowait(ValueType& event)
  {
    stalenessTime_ = utils::getCurrentTime() + stalenessInterval_;
    return events_.deqNowait(event);
  }

  template <class T, class Policy>
  typename ExpirableQueue<T, Policy>::SizeType
  ExpirableQueue<T, Policy>::enqNowait(T const& event, const utils::TimePoint_t& now)
  {
    if ( stale(now) )
    {
      events_.addExternallyDroppedEvents(1);
      return 1;
    }
    return events_.enqNowait(event);
  }  

  template <class T, class Policy>
  inline
  void
  ExpirableQueue<T, Policy>::setStalenessInterval(const utils::Duration_t& t)
  {
    stalenessInterval_ = t;
  }

  template <class T, class Policy>
  inline
  utils::Duration_t
  ExpirableQueue<T, Policy>::stalenessInterval() const
  {
    return stalenessInterval_;
  }

  template <class T, class Policy>
  inline
  typename ExpirableQueue<T, Policy>::SizeType
  ExpirableQueue<T, Policy>::clear()
  {
    return events_.clear();
  }  

  template <class T, class Policy>
  inline
  bool
  ExpirableQueue<T, Policy>::empty() const
  {
    return events_.empty();
  }

  template <class T, class Policy>
  inline
  typename ExpirableQueue<T, Policy>::SizeType
  ExpirableQueue<T, Policy>::size() const
  {
    return events_.size();
  }

  template <class T, class Policy>
  inline
  bool
  ExpirableQueue<T, Policy>::full() const
  {
    return events_.full();
  }

  template <class T, class Policy>
  inline
  bool
  ExpirableQueue<T, Policy>::stale(const utils::TimePoint_t& now) const
  {
    return (stalenessTime_ < now);
  }

  template <class T, class Policy>
  inline
  bool
  ExpirableQueue<T, Policy>::clearIfStale
  (
    const utils::TimePoint_t& now,
    SizeType& clearedEvents
  )
  {
    if (stalenessTime_ < now)
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
  

#endif // EventFilter_StorageManager_ExpirableQueue_h

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

