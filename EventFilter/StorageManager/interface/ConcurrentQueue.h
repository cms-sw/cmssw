// $Id: ConcurrentQueue.h,v 1.15 2013/01/07 11:16:27 eulisse Exp $
/// @file: ConcurrentQueue.h 


#ifndef EventFilter_StorageManager_ConcurrentQueue_h
#define EventFilter_StorageManager_ConcurrentQueue_h

#include <algorithm>
#include <cstddef>
#include <exception>
#include <limits>
#include <list>

#include <iostream> // debugging

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/mutex.hpp>

namespace stor
{
  /**
     Class template ConcurrentQueue provides a FIFO that can be used
     to communicate data between multiple producer and consumer
     threads in an application.

     The template policy EnqPolicy determines the behavior of the
     enqNowait function. In all cases, this function will return
     promptly (that is, it will not wait for a full queue to become
     not-full). However, what is done in the case of the queue being
     full depends on the policy chosen:

        FailIfFull: a std::exeption is thrown if the queue is full.

        KeepNewest: the head of the FIFO is popped (and destroyed),
        and the new item is added to the FIFO. The function returns
        the number of popped (dropped) element.

        RejectNewest: the new item is not put onto the FIFO.
        The function returns the dropped event count (1) if the
        item cannot be added.
   
     $Author: eulisse $
     $Revision: 1.15 $
     $Date: 2013/01/07 11:16:27 $
   */


  namespace detail
  {
    typedef size_t MemoryType;

    /*
      This template is using SFINAE to figure out if the class used to
      instantiate the ConcurrentQueue template has a method memoryUsed
      returning the number of bytes occupied by the class itself.
    */
    template <typename T>
    class hasMemoryUsed
    {
      typedef char TrueType;
      struct FalseType{ TrueType _[2]; };
      
      template <MemoryType (T::*)() const>
      struct TestConst;
      
      template <typename C>
      static TrueType test( TestConst<&C::memoryUsed>* );
      template <typename C>
      static FalseType test(...);
      
    public:
      static const bool value = (sizeof(test<T>(0)) == sizeof(TrueType));
    };
    
    template <typename T>
    MemoryType 
    memoryUsage(const std::pair<T,size_t>& t)
    {
      MemoryType usage(0UL);
      try
      {
        usage = t.first.memoryUsed();
      }
      catch(...)
      {}
      return usage;
    }
    
    template <typename T>
    typename boost::enable_if<hasMemoryUsed<T>, MemoryType>::type
    memoryUsage(const T& t)
    {
      MemoryType usage(0UL);
      try
      {
        usage = t.memoryUsed();
      }
      catch(...)
      {}
      return usage;
    }
  
    template <typename T>
    typename boost::disable_if<hasMemoryUsed<T>, MemoryType>::type
    memoryUsage(const T& t)
    { return sizeof(T); }

  }// end namespace detail

  
  template <class T>
  struct FailIfFull
  {
    typedef void ReturnType;

    typedef T ValueType;
    typedef std::list<T> SequenceType;
    typedef typename SequenceType::size_type SizeType;

    static const struct QueueIsFull : public std::exception
    {
      QueueIsFull() {};
      virtual const char* what() const throw()
      {
        return "Cannot add item to a full queue";
      }
    } queueIsFull;

    static void doInsert
    (
      T const& item,
      SequenceType& elements,
      SizeType& size,
      detail::MemoryType const& itemSize,
      detail::MemoryType& used,
      boost::condition& nonempty
    )
    {
      elements.push_back(item);
      ++size;
      used += itemSize;
      nonempty.notify_one();
    }
               
    static ReturnType doEnq
    (
      T const& item,
      SequenceType& elements,
      SizeType& size,
      SizeType& capacity,
      detail::MemoryType& used,
      detail::MemoryType& memory,
      size_t& elementsDropped,
      boost::condition& nonempty
    )
    {
      detail::MemoryType itemSize = detail::memoryUsage(item);
      if (size >= capacity || used+itemSize > memory)
      {
        ++elementsDropped;
        throw queueIsFull;
      }
      else
      {
        doInsert(item, elements, size, itemSize, used, nonempty);
      }
    }         
  };
  template <class T>
  const typename FailIfFull<T>::QueueIsFull FailIfFull<T>::queueIsFull;

  template <class T>
  struct KeepNewest
  {
    typedef std::pair<T,size_t> ValueType;
    typedef std::list<T> SequenceType;
    typedef typename SequenceType::size_type SizeType;
    typedef SizeType ReturnType;

    static void doInsert
    (
      T const& item,
      SequenceType& elements,
      SizeType& size,
      detail::MemoryType const& itemSize,
      detail::MemoryType& used,
      boost::condition& nonempty
    )
    {
      elements.push_back(item);
      ++size;
      used += itemSize;
      nonempty.notify_one();
    }

    static ReturnType doEnq
    (
      T const& item,
      SequenceType& elements,
      SizeType& size,
      SizeType& capacity,
      detail::MemoryType& used,
      detail::MemoryType& memory,
      size_t& elementsDropped,
      boost::condition& nonempty
    )
    {
      SizeType elementsRemoved(0);
      detail::MemoryType itemSize = detail::memoryUsage(item);
      while ( (size==capacity || used+itemSize > memory) && !elements.empty() )
      {
        SequenceType holder;
        // Move the item out of elements in a manner that will not throw.
        holder.splice(holder.begin(), elements, elements.begin());
        // Record the change in the length of elements.
        --size;
        used -= detail::memoryUsage( holder.front() );
        ++elementsRemoved;
      }
      if (size < capacity && used+itemSize <= memory)
        // we succeeded to make enough room for the new element
      {
        doInsert(item, elements, size, itemSize, used, nonempty);
      }
      else
      {
        // we cannot add the new element
        ++elementsRemoved;
      }
      elementsDropped += elementsRemoved;
      return elementsRemoved;
    }
  };


  template <class T>
  struct RejectNewest
  {
    typedef std::pair<T,size_t> ValueType;
    typedef std::list<T> SequenceType;
    typedef typename SequenceType::size_type SizeType;
    typedef SizeType ReturnType;

    static void doInsert
    (
      T const& item,
      SequenceType& elements,
      SizeType& size,
      detail::MemoryType const& itemSize,
      detail::MemoryType& used,
      boost::condition& nonempty
    )
    {
      elements.push_back(item);
      ++size;
      used += itemSize;
      nonempty.notify_one();
    }

    static ReturnType doEnq
    (
      T const& item,
      SequenceType& elements,
      SizeType& size,
      SizeType& capacity,
      detail::MemoryType& used,
      detail::MemoryType& memory,
      size_t& elementsDropped,
      boost::condition& nonempty
    )
    {
      detail::MemoryType itemSize = detail::memoryUsage(item);
      if (size < capacity && used+itemSize <= memory)
      {
        doInsert(item, elements, size, itemSize, used, nonempty);
        return 0;
      }
      ++elementsDropped;
      return 1;
    }
  };

  /**
     ConcurrentQueue<T> class template declaration.
   */

  template <class T, class EnqPolicy=FailIfFull<T> >
  class ConcurrentQueue
  {
  public:
    typedef typename EnqPolicy::ValueType ValueType;
    typedef typename EnqPolicy::SequenceType SequenceType;
    typedef typename SequenceType::size_type SizeType;

    /**
       ConcurrentQueue is always bounded. By default, the bound is
       absurdly large.
    */
    explicit ConcurrentQueue
    (
      SizeType maxSize = std::numeric_limits<SizeType>::max(),
      detail::MemoryType maxMemory = std::numeric_limits<detail::MemoryType>::max()
    );

    /**
       Applications should arrange to make sure that the destructor of
       a ConcurrentQueue is not called while some other thread is
       using that queue. There is some protection against doing this,
       but it seems impossible to make sufficient protection.
     */
    ~ConcurrentQueue();

    /**
       Copying a ConcurrentQueue is illegal, as is asigning to a
       ConcurrentQueue. The copy constructor and copy assignment
       operator are both private and unimplemented.
     */

    /**
       Add a copy if item to the queue, according to the rules
       determined by the EnqPolicy; see documentation above the the
       provided EnqPolicy choices.  This may throw any exception
       thrown by the assignment operator of type T, or badAlloc.
     */
    typename EnqPolicy::ReturnType enqNowait(T const& item);

    /**
       Add a copy of item to the queue. If the queue is full wait
       until it becomes non-full. This may throw any exception thrown
       by the assignment operator of type T, or badAlloc.
     */
    void enqWait(T const& p);

    /**
       Add a copy of item to the queue. If the queue is full wait
       until it becomes non-full or until timeDuration has passed.
       Return true if the items has been put onto the queue or
       false if the timeout has expired. This may throw any exception
       thrown by the assignment operator of T, or badAlloc.
     */
    bool enqTimedWait(T const& p, boost::posix_time::time_duration const&);

    /**
       Assign the value at the head of the queue to item and then
       remove the head of the queue. If successful, return true; on
       failure, return false. This function fill fail without waiting
       if the queue is empty. This function may throw any exception
       thrown by the assignment operator of type EnqPolicy::ValueType.
     */
    bool deqNowait(ValueType&);

    /**
       Assign the value of the head of the queue to item and then
       remove the head of the queue. If the queue is empty wait until
       is has become non-empty. This may throw any exception thrown by
       the assignment operator of type EnqPolicy::ValueType.
     */
    void deqWait(ValueType&);

    /**
       Assign the value at the head of the queue to item and then
       remove the head of the queue. If the queue is empty wait until
       is has become non-empty or until timeDuration has passed.
       Return true if an item has been removed from the queue
       or false if the timeout has expired. This may throw any
       exception thrown by the assignment operator of type EnqPolicy::ValueType.
     */
    bool deqTimedWait(ValueType&, boost::posix_time::time_duration const&);

    /**
       Return true if the queue is empty, and false if it is not.
     */
    bool empty() const;

    /**
       Return true if the queue is full, and false if it is not.
    */
    bool full() const;

    /**
       Return the size of the queue, that is, the number of items it
       contains.
     */
    SizeType size() const;

    /**
       Return the capacity of the queue, that is, the maximum number
       of items it can contain.
     */
    SizeType capacity() const;

    /**
       Reset the capacity of the queue. This can only be done if the
       queue is empty. This function returns false if the queue was
       not modified, and true if it was modified.
     */
    bool setCapacity(SizeType n);

    /**
       Return the memory in bytes used by items in the queue
     */
    detail::MemoryType used() const;

    /**
       Return the memory of the queue in bytes, that is, the maximum memory
       the items in the queue may occupy
     */
    detail::MemoryType memory() const;

    /**
       Reset the memory usage in bytes of the queue. A value of 0 disabled the
       memory check. This can only be done if the
       queue is empty. This function returns false if the queue was
       not modified, and true if it was modified.
     */
    bool setMemory(detail::MemoryType n);

    /**
       Remove all items from the queue. This changes the size to zero
       but does not change the capacity.
       Returns the number of cleared events.
     */
    SizeType clear();

    /**
       Adds the passed count to the counter of dropped events
     */
    void addExternallyDroppedEvents(SizeType);
    

  private:
    typedef boost::mutex::scoped_lock LockType;

    mutable boost::mutex protectElements_;
    mutable boost::condition queueNotEmpty_;
    mutable boost::condition queueNotFull_;

    SequenceType elements_;
    SizeType capacity_;
    SizeType size_;
    /*
      N.B.: we rely on SizeType *not* being some synthesized large
      type, so that reading the value is an atomic action, as is
      incrementing or decrementing the value. We do *not* assume that
      there is any atomic getAndIncrement or getAndDecrement
      operation.
    */
    detail::MemoryType memory_;
    detail::MemoryType used_;
    size_t elementsDropped_;

    /*
      These private member functions assume that whatever locks
      necessary for safe operation have already been obtained.
     */

    /*
      Insert the given item into the list, if it is not already full,
      and increment size. Return true if the item is inserted, and
      false if not.
    */
    bool insertIfPossible(T const& item);

    /*
      Remove the object at the head of the queue, if there is one, and
      assign item the value of this object.The assignment may throw an
      exception; even if it does, the head will have been removed from
      the queue, and the size appropriately adjusted. It is assumed
      the queue is nonempty. Return true if the queue was nonempty,
      and false if the queue was empty.
     */
    bool removeHeadIfPossible(ValueType& item);

    /*
      Remove the object at the head of the queue, and assign item the
      value of this object. The assignment may throw an exception;
      even if it does, the head will have been removed from the queue,
      and the size appropriately adjusted. It is assumed the queue is
      nonempty.
     */
    void removeHead(ValueType& item);

    void assignItem(T& item, const T& element);
    void assignItem(std::pair<T,size_t>& item, const T& element);

    /*
      Return false if the queue can accept new entries.
     */
    bool isFull() const;

    /*
      These functions are declared private and not implemented to
      prevent their use.
     */
    ConcurrentQueue(ConcurrentQueue<T,EnqPolicy> const&);
    ConcurrentQueue& operator=(ConcurrentQueue<T,EnqPolicy> const&);
  };

  //------------------------------------------------------------------
  // Implementation follows
  //------------------------------------------------------------------

  template <class T, class EnqPolicy>
  ConcurrentQueue<T,EnqPolicy>::ConcurrentQueue
  (
    SizeType maxSize,
    detail::MemoryType maxMemory
  ) :
    protectElements_(),
    elements_(),
    capacity_(maxSize),
    size_(0),
    memory_(maxMemory),
    used_(0),
    elementsDropped_(0)
  {}

  template <class T, class EnqPolicy>
  ConcurrentQueue<T,EnqPolicy>::~ConcurrentQueue()
  {
    LockType lock(protectElements_);
    elements_.clear();
    size_ = 0;
    used_ = 0;
    elementsDropped_ = 0;
  }

  template <class T, class EnqPolicy>
  typename EnqPolicy::ReturnType
  ConcurrentQueue<T,EnqPolicy>::enqNowait(T const& item)
  {
    LockType lock(protectElements_);
    return EnqPolicy::doEnq
      (item, elements_, size_, capacity_, used_, memory_,
        elementsDropped_, queueNotEmpty_);
  }

  template <class T, class EnqPolicy>
  void
  ConcurrentQueue<T,EnqPolicy>::enqWait(T const& item)
  {
    LockType lock(protectElements_);
    while ( isFull() ) queueNotFull_.wait(lock);
    EnqPolicy::doInsert(item, elements_, size_,
      detail::memoryUsage(item), used_, queueNotEmpty_);
  }

  template <class T, class EnqPolicy>
  bool
  ConcurrentQueue<T,EnqPolicy>::enqTimedWait
  (
    T const& item, 
    boost::posix_time::time_duration const& waitTime
  )
  {
    LockType lock(protectElements_);
    if ( isFull() )
    {
      queueNotFull_.timed_wait(lock, waitTime);
    }
    return insertIfPossible(item);
  }

  template <class T, class EnqPolicy>
  bool
  ConcurrentQueue<T,EnqPolicy>::deqNowait(ValueType& item)
  {
    LockType lock(protectElements_);
    return removeHeadIfPossible(item);
  }

  template <class T, class EnqPolicy>
  void
  ConcurrentQueue<T,EnqPolicy>::deqWait(ValueType& item)
  {
    LockType lock(protectElements_);
    while (size_ == 0) queueNotEmpty_.wait(lock);
    removeHead(item);
  }

  template <class T, class EnqPolicy>
  bool
  ConcurrentQueue<T,EnqPolicy>::deqTimedWait
  (
    ValueType& item,
    boost::posix_time::time_duration const& waitTime
  )
  {
    LockType lock(protectElements_);
    if (size_ == 0)
    {
      queueNotEmpty_.timed_wait(lock, waitTime);
    }
    return removeHeadIfPossible(item);
  }

  template <class T, class EnqPolicy>
  bool 
  ConcurrentQueue<T,EnqPolicy>::empty() const
  {
    // No lock is necessary: the read is atomic.
    return size_ == 0;
  }

  template <class T, class EnqPolicy>
  bool
  ConcurrentQueue<T,EnqPolicy>::full() const
  {
    LockType lock(protectElements_);
    return isFull();
  }

  template <class T, class EnqPolicy>
  typename ConcurrentQueue<T,EnqPolicy>::SizeType 
  ConcurrentQueue<T,EnqPolicy>::size() const
  {
    // No lock is necessary: the read is atomic.
    return size_;
  }

  template <class T, class EnqPolicy>
  typename ConcurrentQueue<T,EnqPolicy>::SizeType
  ConcurrentQueue<T,EnqPolicy>::capacity() const
  {
    // No lock is necessary: the read is atomic.
    return capacity_;
  }

  template <class T, class EnqPolicy>
  bool
  ConcurrentQueue<T,EnqPolicy>::setCapacity(SizeType newcapacity)
  {
    LockType lock(protectElements_);
    bool isEmpty = (size_ == 0);
    if (isEmpty) capacity_ = newcapacity;
    return isEmpty;
  }

  template <class T, class EnqPolicy>
  detail::MemoryType 
  ConcurrentQueue<T,EnqPolicy>::used() const
  {
    // No lock is necessary: the read is atomic.
    return used_;
  }

  template <class T, class EnqPolicy>
  detail::MemoryType
  ConcurrentQueue<T,EnqPolicy>::memory() const
  {
    // No lock is necessary: the read is atomic.
    return memory_;
  }

  template <class T, class EnqPolicy>
  bool
  ConcurrentQueue<T,EnqPolicy>::setMemory(detail::MemoryType newmemory)
  {
    LockType lock(protectElements_);
    bool isEmpty = (size_ == 0);
    if (isEmpty) memory_ = newmemory;
    return isEmpty;
  }

  template <class T, class EnqPolicy>
  typename ConcurrentQueue<T,EnqPolicy>::SizeType
  ConcurrentQueue<T,EnqPolicy>::clear()
  {
    LockType lock(protectElements_);
    SizeType clearedEvents = size_;
    elementsDropped_ += size_;
    elements_.clear();
    size_ = 0;
    used_ = 0;
    return clearedEvents;
  }
  
  template <class T, class EnqPolicy>
  void 
  ConcurrentQueue<T,EnqPolicy>::addExternallyDroppedEvents(SizeType n)
  {
    LockType lock(protectElements_);
    elementsDropped_ += n;
  }

  //-----------------------------------------------------------
  // Private member functions
  //-----------------------------------------------------------

  template <class T, class EnqPolicy>
  bool
  ConcurrentQueue<T,EnqPolicy>::insertIfPossible(T const& item)
  {
    if ( isFull() )
    {
      ++elementsDropped_;
      return false;
    }
    else
    {
      EnqPolicy::doInsert(item, elements_, size_,
      detail::memoryUsage(item), used_, queueNotEmpty_);
      return true;
    }
  }

  template <class T, class EnqPolicy>
  bool
  ConcurrentQueue<T,EnqPolicy>::removeHeadIfPossible(ValueType& item)
  {
    if (size_ == 0) return false;

    removeHead(item);
    return true;
  }

  template <class T, class EnqPolicy>
  void
  ConcurrentQueue<T,EnqPolicy>::removeHead(ValueType& item)
  {
    SequenceType holder;
    // Move the item out of elements_ in a manner that will not throw.
    holder.splice(holder.begin(), elements_, elements_.begin());
    // Record the change in the length of elements_.
    --size_;
    queueNotFull_.notify_one();

    assignItem(item, holder.front());
    used_ -= detail::memoryUsage( item );
  }
  
  template <class T, class EnqPolicy>
  void
  ConcurrentQueue<T,EnqPolicy>::assignItem(T& item, const T& element)
  {
    item = element;
  }
  
  template <class T, class EnqPolicy>
  void
  ConcurrentQueue<T,EnqPolicy>::assignItem(std::pair<T,size_t>& item, const T& element)
  {
    item.first = element;
    item.second = elementsDropped_;
    elementsDropped_ = 0;
  }
  
  template <class T, class EnqPolicy>
  bool
  ConcurrentQueue<T,EnqPolicy>::isFull() const
  {
    if (size_ >= capacity_ || used_ >= memory_) return true;
    return false;
  }

} // namespace stor

#endif // EventFilter_StorageManager_ConcurrentQueue_h

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

