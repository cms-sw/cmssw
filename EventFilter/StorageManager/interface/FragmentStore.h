// $Id: FragmentStore.h,v 1.11 2011/08/31 20:11:59 wmtan Exp $
/// @file: FragmentStore.h 

#ifndef EventFilter_StorageManager_FragmentStore_h
#define EventFilter_StorageManager_FragmentStore_h

#include <map>

#include "EventFilter/StorageManager/interface/FragKey.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/Utils.h"


namespace stor {
  
  /**
   * Stores incomplete events
   *
   * Uses a map of I2OChains to store incomplete events.
   *
   * $Author: wmtan $
   * $Revision: 1.11 $
   * $Date: 2011/08/31 20:11:59 $
   */
  
  class FragmentStore
  {
  public:
    
    explicit FragmentStore(size_t maxMemoryUsageMB);

    /**
     * Adds fragments of the I2OChain to the fragment store.
     * If the passed fragments completes an event, it returns true.
     * In this case, the passed I2OChain contains the completed event.
     * Otherwise, it returns false and the I2OChain is empty.
     */
    const bool addFragment(I2OChain&);


    /**
     * Add the duration to the stale window start time for
     * all I2OChains hold by the store.
     */
    void addToStaleEventTimes(const utils::Duration_t);


    /**
     * Resets the stale window start time for all I2OChains hold by
     * the store.
     */
    void resetStaleEventTimes();


    /**
     * Checks for event fragments for which the last event fragment
     * was added longer than timeout seconds ago. If it finds one
     * it returns true and the I2OChain contains the faulty event.
     * Otherwise it returns false and the I2OChain is empty.
     */
    const bool getStaleEvent(I2OChain&, utils::Duration_t timeout);


    /**
     * Clears all fragments hold by the fragment store
     */
    inline void clear()
    { store_.clear(); memoryUsed_ = 0; }


    /**
     * Checks if the fragment store is empty
     */
    inline bool empty() const
    { return store_.empty(); }


    /**
     * Checks if the fragment store is full
     */
    inline bool full() const
    { return (memoryUsed_ >= maxMemoryUsage_); }


    /**
     * Returns the number of events in the fragment store (complete or not).
     */
    inline unsigned int size() const
    { return store_.size(); }


    /**
     * Returns the total memory occupied by the events in the fragment store
     */
    inline size_t memoryUsed() const
    { return memoryUsed_; }

    
  private:

    //Prevent copying of the FragmentStore
    FragmentStore(FragmentStore const&);
    FragmentStore& operator=(FragmentStore const&);

    typedef std::map<FragKey, I2OChain> fragmentMap;
    fragmentMap store_;
    
    size_t memoryUsed_;
    const size_t maxMemoryUsage_;
  };
  
} // namespace stor

#endif // EventFilter_StorageManager_FragmentStore_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
