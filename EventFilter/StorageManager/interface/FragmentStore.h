// $Id: FragmentStore.h,v 1.4 2009/06/29 15:47:46 mommsen Exp $
/// @file: FragmentStore.h 

#ifndef StorageManager_FragmentStore_h
#define StorageManager_FragmentStore_h

#include <map>

#include "IOPool/Streamer/interface/HLTInfo.h"

#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/Utils.h"


namespace stor {
  
  /**
   * Stores incomplete events
   *
   * Uses a map of I2OChains to store incomplete events.
   *
   * $Author: mommsen $
   * $Revision: 1.4 $
   * $Date: 2009/06/29 15:47:46 $
   */
  
  class FragmentStore
  {
  public:
    
    FragmentStore() {};

    /**
     * Adds fragments of the I2OChain to the fragment store.
     * If the passed fragments completes an event, it returns true.
     * In this case, the passed I2OChain contains the completed event.
     * Otherwise, it returns false and the I2OChain is empty.
     */
    const bool addFragment(I2OChain&);


    /**
     * Add the duration_t in seconds to the stale window start time for
     * all I2OChains hold by the store.
     */
    void addToStaleEventTimes(const utils::duration_t);


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
    const bool getStaleEvent(I2OChain&, double timeout);


    /**
     * Clears all fragments hold by the fragment store
     */
    void clear()
    { _store.clear(); }


    /**
     * Checks if the fragment store is empty
     */
    bool empty()
    { return _store.empty(); }

    
  private:

    //Prevent copying of the FragmentStore
    FragmentStore(FragmentStore const&);
    FragmentStore& operator=(FragmentStore const&);

    typedef std::map<FragKey, I2OChain> fragmentMap;
    fragmentMap _store;
    
    
  };
  
} // namespace stor

#endif // StorageManager_FragmentStore_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
