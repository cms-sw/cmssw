// $Id: MonitorCollection.h,v 1.6 2011/03/07 15:31:32 mommsen Exp $
/// @file: MonitorCollection.h 

#ifndef EventFilter_StorageManager_MonitorCollection_h
#define EventFilter_StorageManager_MonitorCollection_h

#include "xdata/Serializable.h"

#include "EventFilter/StorageManager/interface/MonitoredQuantity.h"
#include "EventFilter/StorageManager/interface/Utils.h"

#include <string>


namespace stor {

  /**
   * An abstract collection of MonitoredQuantities
   *
   * $Author: mommsen $
   * $Revision: 1.6 $
   * $Date: 2011/03/07 15:31:32 $
   */
  
  class MonitorCollection
  {
  public:

    typedef std::vector< std::pair<std::string, xdata::Serializable*> > InfoSpaceItems;


    explicit MonitorCollection(const utils::Duration_t& updateInterval);


    // A pure virtual destructor results in a missing symbol
    virtual ~MonitorCollection() {};

    /**
     * Append the info space items to be published in the 
     * monitoring info space to the InfoSpaceItems
     */
    void appendInfoSpaceItems(InfoSpaceItems&);

    /**
     * Calculates the statistics for all quantities
     */
    void calculateStatistics(const utils::TimePoint_t& now);

    /**
     * Update all values of the items put into the monitoring
     * info space. The caller has to make sure that the info
     * space where the items reside is locked and properly unlocked
     * after the call.
     */
    void updateInfoSpaceItems();

    /**
     * Resets the monitored quantities
     */
    void reset(const utils::TimePoint_t& now);

    
  protected:

    virtual void do_calculateStatistics() = 0;
    virtual void do_reset() = 0;
    virtual void do_appendInfoSpaceItems(InfoSpaceItems&) {};
    virtual void do_updateInfoSpaceItems() {};


  private:

    //Prevent copying of the MonitorCollection
    MonitorCollection(MonitorCollection const&);
    MonitorCollection& operator=(MonitorCollection const&);

    const utils::Duration_t updateInterval_;
    utils::TimePoint_t lastCalculateStatistics_;
    bool infoSpaceUpdateNeeded_;

  };
  
} // namespace stor

#endif // EventFilter_StorageManager_MonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
