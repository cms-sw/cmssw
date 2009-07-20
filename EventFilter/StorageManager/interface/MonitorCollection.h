// $Id: MonitorCollection.h,v 1.3 2009/07/09 15:34:44 mommsen Exp $
/// @file: MonitorCollection.h 

#ifndef StorageManager_MonitorCollection_h
#define StorageManager_MonitorCollection_h

#include "xdata/Serializable.h"

#include "EventFilter/StorageManager/interface/MonitoredQuantity.h"

#include <string>


namespace stor {

  /**
   * An abstract collection of MonitoredQuantities
   *
   * $Author: mommsen $
   * $Revision: 1.3 $
   * $Date: 2009/07/09 15:34:44 $
   */
  
  class MonitorCollection
  {
  public:

    typedef std::vector< std::pair<std::string, xdata::Serializable*> > InfoSpaceItems;


    MonitorCollection() {};

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
    void calculateStatistics();

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
    void reset();

    
  protected:

    virtual void do_calculateStatistics() = 0;
    virtual void do_reset() = 0;
    virtual void do_appendInfoSpaceItems(InfoSpaceItems&) {};
    virtual void do_updateInfoSpaceItems() {};


  private:

    //Prevent copying of the MonitorCollection
    MonitorCollection(MonitorCollection const&);
    MonitorCollection& operator=(MonitorCollection const&);

  };
  
} // namespace stor

#endif // StorageManager_MonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
