// $Id$

#ifndef StorageManager_MonitorCollection_h
#define StorageManager_MonitorCollection_h

#include <list>
#include <vector>
#include <utility>
#include <string>

#include "xdaq/Application.h"
#include "xdata/InfoSpace.h"
#include "xdata/Serializable.h"

#include "EventFilter/StorageManager/interface/MonitoredQuantity.h"


namespace stor {

  /**
   * An abstract collection of MonitoredQuantities
   *
   * $Author$
   * $Revision$
   * $Date$
   */
  
  class MonitorCollection
  {
  public:

    explicit MonitorCollection(xdaq::Application*);

    // A pure virtual destructor results in a missing symbol
    virtual ~MonitorCollection() {};

    /**
     * Calculates the statistics and updates the info space
     * for all monitored quantities
     */
    void update();

    /**
     * Calculates the statistics for all quantities
     */
    void calculateStatistics();

    /**
     * Updates the infospace
     */
    void updateInfoSpace();

    /**
     * Resets the monitored quantities
     */
    void reset();

    /**
     * Returns the pointer to the monitoring info space
     */
    xdata::InfoSpace* getMonitoringInfoSpace()
    { return _infoSpace; }

    
  protected:

    virtual void do_calculateStatistics() = 0;
    
    virtual void do_updateInfoSpace() = 0;

    virtual void do_reset() = 0;


    // Stuff dealing with info space
    void putItemsIntoInfoSpace();

    static xdata::InfoSpace *_infoSpace;

    typedef std::vector< std::pair<std::string, xdata::Serializable*> > infoSpaceItems_t;
    infoSpaceItems_t _infoSpaceItems;

    std::list<std::string> _infoSpaceItemNames;

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
