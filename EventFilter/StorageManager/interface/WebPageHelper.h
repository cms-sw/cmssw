// $Id: WebPageHelper.h,v 1.2 2009/06/10 08:15:24 dshpakov Exp $
/// @file: WebPageHelper.h

#ifndef StorageManager_WebPageHelper_h
#define StorageManager_WebPageHelper_h

#include <string>

#include "boost/thread/mutex.hpp"

#include "toolbox/mem/Pool.h"
#include "xdaq/ApplicationDescriptor.h"
#include "xgi/Output.h"

#include "EventFilter/Utilities/interface/Css.h"

#include "EventFilter/StorageManager/interface/DQMEventMonitorCollection.h"
#include "EventFilter/StorageManager/interface/FilesMonitorCollection.h"
#include "EventFilter/StorageManager/interface/FragmentMonitorCollection.h"
#include "EventFilter/StorageManager/interface/ResourceMonitorCollection.h"
#include "EventFilter/StorageManager/interface/RunMonitorCollection.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/interface/StreamsMonitorCollection.h"
#include "EventFilter/StorageManager/interface/Utils.h"
#include "EventFilter/StorageManager/interface/XHTMLMaker.h"

namespace stor {

  /**
   * Helper class to handle web page requests
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:24 $
   */
  
  class WebPageHelper
  {
  public:

    WebPageHelper
    (
      xdaq::ApplicationDescriptor*,
      const std::string SMversion
    );


    /**
     * Create event filter style sheet
     */
    void css(xgi::Input *in, xgi::Output *out)
    { css_.css(in,out); }

    /**
     * Generates the default monitoring webpage
     */
    void defaultWebPage
    (
      xgi::Output*, 
      const SharedResourcesPtr
    );

    /**
     * Generates the output streams monitoring webpage
     */
    void storedDataWebPage
    (
      xgi::Output*,
      const SharedResourcesPtr
    );

    /**
     * Generates the files monitoring webpage
     */
    void filesWebPage
    (
      xgi::Output*,
      const SharedResourcesPtr
    );

    /**
       Generates consumer statistics page
    */
    void consumerStatistics( xgi::Output*,
                             const SharedResourcesPtr );

    /**
       Generates the data sender web page for all resource brokers
    */
    void resourceBrokerOverview( xgi::Output*,
                                 const SharedResourcesPtr );

    /**
       Generates the data sender web page for a specific resource broker
    */
    void resourceBrokerDetail( xgi::Output*,
                               const SharedResourcesPtr,
                               long long );

    /**
     * Generates the DQM event processor monitoring webpage
     */
    void dqmEventWebPage
    (
      xgi::Output*,
      const SharedResourcesPtr
    );

    /**
     * Generates the throughput monitoring webpage
     */
    void throughputWebPage
    (
      xgi::Output*,
      const SharedResourcesPtr
    );


  private:

    /**
      Get base url
    */
    std::string baseURL() const;

    /**
     * Returns the webpage body with the standard header as XHTML node
     */
    XHTMLMaker::Node* createWebPageBody(XHTMLMaker&, const StatisticsReporterPtr);

    /**
     * Adds the links for the other SM webpages
     */
    void addDOMforSMLinks(XHTMLMaker&, XHTMLMaker::Node *parent);

    /**
     * Adds the resource table to the parent DOM element
     */
    void addDOMforResourceUsage
    (
      XHTMLMaker&,
      XHTMLMaker::Node *parent,
      ResourceMonitorCollection const&
    );

    /**
     * Adds fragment monitoring statistics to the parent DOM element
     */
    void addDOMforFragmentMonitor
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *parent,
      FragmentMonitorCollection const&
    );

    /**
     * Adds run monitoring statistics to the parent DOM element
     */
    void addDOMforRunMonitor
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *parent,
      RunMonitorCollection const&
    );

    /**
     * Adds stored data statistics to the parent DOM element
     */
    void addDOMforStoredData
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *parent,
      StreamsMonitorCollection const&
    );

    /**
     * Adds the SM config string to the parent DOM element
     */
    void addDOMforConfigString
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *parent,
      DiskWritingParams const&
    );

    /**
     * Adds files statistics to the parent DOM element
     */
    void addDOMforFiles
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *parent,
      FilesMonitorCollection const&
    );

    /**
     * Adds DQM event processor statistics to the parent DOM element
     */
    void addDOMforProcessedDQMEvents
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *parent,
      DQMEventMonitorCollection const&
    );

    /**
     * Adds statistics for the DQM events to the parent DOM element
     */
    void addDOMforDQMEventStatistics
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *parent,
      DQMEventMonitorCollection const&
    );

    /**
     * Adds throughput statistics to the parent DOM element
     */
    void addDOMforThroughputStatistics
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *parent,
      ThroughputMonitorCollection const&
    );

    /**
     * List stream records statistics
     */
    void listStreamRecordsStats
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *table,
      StreamsMonitorCollection const&,
      const MonitoredQuantity::DataSetType
    );

    /**
     * Add statistics for received fragments
     */
    void addFragmentStats
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *table,
      FragmentMonitorCollection::FragmentStats const&,
      const MonitoredQuantity::DataSetType
    );

    /**
     * Add header with integration duration
     */
    void addDurationToTableHead
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *tableRow,
      const utils::duration_t
    );
    
    /**
     * Add a table row for number of fragment frames received
     */
    void addRowForFramesReceived
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *table,
      FragmentMonitorCollection::FragmentStats const&,
      const MonitoredQuantity::DataSetType
    );

    /**
     * Add a table row for fragment bandwidth
     */
    void addRowForBandwidth
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *table,
      FragmentMonitorCollection::FragmentStats const&,
      const MonitoredQuantity::DataSetType
    );

    /**
     * Add a table row for fragment rate
     */
    void addRowForRate
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *table,
      FragmentMonitorCollection::FragmentStats const&,
      const MonitoredQuantity::DataSetType
    );

    /**
     * Add a table row for fragment latency
     */
    void addRowForLatency
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *table,
      FragmentMonitorCollection::FragmentStats const&,
      const MonitoredQuantity::DataSetType
    );

    /**
     * Add a table row for total fragment volume received
     */
    void addRowForTotalVolume
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *table,
      FragmentMonitorCollection::FragmentStats const&,
      const MonitoredQuantity::DataSetType
    );

    /**
     * Add a table row for maximum fragment bandwidth
     */
    void addRowForMaxBandwidth
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *table,
      FragmentMonitorCollection::FragmentStats const&,
      const MonitoredQuantity::DataSetType
    );

    /**
     * Add a table row for minimum fragment bandwidth
     */
    void addRowForMinBandwidth
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *table,
      FragmentMonitorCollection::FragmentStats const&,
      const MonitoredQuantity::DataSetType
    );

    /**
     * Adds top-level output module statistics to the parent DOM element
     */
    void addOutputModuleTables
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *parent,
      DataSenderMonitorCollection const&
    );

    /**
     * Adds output module statistics from the specified resource
     * broker to the parent DOM element
     */
    void addOutputModuleStatistics
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *parent,
      long long uniqueRBID,
      DataSenderMonitorCollection const&
    );

    /**
     * Adds output module statistics to the parent DOM element
     */
    void addOutputModuleStatistics
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *parent,
      DataSenderMonitorCollection::OutputModuleResultsList const&
    );

    /**
     * Adds output module summary information to the parent DOM element
     */
    void addOutputModuleSummary
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *parent,
      DataSenderMonitorCollection::OutputModuleResultsList const&
    );

    /**
     * Adds the list of data senders (resource brokers) to the
     * parent DOM element
     */
    void addResourceBrokerList
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *parent,
      DataSenderMonitorCollection const&
    );

    /**
     * Adds information about a specific resource broker to the
     * parent DOM element
     */
    void addResourceBrokerDetails
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *parent,
      long long uniqueRBID,
      DataSenderMonitorCollection const&
    );

    /**
     * Adds information about the filter units for a specific
     * resource broker to the parent DOM element
     */
    void addFilterUnitList
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *parent,
      long long uniqueRBID,
      DataSenderMonitorCollection const&
    );

    /**
     * Add statistics for processed DQM events
     */
    void addDQMEventStats
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *table,
      DQMEventMonitorCollection::DQMEventStats const&,
      const MonitoredQuantity::DataSetType
    );
    
    /**
     * Add a table row for number of DQM events processed
     */
    void addRowForDQMEventsProcessed
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *table,
      DQMEventMonitorCollection::DQMEventStats const&,
      const MonitoredQuantity::DataSetType
    );

    /**
     * Add a table row for DQM event bandwidth
     */
    void addRowForDQMEventBandwidth
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *table,
      DQMEventMonitorCollection::DQMEventStats const&,
      const MonitoredQuantity::DataSetType
    );

    /**
     * Add a table row for total fragment volume received
     */
    void addRowForTotalDQMEventVolume
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *table,
      DQMEventMonitorCollection::DQMEventStats const&,
      const MonitoredQuantity::DataSetType
    );

    /**
     * Add a table row for maximum fragment bandwidth
     */
    void addRowForMaxDQMEventBandwidth
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *table,
      DQMEventMonitorCollection::DQMEventStats const&,
      const MonitoredQuantity::DataSetType
    );

    /**
     * Add a table row for minimum fragment bandwidth
     */
    void addRowForMinDQMEventBandwidth
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *table,
      DQMEventMonitorCollection::DQMEventStats const&,
      const MonitoredQuantity::DataSetType
    );

    /**
     * Add a table for resource usage
     */
    void addTableForResourceUsages
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *parent,
      ResourceMonitorCollection::Stats const&
    );

    /**
     * Add a table row for memory usage
     */
    void addRowsForMemoryUsage
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *table,
      ResourceMonitorCollection::Stats const&
    );

    /**
     * Add a table row for copy/inject workers
     */
    void addRowsForWorkers
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *table,
      ResourceMonitorCollection::Stats const&
    );

    /**
     * Add a table for disk usage
     */
    void addTableForDiskUsages
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node *parent,
      ResourceMonitorCollection::Stats const&
    );

    /**
     * Smooth out binned idle times for the throughput display.
     * Returns the index to be used for the next section to smooth.
     * Note that this method works on the idleTimes and durations
     * lists in *reverse* order.  So, the initial indices should be
     * idleTimes.size()-1.
     */
    int smoothIdleTimes
    (
      std::vector<double>& idleTimes,
      std::vector<utils::duration_t>& durations,
      int firstIndex, int lastIndex
    );


  private:

    //Prevent copying of the WebPageHelper
    WebPageHelper(WebPageHelper const&);
    WebPageHelper& operator=(WebPageHelper const&);

    evf::Css css_;

    static boost::mutex _xhtmlMakerMutex;
    xdaq::ApplicationDescriptor* _appDescriptor;
    const std::string _smVersion;

    XHTMLMaker::AttrMap _tableAttr;
    XHTMLMaker::AttrMap _rowAttr;
    XHTMLMaker::AttrMap _tableLabelAttr;
    XHTMLMaker::AttrMap _tableValueAttr;
    XHTMLMaker::AttrMap _specialRowAttr;

  };

} // namespace stor

#endif // StorageManager_WebPageHelper_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
