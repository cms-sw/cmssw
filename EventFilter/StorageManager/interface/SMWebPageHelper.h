// $Id: SMWebPageHelper.h,v 1.4 2011/11/18 14:47:56 mommsen Exp $
/// @file: SMWebPageHelper.h

#ifndef EventFilter_StorageManager_SMWebPageHelper_h
#define EventFilter_StorageManager_SMWebPageHelper_h

#include <string>
#include <map>

#include "toolbox/mem/Pool.h"
#include "xdaq/ApplicationDescriptor.h"
#include "xgi/Output.h"

#include "EventFilter/Utilities/interface/Css.h"

#include "EventFilter/StorageManager/interface/ConsumerWebPageHelper.h"
#include "EventFilter/StorageManager/interface/RegistrationCollection.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/interface/Utils.h"
#include "EventFilter/StorageManager/interface/WebPageHelper.h"
#include "EventFilter/StorageManager/interface/XHTMLMaker.h"


namespace stor {

  class DQMEventMonitorCollection;
  class FilesMonitorCollection;
  class FragmentMonitorCollection;
  class ResourceMonitorCollection;
  class RunMonitorCollection;
  class StreamsMonitorCollection;


  /**
   * Helper class to handle web page requests
   *
   * $Author: mommsen $
   * $Revision: 1.4 $
   * $Date: 2011/11/18 14:47:56 $
   */
  
  class SMWebPageHelper : public WebPageHelper<SMWebPageHelper>
  {
  public:

    SMWebPageHelper
    (
      xdaq::ApplicationDescriptor*,
      SharedResourcesPtr
    );

    /**
     * Generates the default monitoring webpage
     */
    void defaultWebPage(xgi::Output*) const;

    /**
     * Generates the I2O input monitoring webpage
     */
    void inputWebPage(xgi::Output*) const;

    /**
     * Generates the output streams monitoring webpage
     */
    void storedDataWebPage(xgi::Output*) const;

    /**
     * Generates the files monitoring webpage
     */
    void filesWebPage(xgi::Output*) const;

    /**
       Generates consumer statistics page
    */
    void consumerStatistics(xgi::Output*) const;

    /**
       Generates the data sender web page for all resource brokers
    */
    void resourceBrokerOverview(xgi::Output*) const;

    /**
       Generates the data sender web page for a specific resource broker
    */
    void resourceBrokerDetail(xgi::Output*, const long long& uniqueRBID) const;

    /**
     * Generates the DQM event processor monitoring webpage
     */
    void dqmEventWebPage(xgi::Output*) const;

    /**
     * Generates the throughput monitoring webpage
     */
    void throughputWebPage(xgi::Output*) const;
        
    
  private:

    /**
     * Returns the webpage body with the standard header as XHTML node
     */
    XHTMLMaker::Node* createWebPageBody
    (
      XHTMLMaker&,
      const std::string& pageTitle,
      const StateMachineMonitorCollection&
    ) const;
    
    /**
     * Adds the links for the other hyperdaq webpages
     */
    void addDOMforHyperLinks(XHTMLMaker&, XHTMLMaker::Node* parent) const;

    /**
     * Adds the summary information to the parent DOM element
     */
    void addDOMforSummaryInformation
    (
      XHTMLMaker&,
      XHTMLMaker::Node* parent,
      DataSenderMonitorCollection const&,
      StreamsMonitorCollection const&,
      EventConsumerMonitorCollection const&,
      DQMEventMonitorCollection const&,
      RegistrationCollectionPtr
    ) const;

    /**
     * Adds the resource table to the parent DOM element
     */
    void addDOMforResourceUsage
    (
      XHTMLMaker&,
      XHTMLMaker::Node* parent,
      ResourceMonitorCollection const&,
      ThroughputMonitorCollection const&
    ) const;

    /**
     * Adds fragment monitoring statistics to the parent DOM element
     */
    void addDOMforFragmentMonitor
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      FragmentMonitorCollection const&
    ) const;

    /**
     * Adds run monitoring statistics to the parent DOM element
     */
    void addDOMforRunMonitor
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      RunMonitorCollection const&
    ) const;

    /**
     * Adds stored data statistics to the parent DOM element
     */
    void addDOMforStoredData
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      StreamsMonitorCollection const&
    ) const;

    /**
     * Adds the SM config string to the parent DOM element
     */
    void addDOMforConfigString
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      DiskWritingParams const&
    ) const;

    /**
     * Adds files statistics to the parent DOM element
     */
    void addDOMforFiles
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      FilesMonitorCollection const&
    ) const;

    /**
     * Adds throughput statistics to the parent DOM element
     */
    void addDOMforThroughputStatistics
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      ThroughputMonitorCollection const&
    ) const;
    
    /**
     * Return the aggregated bandwidth of data served to the
     * event consumers for the given output module label
     */
    double getServedConsumerBandwidth
    (
      const std::string& label,
      RegistrationCollectionPtr,
      const EventConsumerMonitorCollection& eventConsumerCollection
    ) const;
    
    /**
     * Add table row using the snapshot values
     */
    void addRowForThroughputStatistics
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      const ThroughputMonitorCollection::Stats::Snapshot&,
      bool const isAverage = false
    ) const;

    /**
     * List stream records statistics
     */
    void listStreamRecordsStats
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      StreamsMonitorCollection const&,
      const MonitoredQuantity::DataSetType
    ) const;

    /**
     * Add statistics for received fragments
     */
    void addFragmentStats
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      FragmentMonitorCollection::FragmentStats const&,
      const MonitoredQuantity::DataSetType
    ) const;
    
    /**
     * Add a table row for number of fragment frames received
     */
    void addRowForFramesReceived
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      FragmentMonitorCollection::FragmentStats const&,
      const MonitoredQuantity::DataSetType
    ) const;

    /**
     * Add a table row for fragment bandwidth
     */
    void addRowForBandwidth
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      FragmentMonitorCollection::FragmentStats const&,
      const MonitoredQuantity::DataSetType
    ) const;

    /**
     * Add a table row for fragment rate
     */
    void addRowForRate
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      FragmentMonitorCollection::FragmentStats const&,
      const MonitoredQuantity::DataSetType
    ) const;

    /**
     * Add a table row for fragment latency
     */
    void addRowForLatency
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      FragmentMonitorCollection::FragmentStats const&,
      const MonitoredQuantity::DataSetType
    ) const;

    /**
     * Add a table row for total fragment volume received
     */
    void addRowForTotalVolume
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      FragmentMonitorCollection::FragmentStats const&,
      const MonitoredQuantity::DataSetType
    ) const;

    /**
     * Add a table row for maximum fragment bandwidth
     */
    void addRowForMaxBandwidth
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      FragmentMonitorCollection::FragmentStats const&,
      const MonitoredQuantity::DataSetType
    ) const;

    /**
     * Add a table row for minimum fragment bandwidth
     */
    void addRowForMinBandwidth
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      FragmentMonitorCollection::FragmentStats const&,
      const MonitoredQuantity::DataSetType
    ) const;

    /**
     * Adds top-level output module statistics to the parent DOM element
     */
    void addOutputModuleTables
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      DataSenderMonitorCollection const&
    ) const;

    /**
     * Adds output module statistics from the specified resource
     * broker to the parent DOM element
     */
    void addOutputModuleStatistics
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      long long uniqueRBID,
      DataSenderMonitorCollection const&
    ) const;

    /**
     * Adds output module statistics to the parent DOM element
     */
    void addOutputModuleStatistics
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      DataSenderMonitorCollection::OutputModuleResultsList const&
    ) const;

    /**
     * Adds output module summary information to the parent DOM element
     */
    void addOutputModuleSummary
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      DataSenderMonitorCollection::OutputModuleResultsList const&
    ) const;

    /**
     * Adds the list of data senders (resource brokers) to the
     * parent DOM element
     */
    void addResourceBrokerList
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      DataSenderMonitorCollection const&
    ) const;

    /**
     * Adds information about a specific resource broker to the
     * parent DOM element
     */
    void addResourceBrokerDetails
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      long long uniqueRBID,
      DataSenderMonitorCollection const&
    ) const;

    /**
     * Adds information about the filter units for a specific
     * resource broker to the parent DOM element
     */
    void addFilterUnitList
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      long long uniqueRBID,
      DataSenderMonitorCollection const&
    ) const;

    /**
     * Add a table for resource usage
     */
    void addTableForResourceUsages
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      ThroughputMonitorCollection::Stats const&,
      ResourceMonitorCollection::Stats const&
    ) const;

    /**
     * Add a table rows for throughput usage summary
     */
    void addRowsForThroughputUsage
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      ThroughputMonitorCollection::Stats const&
    ) const;

    /**
     * Add a table row for copy/inject workers
     */
    void addRowsForWorkers
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      ResourceMonitorCollection::Stats const&
    ) const;

    /**
     * Add a table row for SATA beast status
     */
    void addRowsForSataBeast
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      ResourceMonitorCollection::Stats const&
    ) const;

    /**
     * Add a table for disk usage
     */
    void addTableForDiskUsages
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      ResourceMonitorCollection::Stats const&
    ) const;


  private:

    //Prevent copying of the SMWebPageHelper
    SMWebPageHelper(SMWebPageHelper const&);
    SMWebPageHelper& operator=(SMWebPageHelper const&);

    SharedResourcesPtr sharedResources_;

    typedef ConsumerWebPageHelper<SMWebPageHelper,
                                  EventQueueCollection,
                                  StatisticsReporter> ConsumerWebPageHelper_t;
    ConsumerWebPageHelper_t consumerWebPageHelper_;

  };

} // namespace stor

#endif // EventFilter_StorageManager_SMWebPageHelper_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
