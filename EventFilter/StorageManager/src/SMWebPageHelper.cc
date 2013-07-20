// $Id: SMWebPageHelper.cc,v 1.8 2012/04/23 08:37:24 mommsen Exp $
/// @file: SMWebPageHelper.cc

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdio.h>

#include "boost/lexical_cast.hpp"

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/EventConsumerMonitorCollection.h"
#include "EventFilter/StorageManager/interface/FilesMonitorCollection.h"
#include "EventFilter/StorageManager/interface/FragmentMonitorCollection.h"
#include "EventFilter/StorageManager/interface/MonitoredQuantity.h"
#include "EventFilter/StorageManager/interface/RegistrationCollection.h"
#include "EventFilter/StorageManager/interface/ResourceMonitorCollection.h"
#include "EventFilter/StorageManager/interface/RunMonitorCollection.h"
#include "EventFilter/StorageManager/interface/StreamsMonitorCollection.h"
#include "EventFilter/StorageManager/interface/SMWebPageHelper.h"
#include "EventFilter/StorageManager/interface/XHTMLMonitor.h"
#include "EventFilter/StorageManager/src/ConsumerWebPageHelper.icc"

#include "toolbox/net/Utils.h"


namespace stor
{

  SMWebPageHelper::SMWebPageHelper
  (
    xdaq::ApplicationDescriptor* appDesc,
    SharedResourcesPtr sharedResources
  ) :
  WebPageHelper<SMWebPageHelper>(appDesc, "$Name: CMSSW_6_2_0 $", this, &stor::SMWebPageHelper::addDOMforHyperLinks),
  sharedResources_(sharedResources),
  consumerWebPageHelper_(appDesc, "$Name: CMSSW_6_2_0 $", this, &stor::SMWebPageHelper::addDOMforHyperLinks)
  {}


  void SMWebPageHelper::defaultWebPage(xgi::Output *out) const
  {
    XHTMLMonitor theMonitor;
    XHTMLMaker maker;
    
    StatisticsReporterPtr statReporter = sharedResources_->statisticsReporter_;
    
    // Create the body with the standard header
    XHTMLMaker::Node* body = createWebPageBody(maker, "Main",
      statReporter->getStateMachineMonitorCollection());
    
    // Run and event summary
    addDOMforRunMonitor(maker, body, statReporter->getRunMonitorCollection());
    
    // Resource usage
    addDOMforResourceUsage(maker, body, 
      statReporter->getResourceMonitorCollection(),
      statReporter->getThroughputMonitorCollection());
    
    // Summary
    addDOMforSummaryInformation(maker, body,
      statReporter->getDataSenderMonitorCollection(),
      statReporter->getStreamsMonitorCollection(),
      statReporter->getEventConsumerMonitorCollection(),
      statReporter->getDQMEventMonitorCollection(),
      sharedResources_->registrationCollection_);
    
    addDOMforHyperLinks(maker, body);
    
    // Dump the webpage to the output stream
    maker.out(*out);
  }
  
  
  void SMWebPageHelper::inputWebPage(xgi::Output *out) const
  {
    XHTMLMonitor theMonitor;
    XHTMLMaker maker;

    StatisticsReporterPtr statReporter = sharedResources_->statisticsReporter_;
    
    // Create the body with the standard header
    XHTMLMaker::Node* body = createWebPageBody(maker, "Input",
      statReporter->getStateMachineMonitorCollection());
    
    // Add the received data statistics table
    addDOMforFragmentMonitor(maker, body,
      statReporter->getFragmentMonitorCollection());
    
    addDOMforHyperLinks(maker, body);
    
    // Dump the webpage to the output stream
    maker.out(*out);
  }
  
  
  void SMWebPageHelper::storedDataWebPage(xgi::Output *out) const
  {
    XHTMLMonitor theMonitor;
    XHTMLMaker maker;
    
    StatisticsReporterPtr statReporter = sharedResources_->statisticsReporter_;
    
    // Create the body with the standard header
    XHTMLMaker::Node* body = createWebPageBody(maker, "Stored Data",
      statReporter->getStateMachineMonitorCollection());
    
    addDOMforStoredData(maker, body, statReporter->getStreamsMonitorCollection());
    
    maker.addNode("hr", body);
    
    addDOMforConfigString(maker, body, sharedResources_->configuration_->getDiskWritingParams());  
    
    addDOMforHyperLinks(maker, body);
    
    // Dump the webpage to the output stream
    maker.out(*out);
  }
  
  
  void SMWebPageHelper::filesWebPage(xgi::Output *out) const
  {
    XHTMLMonitor theMonitor;
    XHTMLMaker maker;
    
    StatisticsReporterPtr statReporter = sharedResources_->statisticsReporter_;
    
    // Create the body with the standard header
    XHTMLMaker::Node* body = createWebPageBody(maker, "Files",
      statReporter->getStateMachineMonitorCollection());
    
    addDOMforFiles(maker, body, statReporter->getFilesMonitorCollection());  
    
    addDOMforHyperLinks(maker, body);
    
    // Dump the webpage to the output stream
    maker.out(*out);
  }
  
  
  void SMWebPageHelper::consumerStatistics(xgi::Output *out) const
  {
    const StateMachineMonitorCollection& stateMachineMonitorCollection =
      sharedResources_->statisticsReporter_->getStateMachineMonitorCollection();

    std::string errorMsg;
    stateMachineMonitorCollection.statusMessage(errorMsg);

    consumerWebPageHelper_.consumerStatistics(out,
      stateMachineMonitorCollection.externallyVisibleState(),
      stateMachineMonitorCollection.innerStateName(),
      errorMsg,
      sharedResources_->statisticsReporter_,
      sharedResources_->registrationCollection_,
      sharedResources_->eventQueueCollection_,
      sharedResources_->dqmEventQueueCollection_
    );
  }
  
  
  void SMWebPageHelper::resourceBrokerOverview(xgi::Output *out) const
  {
    XHTMLMonitor theMonitor;
    XHTMLMaker maker;
    
    StatisticsReporterPtr statReporter = sharedResources_->statisticsReporter_;
    
    // Create the body with the standard header
    XHTMLMaker::Node* body = createWebPageBody(maker, "Resource Broker Overview",
      statReporter->getStateMachineMonitorCollection());
    
    addOutputModuleTables(maker, body,
      statReporter->getDataSenderMonitorCollection());  
    
    maker.addNode("hr", body);
    
    addResourceBrokerList(maker, body,
      statReporter->getDataSenderMonitorCollection());
    
    addDOMforHyperLinks(maker, body);
    
    // Dump the webpage to the output stream
    maker.out(*out);
  }
  
  
  void SMWebPageHelper::resourceBrokerDetail
  (
    xgi::Output *out,
    const long long& uniqueRBID
  ) const
  {
    XHTMLMonitor theMonitor;
    XHTMLMaker maker;
    
    StatisticsReporterPtr statReporter = sharedResources_->statisticsReporter_;
    
    // Create the body with the standard header
    std::ostringstream pageTitle;
    pageTitle << "Resource Broker " << uniqueRBID << " Detail";
    XHTMLMaker::Node* body = createWebPageBody(maker, pageTitle.str(),
      statReporter->getStateMachineMonitorCollection());
    
    addResourceBrokerDetails(maker, body, uniqueRBID,
      statReporter->getDataSenderMonitorCollection());  
    
    addOutputModuleStatistics(maker, body, uniqueRBID,
      statReporter->getDataSenderMonitorCollection());  
    
    addFilterUnitList(maker, body, uniqueRBID,
      statReporter->getDataSenderMonitorCollection());  
    
    addDOMforHyperLinks(maker, body);
    
    // Dump the webpage to the output stream
    maker.out(*out);
  }
  
  
  void SMWebPageHelper::dqmEventWebPage(xgi::Output* out) const
  {
    XHTMLMonitor theMonitor;
    XHTMLMaker maker;
    
    StatisticsReporterPtr statReporter = sharedResources_->statisticsReporter_;
    
    // Create the body with the standard header
    XHTMLMaker::Node* body = createWebPageBody(maker, "DQM Event Processor",
      statReporter->getStateMachineMonitorCollection());
    
    addDOMforProcessedDQMEvents(maker, body, statReporter->getDQMEventMonitorCollection());  
    addDOMforDQMEventStatistics(maker, body, statReporter->getDQMEventMonitorCollection());  
    
    addDOMforHyperLinks(maker, body);
    
    // Dump the webpage to the output stream
    maker.out(*out);
  }
  
  
  void SMWebPageHelper::throughputWebPage(xgi::Output *out) const
  {
    XHTMLMonitor theMonitor;
    XHTMLMaker maker;
    
    StatisticsReporterPtr statReporter = sharedResources_->statisticsReporter_;
    
    // Create the body with the standard header
    XHTMLMaker::Node* body = createWebPageBody(maker, "Throughput",
      statReporter->getStateMachineMonitorCollection());
    
    addDOMforThroughputStatistics(maker, body, statReporter->getThroughputMonitorCollection());  
    
    addDOMforHyperLinks(maker, body);
    
    // Dump the webpage to the output stream
    maker.out(*out);
  }
  
  
  XHTMLMaker::Node* SMWebPageHelper::createWebPageBody
  (
    XHTMLMaker& maker,
    const std::string& pageTitle,
    const StateMachineMonitorCollection& stateMachineMonitorCollection
  ) const
  {
    std::string errorMsg;
    stateMachineMonitorCollection.statusMessage(errorMsg);
    
    return WebPageHelper<SMWebPageHelper>::createWebPageBody(
      maker,
      pageTitle,
      stateMachineMonitorCollection.externallyVisibleState(),
      stateMachineMonitorCollection.innerStateName(),
      errorMsg
    );
  }
  
  
  void SMWebPageHelper::addDOMforHyperLinks
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *parent
  ) const
  {
    std::string url = appDescriptor_->getContextDescriptor()->getURL()
      + "/" + appDescriptor_->getURN();
    
    XHTMLMaker::AttrMap linkAttr;
    XHTMLMaker::Node *link;
    
    maker.addNode("hr", parent);
    
    linkAttr[ "href" ] = url;
    link = maker.addNode("a", parent, linkAttr);
    maker.addText(link, "Main web page");
    
    maker.addNode("hr", parent);
    
    linkAttr[ "href" ] = url + "/input";
    link = maker.addNode("a", parent, linkAttr);
    maker.addText(link, "I2O input web page");
    
    maker.addNode("hr", parent);
    
    linkAttr[ "href" ] = url + "/storedData";
    link = maker.addNode("a", parent, linkAttr);
    maker.addText(link, "Stored data web page");
    
    maker.addNode("hr", parent);
    
    linkAttr[ "href" ] = url + "/rbsenderlist";
    link = maker.addNode("a", parent, linkAttr);
    maker.addText(link, "RB Sender list web page");
    
    maker.addNode("hr", parent);
    
    linkAttr[ "href" ] = url + "/fileStatistics";
    link = maker.addNode("a", parent, linkAttr);
    maker.addText(link, "File Statistics web page");
    
    maker.addNode("hr", parent);
    
    linkAttr[ "href" ] = url + "/consumerStatistics";
    link = maker.addNode("a", parent, linkAttr);
    maker.addText(link, "Consumer Statistics");
    
    maker.addNode("hr", parent);
    
    linkAttr[ "href" ] = url + "/dqmEventStatistics";
    link = maker.addNode("a", parent, linkAttr);
    maker.addText(link, "DQM event processor statistics");
    
    maker.addNode("hr", parent);
    
    linkAttr[ "href" ] = url + "/throughputStatistics";
    link = maker.addNode("a", parent, linkAttr);
    maker.addText(link, "Throughput statistics");
    
    maker.addNode("hr", parent);
  }
  
  
  void SMWebPageHelper::addDOMforResourceUsage
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *parent,
    ResourceMonitorCollection const& rmc,
    ThroughputMonitorCollection const& tmc
  ) const
  {
    ResourceMonitorCollection::Stats rmcStats;
    rmc.getStats(rmcStats);
    
    ThroughputMonitorCollection::Stats tmcStats;
    tmc.getStats(tmcStats,10);
    
    XHTMLMaker::AttrMap halfWidthAttr;
    halfWidthAttr[ "width" ] = "50%";
    
    XHTMLMaker::Node* table = maker.addNode("table", parent, tableAttr_);
    
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    
    XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow, halfWidthAttr);
    addTableForResourceUsages(maker, tableDiv, tmcStats, rmcStats);
    
    tableDiv = maker.addNode("td", tableRow, halfWidthAttr);
    addTableForDiskUsages(maker, tableDiv, rmcStats);
  }
  
  
  void SMWebPageHelper::addTableForResourceUsages
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *parent,
    ThroughputMonitorCollection::Stats const& tmcStats,
    ResourceMonitorCollection::Stats const& rmcStats
  ) const
  {
    XHTMLMaker::AttrMap colspanAttr;
    colspanAttr[ "colspan" ] = "2";
    
    XHTMLMaker::Node* table = maker.addNode("table", parent, tableAttr_);
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
    maker.addText(tableDiv, "Resource Usage");
    
    addRowsForThroughputUsage(maker, table, tmcStats);
    addRowsForWorkers(maker, table, rmcStats);
    addRowsForSataBeast(maker, table, rmcStats);
  }
  
  
  void SMWebPageHelper::addRowsForThroughputUsage
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *table,
    ThroughputMonitorCollection::Stats const& stats
  ) const
  {
    XHTMLMaker::AttrMap tableLabelAttr = tableLabelAttr_;
    tableLabelAttr[ "width" ] = "54%";
    
    XHTMLMaker::AttrMap tableValueAttr = tableValueAttr_;
    tableValueAttr[ "width" ] = "46%";
    
    // Memory pool usage
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Memory pool used (kB)");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, stats.average.poolUsage / (double)0x400, 0 );

    // Input thread
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Input thread busy (%)");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, stats.average.fragmentProcessorBusy, 0 );

    // Disk writing thread
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Disk writing thread busy (%)");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, stats.average.diskWriterBusy, 0 );

    // DQM summing thread
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "DQM summing thread busy (%)");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, stats.average.dqmEventProcessorBusy, 0 );
  }
  
  
  void SMWebPageHelper::addRowsForWorkers
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *table,
    ResourceMonitorCollection::Stats const& stats
  ) const
  {
    XHTMLMaker::AttrMap tableValueAttr = tableValueAttr_;
    tableValueAttr[ "width" ] = "46%";
    
    // # copy worker
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "# CopyWorker");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addInt( tableDiv, stats.numberOfCopyWorkers );
    
    // # inject worker
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "# InjectWorker");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addInt( tableDiv, stats.numberOfInjectWorkers );
  }
  
  
  void SMWebPageHelper::addRowsForSataBeast
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *table,
    ResourceMonitorCollection::Stats const& stats
  ) const
  {
    XHTMLMaker::AttrMap tableValueAttr = tableValueAttr_;
    tableValueAttr[ "width" ] = "46%";
    
    XHTMLMaker::Node *tableRow, *tableDiv;
    
    XHTMLMaker::AttrMap warningAttr = rowAttr_;
    
    if (stats.sataBeastStatus < 0 )
    {
      warningAttr[ "bgcolor" ] = alarmColors_.find(AlarmHandler::WARNING)->second;
      
      XHTMLMaker::AttrMap colspanAttr = tableLabelAttr_;
      colspanAttr[ "colspan" ] = "2";
      
      tableRow = maker.addNode("tr", table, warningAttr);
      tableDiv = maker.addNode("td", tableRow, colspanAttr);
      maker.addText(tableDiv, "No SATA disks found");
    }
    else
    {
      if ( stats.sataBeastStatus > 0 )
        warningAttr[ "bgcolor" ] = alarmColors_.find(AlarmHandler::ERROR)->second;
      tableRow = maker.addNode("tr", table, warningAttr);
      tableDiv = maker.addNode("td", tableRow);
      maker.addText(tableDiv, "SATA beast status");
      tableDiv = maker.addNode("td", tableRow, tableValueAttr);
      maker.addInt( tableDiv, stats.sataBeastStatus );
    }
  }
  
  
  void SMWebPageHelper::addTableForDiskUsages
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *parent,
    ResourceMonitorCollection::Stats const& stats
  ) const
  {
    XHTMLMaker::AttrMap colspanAttr;
    colspanAttr[ "colspan" ] = "2";
    
    XHTMLMaker::AttrMap tableLabelAttr = tableLabelAttr_;
    tableLabelAttr[ "width" ] = "54%";
    
    XHTMLMaker::AttrMap tableValueAttr = tableValueAttr_;
    tableValueAttr[ "width" ] = "46%";
    
    XHTMLMaker::AttrMap warningAttr = rowAttr_;
    
    XHTMLMaker::Node* table = maker.addNode("table", parent, tableAttr_);
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
    maker.addText(tableDiv, "Disk Space Usage");
    
    
    for (ResourceMonitorCollection::DiskUsageStatsPtrList::const_iterator
           it = stats.diskUsageStatsList.begin(),
           itEnd = stats.diskUsageStatsList.end();
         it != itEnd;
         ++it)
    {
      warningAttr[ "bgcolor" ] = alarmColors_.find( (*it)->alarmState )->second;
      tableRow = maker.addNode("tr", table, warningAttr);
      tableDiv = maker.addNode("td", tableRow);
      maker.addText(tableDiv, (*it)->pathName);
      tableDiv = maker.addNode("td", tableRow, tableValueAttr);
      if ( (*it)->diskSize > 0 )
      {
        std::ostringstream tmpString;
        tmpString << std::fixed << std::setprecision(0) <<
          (*it)->relDiskUsage << "% (" <<
          (*it)->absDiskUsage << " of " << 
          (*it)->diskSize << " GB)";
        maker.addText(tableDiv, tmpString.str());
      }
      else
      {
        maker.addText(tableDiv, "not mounted");
      }
    }
  }
  
  
  void SMWebPageHelper::addDOMforSummaryInformation
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *parent,
    DataSenderMonitorCollection const& dsmc,
    StreamsMonitorCollection const& smc,
    EventConsumerMonitorCollection const& ecmc,
    DQMEventMonitorCollection const& dmc,
    RegistrationCollectionPtr registrationCollection
  ) const
  {
    DataSenderMonitorCollection::OutputModuleResultsList resultsList =
      dsmc.getTopLevelOutputModuleResults();

    StreamsMonitorCollection::StreamRecordList streamRecords;
    smc.getStreamRecords(streamRecords);

    XHTMLMaker::AttrMap colspanAttr;
    colspanAttr[ "colspan" ] = "6";

    XHTMLMaker::AttrMap bandwidthColspanAttr;
    bandwidthColspanAttr[ "colspan" ] = "3";

    XHTMLMaker::AttrMap tableValueWidthAttr;
    tableValueWidthAttr[ "width" ] = "15%";
    
    XHTMLMaker::AttrMap rowspanAttr = tableValueWidthAttr;
    rowspanAttr[ "rowspan" ] = "2";
    rowspanAttr[ "valign" ] = "top";
    
    XHTMLMaker::Node* table = maker.addNode("table", parent, tableAttr_);
    
    // Summary header
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
    maker.addText(tableDiv, "Data Flow Summary");
    
    // Parameter/Value header
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("th", tableRow, rowspanAttr);
    maker.addText(tableDiv, "Output Module");
    tableDiv = maker.addNode("th", tableRow, rowspanAttr);
    maker.addText(tableDiv, "Event size (kB)");
    tableDiv = maker.addNode("th", tableRow, rowspanAttr);
    maker.addText(tableDiv, "Rate (Hz)");
    tableDiv = maker.addNode("th", tableRow, bandwidthColspanAttr);
    maker.addText(tableDiv, "Bandwidth (MB/s)");

    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("th", tableRow, tableValueWidthAttr);
    maker.addText(tableDiv, "Input");
    tableDiv = maker.addNode("th", tableRow, tableValueWidthAttr);
    maker.addText(tableDiv, "To disk");
    tableDiv = maker.addNode("th", tableRow, tableValueWidthAttr);
    maker.addText(tableDiv, "To consumers");

    
    if (resultsList.empty())
    {
      XHTMLMaker::AttrMap messageAttr = colspanAttr;
      messageAttr[ "align" ] = "center";
      
      tableRow = maker.addNode("tr", table, rowAttr_);
      tableDiv = maker.addNode("td", tableRow, messageAttr);
      maker.addText(tableDiv, "No output modules are available yet.");
      return;
    }
    else
    {
      double totalInputBandwidth = 0;
      double totalDiskBandwidth = 0;
      double totalConsumerBandwidth = 0;

      for (
        DataSenderMonitorCollection::OutputModuleResultsList::const_iterator
          it = resultsList.begin(), itEnd = resultsList.end();
        it != itEnd; ++it
      )
      {
        const std::string outputModuleLabel = (*it)->name;

        const double inputBandwidth =
          (*it)->eventStats.getValueRate(MonitoredQuantity::RECENT)/(double)0x100000;
        totalInputBandwidth += inputBandwidth;

        StreamsMonitorCollection::StreamRecordList streamRecords;
        double diskBandwidth = 0;
        if ( smc.getStreamRecordsForOutputModuleLabel(outputModuleLabel, streamRecords) )
        {
          for (
            StreamsMonitorCollection::StreamRecordList::const_iterator
              it = streamRecords.begin(), itEnd = streamRecords.end();
            it != itEnd; ++it
          )
          {
            MonitoredQuantity::Stats streamBandwidthStats;
            (*it)->bandwidth.getStats(streamBandwidthStats);
            diskBandwidth += streamBandwidthStats.getValueRate(MonitoredQuantity::RECENT);
          }
          totalDiskBandwidth += diskBandwidth;
        }
        else
        {
          diskBandwidth = -1;
        }

        const double consumerBandwidth =
          getServedConsumerBandwidth(outputModuleLabel,
            registrationCollection, ecmc);
        totalConsumerBandwidth += consumerBandwidth;

        tableRow = maker.addNode("tr", table, rowAttr_);
        
        // Output module label
        tableDiv = maker.addNode("td", tableRow);
        maker.addText(tableDiv, outputModuleLabel);
        
        // event size
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addDouble( tableDiv,
          (*it)->eventStats.getValueAverage(MonitoredQuantity::RECENT)/(double)0x400, 1 );
        
        // rate
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addDouble( tableDiv,
          (*it)->eventStats.getSampleRate(MonitoredQuantity::RECENT), 1 );
        
        // input b/w
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addDouble( tableDiv, inputBandwidth, 1 );
        
        // b/w to disk
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        if ( diskBandwidth < 0 )
          maker.addText( tableDiv, "not written" );
        else
          maker.addDouble( tableDiv, diskBandwidth, 1 );

        // b/w to consumers
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addDouble( tableDiv, consumerBandwidth, 1 );
      }
      
      // DQM
      DQMEventMonitorCollection::DQMEventStats dqmStats;
      dmc.getStats(dqmStats);
      
      tableRow = maker.addNode("tr", table, rowAttr_);
      tableDiv = maker.addNode("td", tableRow);
      maker.addText(tableDiv, "DQM histograms");
      
      // DQM event size
      tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
      maker.addDouble( tableDiv,
        dqmStats.dqmEventSizeStats.getValueAverage(MonitoredQuantity::RECENT)/(double)0x400, 1 );
      
      // DQM rate
      tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
      maker.addDouble( tableDiv,
        dqmStats.numberOfTopLevelFoldersStats.getSampleRate(MonitoredQuantity::RECENT), 1 );
      
      // DQM input b/w
      tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
      const double dqmInputBandwidth = dqmStats.dqmEventSizeStats.getValueRate(MonitoredQuantity::RECENT);
      totalInputBandwidth += dqmInputBandwidth;
      maker.addDouble( tableDiv, dqmInputBandwidth, 1 );
      
      // DQM b/w to disk
      tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
      maker.addText( tableDiv, "not written" );
      // const double dqmDiskBandwidth = dqmStats.writtenDQMEventSizeStats.getValueRate(MonitoredQuantity::RECENT);
      // totalDiskBandwidth += dqmDiskBandwidth;
      // maker.addDouble( tableDiv, dqmDiskBandwidth, 1 );
      
      // DQM b/w to consumers
      tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
      const double dqmConsumerBandwidth = dqmStats.servedDQMEventSizeStats.getValueRate(MonitoredQuantity::RECENT);
      totalConsumerBandwidth += dqmConsumerBandwidth;
      maker.addDouble( tableDiv, dqmConsumerBandwidth, 1 );
      
      
      // Totals
      tableRow = maker.addNode("tr", table, specialRowAttr_);
      tableDiv = maker.addNode("td", tableRow);
      maker.addText(tableDiv, "Total");
      tableDiv = maker.addNode("td", tableRow);
      tableDiv = maker.addNode("td", tableRow);
      tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
      maker.addDouble( tableDiv, totalInputBandwidth, 1 );
      tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
      maker.addDouble( tableDiv, totalDiskBandwidth, 1 );
      tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
      maker.addDouble( tableDiv, totalConsumerBandwidth, 1 );
    }
  }
  
  
  double SMWebPageHelper::getServedConsumerBandwidth
  (
    const std::string& label,
    RegistrationCollectionPtr registrationCollection,
    const EventConsumerMonitorCollection& eventConsumerCollection
  ) const
  {
    double bandwidth = 0;

    RegistrationCollection::ConsumerRegistrations consumers;
    registrationCollection->getEventConsumers(consumers);

    for( RegistrationCollection::ConsumerRegistrations::const_iterator
           it = consumers.begin(), itEnd = consumers.end();
         it != itEnd; ++it )
    {
      if ( (*it)->outputModuleLabel() == label )
      {
        // Events served:
        MonitoredQuantity::Stats servedStats;
        if ( eventConsumerCollection.getServed( (*it)->queueId(), servedStats ) )
        {
          bandwidth += servedStats.getValueRate(MonitoredQuantity::RECENT);
        }
      }
    }
    
    return ( bandwidth/(double)0x100000 );
  }
  
  
  void SMWebPageHelper::addDOMforFragmentMonitor
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *parent,
    FragmentMonitorCollection const& fmc
  ) const
  {
    FragmentMonitorCollection::FragmentStats stats;
    fmc.getStats(stats);
    
    XHTMLMaker::AttrMap colspanAttr;
    colspanAttr[ "colspan" ] = "4";
    
    XHTMLMaker::Node* table = maker.addNode("table", parent, tableAttr_);
    
    // Received Data Statistics header
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
    maker.addText(tableDiv, "Received I2O Frames");
    
    // Parameter/Value header
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Parameter");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Total");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Events");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "DQM histos");
    
    addFragmentStats(maker, table, stats,  MonitoredQuantity::FULL);
    
    addFragmentStats(maker, table, stats,  MonitoredQuantity::RECENT);
  }
  
  void SMWebPageHelper::addFragmentStats
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *table,
    FragmentMonitorCollection::FragmentStats const& stats,
    const MonitoredQuantity::DataSetType dataSet
  ) const
  {
    // Mean performance header
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow);
    if ( dataSet == MonitoredQuantity::FULL )
      maker.addText(tableDiv, "Performance for full run");
    else
      maker.addText(tableDiv, "Recent performance for last");
    
    addDurationToTableHead(maker, tableRow,
      stats.allFragmentSizeStats.getDuration(dataSet));
    addDurationToTableHead(maker, tableRow,
      stats.eventFragmentSizeStats.getDuration(dataSet));
    addDurationToTableHead(maker, tableRow,
      stats.dqmEventFragmentSizeStats.getDuration(dataSet));
    
    addRowForFramesReceived(maker, table, stats, dataSet);
    addRowForBandwidth(maker, table, stats, dataSet);
    addRowForRate(maker, table, stats, dataSet);
    addRowForLatency(maker, table, stats, dataSet);
    if ( dataSet == MonitoredQuantity::FULL )
    {
      addRowForTotalVolume(maker, table, stats, dataSet);
    }
    else
    {
      addRowForMaxBandwidth(maker, table, stats, dataSet);
      addRowForMinBandwidth(maker, table, stats, dataSet);
    }
  }
  
  
  void SMWebPageHelper::addRowForFramesReceived
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *table,
    FragmentMonitorCollection::FragmentStats const& stats,
    const MonitoredQuantity::DataSetType dataSet
  ) const
  {
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Frames Received");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, stats.allFragmentSizeStats.getSampleCount(dataSet) );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, stats.eventFragmentSizeStats.getSampleCount(dataSet) );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, stats.dqmEventFragmentSizeStats.getSampleCount(dataSet) );
  }
  
  
  void SMWebPageHelper::addRowForBandwidth
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *table,
    FragmentMonitorCollection::FragmentStats const& stats,
    const MonitoredQuantity::DataSetType dataSet
  ) const
  {
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Bandwidth (MB/s)");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, stats.allFragmentSizeStats.getValueRate(dataSet) );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, stats.eventFragmentSizeStats.getValueRate(dataSet) );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, stats.dqmEventFragmentSizeStats.getValueRate(dataSet) );
  }
  
  
  void SMWebPageHelper::addRowForRate
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *table,
    FragmentMonitorCollection::FragmentStats const& stats,
    const MonitoredQuantity::DataSetType dataSet
  ) const
  {
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Rate (frames/s)");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, stats.allFragmentSizeStats.getSampleRate(dataSet) );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, stats.eventFragmentSizeStats.getSampleRate(dataSet) );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, stats.dqmEventFragmentSizeStats.getSampleRate(dataSet) );
  }
  
  
  void SMWebPageHelper::addRowForLatency
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *table,
    FragmentMonitorCollection::FragmentStats const& stats,
    const MonitoredQuantity::DataSetType dataSet
  ) const
  {
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Latency (us/frame)");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, stats.allFragmentSizeStats.getSampleLatency(dataSet) );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, stats.eventFragmentSizeStats.getSampleLatency(dataSet) );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, stats.dqmEventFragmentSizeStats.getSampleLatency(dataSet) );
  }
  
  
  void SMWebPageHelper::addRowForTotalVolume
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *table,
    FragmentMonitorCollection::FragmentStats const& stats,
    const MonitoredQuantity::DataSetType dataSet
  ) const
  {
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Total volume received (MB)");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, stats.allFragmentSizeStats.getValueSum(dataSet), 3 );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, stats.eventFragmentSizeStats.getValueSum(dataSet), 3 );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, stats.dqmEventFragmentSizeStats.getValueSum(dataSet), 3 );
  }
  
  
  void SMWebPageHelper::addRowForMaxBandwidth
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *table,
    FragmentMonitorCollection::FragmentStats const& stats,
    const MonitoredQuantity::DataSetType dataSet
  ) const
  {
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Maximum Bandwidth (MB/s)");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, stats.allFragmentBandwidthStats.getValueMax(dataSet) );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, stats.eventFragmentBandwidthStats.getValueMax(dataSet) );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, stats.dqmEventFragmentBandwidthStats.getValueMax(dataSet) );
  }
  
  
  void SMWebPageHelper::addRowForMinBandwidth
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *table,
    FragmentMonitorCollection::FragmentStats const& stats,
    const MonitoredQuantity::DataSetType dataSet
  ) const
  {
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Minimum Bandwidth (MB/s)");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, stats.allFragmentBandwidthStats.getValueMin(dataSet) );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, stats.eventFragmentBandwidthStats.getValueMin(dataSet) );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, stats.dqmEventFragmentBandwidthStats.getValueMin(dataSet) );
  }
  
  
  void SMWebPageHelper::addDOMforRunMonitor
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *parent,
    RunMonitorCollection const& rmc
  ) const
  {
    MonitoredQuantity::Stats eventIDsReceivedStats;
    rmc.getEventIDsReceivedMQ().getStats(eventIDsReceivedStats);
    MonitoredQuantity::Stats errorEventIDsReceivedStats;
    rmc.getErrorEventIDsReceivedMQ().getStats(errorEventIDsReceivedStats);
    MonitoredQuantity::Stats unwantedEventIDsReceivedStats;
    rmc.getUnwantedEventIDsReceivedMQ().getStats(unwantedEventIDsReceivedStats);
    MonitoredQuantity::Stats runNumbersSeenStats;
    rmc.getRunNumbersSeenMQ().getStats(runNumbersSeenStats);
    MonitoredQuantity::Stats lumiSectionsSeenStats;
    rmc.getLumiSectionsSeenMQ().getStats(lumiSectionsSeenStats);
    MonitoredQuantity::Stats eolsSeenStats;
    rmc.getEoLSSeenMQ().getStats(eolsSeenStats);
    
    XHTMLMaker::AttrMap colspanAttr;
    colspanAttr[ "colspan" ] = "6";
    
    XHTMLMaker::AttrMap tableValueAttr = tableValueAttr_;
    tableValueAttr[ "width" ] = "16%";
    
    XHTMLMaker::Node* table = maker.addNode("table", parent, tableAttr_);
    
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
    maker.addText(tableDiv, "Storage Manager Statistics");
    
    // Run number and lumi section
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Run number");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, runNumbersSeenStats.getLastSampleValue(), 0 );
    tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Current lumi section");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, lumiSectionsSeenStats.getLastSampleValue(), 0 );
    tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Last EoLS");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, eolsSeenStats.getLastSampleValue(), 0 );
    
    // Total events received
    tableRow = maker.addNode("tr", table, specialRowAttr_);
    tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Events received (non-unique)");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addInt( tableDiv, eventIDsReceivedStats.getSampleCount() );
    tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Error events received");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addInt( tableDiv, errorEventIDsReceivedStats.getSampleCount() );
    tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Unwanted events received");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addInt( tableDiv, unwantedEventIDsReceivedStats.getSampleCount() );
    
    // Last event IDs
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Last event ID");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, eventIDsReceivedStats.getLastSampleValue(), 0 );
    tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Last error event ID");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, errorEventIDsReceivedStats.getLastSampleValue(), 0 );
    tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Last unwanted event ID");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, unwantedEventIDsReceivedStats.getLastSampleValue(), 0 );
  }
  
  
  void SMWebPageHelper::addDOMforStoredData
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *parent,
    StreamsMonitorCollection const& smc
  ) const
  {
    MonitoredQuantity::Stats allStreamsVolumeStats;
    smc.getAllStreamsVolumeMQ().getStats(allStreamsVolumeStats);
    
    XHTMLMaker::AttrMap tableValueWidthAttr;
    tableValueWidthAttr[ "width" ] = "11%";
    
    XHTMLMaker::AttrMap rowspanAttr = tableValueWidthAttr;
    rowspanAttr[ "rowspan" ] = "2";
    rowspanAttr[ "valign" ] = "top";
    
    XHTMLMaker::AttrMap colspanAttr;
    colspanAttr[ "colspan" ] = "9";
    
    XHTMLMaker::AttrMap bandwidthColspanAttr;
    bandwidthColspanAttr[ "colspan" ] = "4";
    
    XHTMLMaker::Node* table = maker.addNode("table", parent, tableAttr_);
    
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
    maker.addText(tableDiv, "Stored Data Statistics");
    
    // Header
    tableRow = maker.addNode("tr", table, specialRowAttr_);
    tableDiv = maker.addNode("th", tableRow, rowspanAttr);
    maker.addText(tableDiv, "Stream");
    tableDiv = maker.addNode("th", tableRow, rowspanAttr);
    maker.addText(tableDiv, "Fraction to disk");
    tableDiv = maker.addNode("th", tableRow, rowspanAttr);
    maker.addText(tableDiv, "Files");
    tableDiv = maker.addNode("th", tableRow, rowspanAttr);
    maker.addText(tableDiv, "Events");
    tableDiv = maker.addNode("th", tableRow, rowspanAttr);
    maker.addText(tableDiv, "Events/s");
    tableDiv = maker.addNode("th", tableRow, rowspanAttr);
    maker.addText(tableDiv, "Volume (MB)");
    tableDiv = maker.addNode("th", tableRow, bandwidthColspanAttr);
    maker.addText(tableDiv, "Bandwidth (MB/s)");
    
    tableRow = maker.addNode("tr", table, specialRowAttr_);
    tableDiv = maker.addNode("th", tableRow, tableValueWidthAttr);
    maker.addText(tableDiv, "average");
    tableDiv = maker.addNode("th", tableRow, tableValueWidthAttr);
    maker.addText(tableDiv, "min");
    tableDiv = maker.addNode("th", tableRow, tableValueWidthAttr);
    maker.addText(tableDiv, "max");
    
    if (! smc.streamRecordsExist())
    {
      tableRow = maker.addNode("tr", table, rowAttr_);
      tableDiv = maker.addNode("td", tableRow, colspanAttr);
      maker.addText(tableDiv, "no streams available yet");
      return;
    }
    // Mean performance
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("th", tableRow, colspanAttr);
    {
      std::ostringstream tmpString;
      tmpString << "Mean performance for " <<
        allStreamsVolumeStats.getDuration().total_seconds() << " s";
      maker.addText(tableDiv, tmpString.str());
    }
    listStreamRecordsStats(maker, table, smc, MonitoredQuantity::FULL);
    
    
    // Recent performance
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("th", tableRow, colspanAttr);
    {
      std::ostringstream tmpString;
      tmpString << "Recent performance for the last " <<
        allStreamsVolumeStats.getDuration(MonitoredQuantity::RECENT).total_seconds() << " s";
      maker.addText(tableDiv, tmpString.str());
    }
    listStreamRecordsStats(maker, table, smc, MonitoredQuantity::RECENT);
  }
  
  
  void SMWebPageHelper::addDOMforConfigString
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *parent,
    DiskWritingParams const& dwParams
  ) const
  {
    XHTMLMaker::Node* table = maker.addNode("table", parent);
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, specialRowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "SM Configuration");
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow);
    XHTMLMaker::AttrMap textareaAttr;
    textareaAttr[ "rows" ] = "10";
    textareaAttr[ "cols" ] = "100";
    textareaAttr[ "scroll" ] = "yes";
    textareaAttr[ "readonly" ];
    textareaAttr[ "title" ] = "SM config";
    XHTMLMaker::Node* textarea = maker.addNode("textarea", tableDiv, textareaAttr);
    maker.addText(textarea, dwParams.streamConfiguration_);
  }  
  
  
  void SMWebPageHelper::listStreamRecordsStats
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *table,
    StreamsMonitorCollection const& smc,
    const MonitoredQuantity::DataSetType dataSet
  ) const
  {
    StreamsMonitorCollection::StreamRecordList streamRecords;
    smc.getStreamRecords(streamRecords);
    MonitoredQuantity::Stats allStreamsFileCountStats;
    smc.getAllStreamsFileCountMQ().getStats(allStreamsFileCountStats);
    MonitoredQuantity::Stats allStreamsVolumeStats;
    smc.getAllStreamsVolumeMQ().getStats(allStreamsVolumeStats);
    MonitoredQuantity::Stats allStreamsBandwidthStats;
    smc.getAllStreamsBandwidthMQ().getStats(allStreamsBandwidthStats);
    
    XHTMLMaker::Node *tableRow, *tableDiv;
    
    XHTMLMaker::AttrMap tableValueAttr = tableValueAttr_;
    tableValueAttr[ "width" ] = "11%";
    
    
    for (
      StreamsMonitorCollection::StreamRecordList::const_iterator 
        it = streamRecords.begin(), itEnd = streamRecords.end();
      it != itEnd;
      ++it
    )
    {
      MonitoredQuantity::Stats streamFileCountStats;
      (*it)->fileCount.getStats(streamFileCountStats);
      MonitoredQuantity::Stats streamVolumeStats;
      (*it)->volume.getStats(streamVolumeStats);
      MonitoredQuantity::Stats streamBandwidthStats;
      (*it)->bandwidth.getStats(streamBandwidthStats);
      
      
      tableRow = maker.addNode("tr", table, rowAttr_);
      tableDiv = maker.addNode("td", tableRow);
      maker.addText(tableDiv, (*it)->streamName);
      tableDiv = maker.addNode("td", tableRow);
      maker.addDouble(tableDiv, (*it)->fractionToDisk, 2);
      tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
      maker.addInt( tableDiv, streamFileCountStats.getSampleCount(dataSet) );
      tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
      maker.addInt( tableDiv, streamVolumeStats.getSampleCount(dataSet) );
      tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
      maker.addDouble( tableDiv, streamVolumeStats.getSampleRate(dataSet), 1 );
      tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
      maker.addDouble( tableDiv, streamVolumeStats.getValueSum(dataSet), 1 );
      tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
      maker.addDouble( tableDiv, streamBandwidthStats.getValueRate(dataSet) );
      tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
      maker.addDouble( tableDiv, streamBandwidthStats.getValueMin(dataSet) );
      tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
      maker.addDouble( tableDiv, streamBandwidthStats.getValueMax(dataSet) );
    }
    
    tableRow = maker.addNode("tr", table, specialRowAttr_);
    tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Total");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, allStreamsFileCountStats.getSampleCount(dataSet) );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, allStreamsVolumeStats.getSampleCount(dataSet) );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, allStreamsVolumeStats.getSampleRate(dataSet), 1 );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, allStreamsVolumeStats.getValueSum(dataSet), 1 );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, allStreamsBandwidthStats.getValueRate(dataSet) );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, allStreamsBandwidthStats.getValueMin(dataSet) );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, allStreamsBandwidthStats.getValueMax(dataSet) );
  }
  
  
  void SMWebPageHelper::addDOMforFiles
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *parent,
    FilesMonitorCollection const& fmc
  ) const
  {
    FilesMonitorCollection::FileRecordList fileRecords;
    fmc.getFileRecords(fileRecords);
    
    XHTMLMaker::AttrMap colspanAttr;
    colspanAttr[ "colspan" ] = "6";
    
    XHTMLMaker::AttrMap tableLabelAttr = tableLabelAttr_;
    tableLabelAttr[ "align" ] = "center";
    
    XHTMLMaker::AttrMap tableValueWidthAttr;
    tableValueWidthAttr[ "width" ] = "11%";
    
    XHTMLMaker::AttrMap tableCounterWidthAttr;
    tableCounterWidthAttr[ "width" ] = "5%";
    
    XHTMLMaker::Node* table = maker.addNode("table", parent, tableAttr_);
    
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
    maker.addText(tableDiv, "File Statistics (most recent first)");
    
    // Header
    tableRow = maker.addNode("tr", table, specialRowAttr_);
    tableDiv = maker.addNode("th", tableRow, tableCounterWidthAttr);
    maker.addText(tableDiv, "#");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Pathname");
    tableDiv = maker.addNode("th", tableRow, tableValueWidthAttr);
    maker.addText(tableDiv, "Events");
    tableDiv = maker.addNode("th", tableRow, tableValueWidthAttr);
    maker.addText(tableDiv, "Size (Bytes)");
    tableDiv = maker.addNode("th", tableRow, tableValueWidthAttr);
    maker.addText(tableDiv, "Closing reason");
    tableDiv = maker.addNode("th", tableRow, tableValueWidthAttr);
    maker.addText(tableDiv, "Adler32");
    
    // File list
    if (fileRecords.empty())
    {
      tableRow = maker.addNode("tr", table, rowAttr_);
      tableDiv = maker.addNode("td", tableRow, colspanAttr);
      maker.addText(tableDiv, "no files available yet");
      return;
    }
    
    for (
      FilesMonitorCollection::FileRecordList::const_reverse_iterator 
        it = fileRecords.rbegin(), itEnd = fileRecords.rend();
      it != itEnd;
      ++it
    )
    {
      tableRow = maker.addNode("tr", table, rowAttr_);
      tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
      maker.addInt( tableDiv, (*it)->entryCounter );
      tableDiv = maker.addNode("td", tableRow);
      maker.addText(tableDiv, (*it)->completeFileName());
      tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
      maker.addInt( tableDiv, (*it)->eventCount );
      tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
      maker.addInt( tableDiv, (*it)->fileSize );
      tableDiv = maker.addNode("td", tableRow, tableLabelAttr);
      maker.addText(tableDiv, (*it)->closingReason());
      tableDiv = maker.addNode("td", tableRow, tableLabelAttr);
      maker.addHex(tableDiv, (*it)->adler32);
    }
  }
  
  
  void SMWebPageHelper::addDOMforThroughputStatistics
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *parent,
    ThroughputMonitorCollection const& tmc
  ) const
  {
    XHTMLMaker::AttrMap colspanAttr;
    colspanAttr[ "colspan" ] = "21";
    
    XHTMLMaker::Node* table = maker.addNode("table", parent, tableAttr_);
    
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
    maker.addText(tableDiv, "Throughput Statistics");
    
    // Header
    tableRow = maker.addNode("tr", table, specialRowAttr_);
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Time (UTC)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Memory pool usage (bytes)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Instantaneous Number of Fragments in Fragment Queue");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Memory used in Fragment Queue (MB)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Number of Fragments Popped from Fragment Queue (Hz)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Data Rate Popped from Fragment Queue (MB/sec)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Fragment Processor Thread Busy Percentage");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Instantaneous Number of Events in Fragment Store");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Memory used in Fragment Store (MB)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Instantaneous Number of Events in Stream Queue");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Memory used in Stream Queue (MB)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Number of Events Popped from Stream Queue (Hz)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Data Rate Popped from Stream Queue (MB/sec)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Disk Writer Thread Busy Percentage");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Number of Events Written to Disk (Hz)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Data  Rate to Disk (MB/sec)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Instantaneous Number of DQMEvents in DQMEvent Queue");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Memory used in DQMEvent Queue (MB)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Number of DQMEvents Popped from DQMEvent Queue (Hz)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Data Rate Popped from DQMEvent Queue (MB/sec)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "DQMEvent Processor Thread Busy Percentage");
    
    ThroughputMonitorCollection::Stats stats;
    tmc.getStats(stats);
    
    addRowForThroughputStatistics(maker, table, stats.average, true);
    
    for (ThroughputMonitorCollection::Stats::Snapshots::const_iterator
           it = stats.snapshots.begin(),
           itEnd = stats.snapshots.end();
         it != itEnd;
         ++it)
    {
      addRowForThroughputStatistics(maker, table, (*it));
    }
    
    addRowForThroughputStatistics(maker, table, stats.average, true);
  }
  
  
  void SMWebPageHelper::addRowForThroughputStatistics
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node* table,
    const ThroughputMonitorCollection::Stats::Snapshot& snapshot,
    const bool isAverage
  ) const
  {
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv;
    XHTMLMaker::AttrMap tableValueAttr = tableValueAttr_;
    
    if (isAverage)
    {
      tableValueAttr[ "style" ] = "background-color: yellow;";
      tableDiv = maker.addNode("td", tableRow, tableValueAttr);
      std::ostringstream avg;
      avg << "<" << snapshot.duration.total_seconds() << "s>";
      maker.addText(tableDiv, avg.str());
    }
    else
    {
      tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
      maker.addText( tableDiv, utils::timeStampUTC(snapshot.absoluteTime) );
    }
    
    // memory pool usage
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.poolUsage, 0 );
    
    // number of fragments in fragment queue
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.entriesInFragmentQueue, 0 );
    
    // memory used in fragment queue
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.memoryUsedInFragmentQueue, 1 );
    
    // number of fragments popped from fragment queue
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.fragmentQueueRate, 0 );
    
    // data rate popped from fragment queue
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.fragmentQueueBandwidth, 1 );
    
    // fragment processor thread busy percentage
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.fragmentProcessorBusy, 1 );
    
    // number of events in fragment store
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.fragmentStoreSize, 0 );
    
    // memory used in fragment store
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.fragmentStoreMemoryUsed, 1 );
    
    // number of events in stream queue
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.entriesInStreamQueue, 0 );
    
    // memory used in stream queue
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.memoryUsedInStreamQueue, 1 );
    
    // number of events popped from stream queue
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.streamQueueRate, 0 );
    
    // data rate popped from stream queue
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.streamQueueBandwidth, 1 );
  
    // disk writer thread busy percentage
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.diskWriterBusy, 1 );
  
    // number of events written to disk
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.writtenEventsRate, 0 );
    
    // date rate written to disk
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.writtenEventsBandwidth, 1 );
    
    // number of dqm events in DQMEvent queue
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.entriesInDQMQueue, 0 );
    
    // memory used in DQMEvent queue
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.memoryUsedInDQMQueue, 1 );
    
    // number of dqm events popped from DQMEvent queue
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.dqmQueueRate, 0 );
    
    // data rate popped from DQMEvent queue
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.dqmQueueBandwidth, 1 );
    
    // DQMEvent processor thread busy percentage
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, snapshot.dqmEventProcessorBusy, 1 );
  }
  
  
  void SMWebPageHelper::addOutputModuleTables
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *parent,
    DataSenderMonitorCollection const& dsmc
  ) const
  {
    DataSenderMonitorCollection::OutputModuleResultsList resultsList =
      dsmc.getTopLevelOutputModuleResults();
    
    addOutputModuleSummary(maker, parent, resultsList);
    addOutputModuleStatistics(maker, parent, resultsList);
  }
  
  
  void SMWebPageHelper::addOutputModuleStatistics
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *parent,
    long long uniqueRBID,
    DataSenderMonitorCollection const& dsmc
  ) const
  {
    DataSenderMonitorCollection::OutputModuleResultsList resultsList =
      dsmc.getOutputModuleResultsForRB(uniqueRBID);
    
    addOutputModuleStatistics(maker, parent, resultsList);
  }
  
  
  void SMWebPageHelper::addOutputModuleStatistics
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *parent,
    DataSenderMonitorCollection::OutputModuleResultsList const& resultsList
  ) const
  {
    XHTMLMaker::AttrMap colspanAttr;
    colspanAttr[ "colspan" ] = "7";
    
    XHTMLMaker::Node* table = maker.addNode("table", parent, tableAttr_);
    
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
    maker.addText(tableDiv, "Received Data Statistics (by output module)");
    
    // Header
    tableRow = maker.addNode("tr", table, specialRowAttr_);
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Output Module");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Events");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Size (MB)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Size/Evt (KB)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "RMS (KB)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Min (KB)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Max (KB)");
    
    if (resultsList.empty())
    {
      XHTMLMaker::AttrMap messageAttr = colspanAttr;
      messageAttr[ "align" ] = "center";
      
      tableRow = maker.addNode("tr", table, rowAttr_);
      tableDiv = maker.addNode("td", tableRow, messageAttr);
      maker.addText(tableDiv, "No output modules are available yet.");
      return;
    }
    else
    {
      for (unsigned int idx = 0; idx < resultsList.size(); ++idx)
      {
        std::string outputModuleLabel = resultsList[idx]->name;
        
        tableRow = maker.addNode("tr", table, rowAttr_);
        tableDiv = maker.addNode("td", tableRow);
        maker.addText(tableDiv, outputModuleLabel);
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addInt( tableDiv, resultsList[idx]->eventStats.getSampleCount() );
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addDouble( tableDiv,
          resultsList[idx]->eventStats.getValueSum()/(double)0x100000 );
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addDouble( tableDiv,
          resultsList[idx]->eventStats.getValueAverage()/(double)0x400 );
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addDouble( tableDiv,
          resultsList[idx]->eventStats.getValueRMS()/(double)0x400 );
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addDouble( tableDiv,
          resultsList[idx]->eventStats.getValueMin()/(double)0x400 );
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addDouble( tableDiv,
          resultsList[idx]->eventStats.getValueMax()/(double)0x400 );
      }
    }
  }
  
  
  void SMWebPageHelper::addOutputModuleSummary
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *parent,
    DataSenderMonitorCollection::OutputModuleResultsList const& resultsList
  ) const
  {
    XHTMLMaker::AttrMap colspanAttr;
    colspanAttr[ "colspan" ] = "3";
    
    XHTMLMaker::Node* table = maker.addNode("table", parent, tableAttr_);
    
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
    maker.addText(tableDiv, "Output Module Summary");
    
    // Header
    tableRow = maker.addNode("tr", table, specialRowAttr_);
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Name");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "ID");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Header Size (bytes)");
    
    if (resultsList.empty())
    {
      XHTMLMaker::AttrMap messageAttr = colspanAttr;
      messageAttr[ "align" ] = "center";
      
      tableRow = maker.addNode("tr", table, rowAttr_);
      tableDiv = maker.addNode("td", tableRow, messageAttr);
      maker.addText(tableDiv, "No output modules are available yet.");
      return;
    }
    else
    {
      for (unsigned int idx = 0; idx < resultsList.size(); ++idx)
      {
        tableRow = maker.addNode("tr", table, rowAttr_);
        tableDiv = maker.addNode("td", tableRow);
        maker.addText(tableDiv, resultsList[idx]->name);
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addInt( tableDiv, resultsList[idx]->id );
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addInt( tableDiv, resultsList[idx]->initMsgSize );
      }
    }
  }
  
  
  void SMWebPageHelper::addResourceBrokerList
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *parent,
    DataSenderMonitorCollection const& dsmc
  ) const
  {
    DataSenderMonitorCollection::ResourceBrokerResultsList rbResultsList =
      dsmc.getAllResourceBrokerResults();
    std::sort(rbResultsList.begin(), rbResultsList.end(), compareRBResultPtrValues);
    
    XHTMLMaker::AttrMap colspanAttr;
    colspanAttr[ "colspan" ] = "15";
    
    XHTMLMaker::AttrMap tableSuspiciousValueAttr = tableValueAttr_;
    tableSuspiciousValueAttr[ "style" ] = "background-color: yellow;";
    
    XHTMLMaker::Node* table = maker.addNode("table", parent, tableAttr_);
    
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
    maker.addText(tableDiv, "Data Sender Overview");
    
    // Header
    tableRow = maker.addNode("tr", table, specialRowAttr_);
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Resource Broker URL");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "RB instance");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "RB TID");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of EPs");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of INIT messages");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of events");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of error events");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of faulty events");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of outstanding data discards");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of DQM events");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of faulty DQM events");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of outstanding DQM discards");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of ignored discards");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Recent event rate (Hz)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Last event number received");
    
    if (rbResultsList.empty())
    {
      XHTMLMaker::AttrMap messageAttr = colspanAttr;
      messageAttr[ "align" ] = "center";
      
      tableRow = maker.addNode("tr", table, rowAttr_);
      tableDiv = maker.addNode("td", tableRow, messageAttr);
      maker.addText(tableDiv, "No data senders have registered yet.");
      return;
    }
    else
    {
      for (unsigned int idx = 0; idx < rbResultsList.size(); ++idx)
      {
        tableRow = maker.addNode("tr", table, rowAttr_);
        
        tableDiv = maker.addNode("td", tableRow);
        XHTMLMaker::AttrMap linkAttr;
        linkAttr[ "href" ] = baseURL() + "/rbsenderdetail?id=" +
          boost::lexical_cast<std::string>(rbResultsList[idx]->uniqueRBID);
        XHTMLMaker::Node* link = maker.addNode("a", tableDiv, linkAttr);
        maker.addText(link, rbResultsList[idx]->key.hltURL);
        
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addInt( tableDiv, rbResultsList[idx]->key.hltInstance );
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addInt( tableDiv, rbResultsList[idx]->key.hltTid );
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addInt( tableDiv, rbResultsList[idx]->filterUnitCount );
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addInt(tableDiv, rbResultsList[idx]->initMsgCount );
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addInt( tableDiv, rbResultsList[idx]->eventStats.getSampleCount() );
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addInt( tableDiv, rbResultsList[idx]->errorEventStats.getSampleCount() );
        
        if (rbResultsList[idx]->faultyEventStats.getSampleCount() != 0)
        {
          tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
        }
        else
        {
          tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        }
        maker.addInt( tableDiv, rbResultsList[idx]->faultyEventStats.getSampleCount() );
        
        if (rbResultsList[idx]->outstandingDataDiscardCount != 0)
        {
          tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
        }
        else
        {
          tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        }
        maker.addInt( tableDiv, rbResultsList[idx]->outstandingDataDiscardCount );
        
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addInt( tableDiv, rbResultsList[idx]->dqmEventStats.getSampleCount() );
        
        if (rbResultsList[idx]->faultyDQMEventStats.getSampleCount() != 0)
        {
          tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
        }
        else
        {
          tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        }
        maker.addInt( tableDiv, rbResultsList[idx]->faultyDQMEventStats.getSampleCount() );
        
        if (rbResultsList[idx]->outstandingDQMDiscardCount != 0)
        {
          tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
        }
        else
        {
          tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        }
        maker.addInt( tableDiv, rbResultsList[idx]->outstandingDQMDiscardCount );
        
        const int skippedDiscards = rbResultsList[idx]->skippedDiscardStats.getSampleCount();
        if (skippedDiscards != 0)
        {
          tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
        }
        else
        {
          tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        }
        maker.addInt( tableDiv, skippedDiscards );
        
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addDouble( tableDiv, rbResultsList[idx]->eventStats.
          getSampleRate(MonitoredQuantity::RECENT) );
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addInt( tableDiv, rbResultsList[idx]->lastEventNumber );
      }
    }
  }
  

  void SMWebPageHelper::addResourceBrokerDetails
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *parent,
    long long uniqueRBID,
    DataSenderMonitorCollection const& dsmc
  ) const
  {
    DataSenderMonitorCollection::RBResultPtr rbResultPtr =
      dsmc.getOneResourceBrokerResult(uniqueRBID);
    
    if (rbResultPtr.get() == 0)
    {
      maker.addText(parent, "The requested resource broker page is not currently available.");
      return;
    }
    
    int tmpDuration;
    std::string tmpText;
    
    XHTMLMaker::AttrMap colspanAttr;
    colspanAttr[ "colspan" ] = "2";
    
    XHTMLMaker::AttrMap tableAttr = tableAttr_;
    tableAttr[ "width" ] = "";
    
    XHTMLMaker::Node* table = maker.addNode("table", parent, tableAttr);
    
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
    maker.addText(tableDiv, "Resource Broker Details");
    
    // Header
    tableRow = maker.addNode("tr", table, specialRowAttr_);
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Parameter");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Value");
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    maker.addText(tableDiv, "URL");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    XHTMLMaker::AttrMap linkAttr;
    linkAttr[ "href" ] = rbResultPtr->key.hltURL + "/urn:xdaq-application:lid=" +
      boost::lexical_cast<std::string>(rbResultPtr->key.hltLocalId);
    XHTMLMaker::Node* link = maker.addNode("a", tableDiv, linkAttr);
    maker.addText(link, rbResultPtr->key.hltURL);
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    maker.addText(tableDiv, "Class Name");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addText(tableDiv, rbResultPtr->key.hltClassName);
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    maker.addText(tableDiv, "Instance");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, rbResultPtr->key.hltInstance );
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    maker.addText(tableDiv, "Local ID");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, rbResultPtr->key.hltLocalId );
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    maker.addText(tableDiv, "Tid");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, rbResultPtr->key.hltTid );
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    maker.addText(tableDiv, "INIT Message Count");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, rbResultPtr->initMsgCount );
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    maker.addText(tableDiv, "Event Count");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, rbResultPtr->eventStats.getSampleCount() );
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    maker.addText(tableDiv, "Error Event Count");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, rbResultPtr->errorEventStats.getSampleCount() );
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    maker.addText(tableDiv, "Faulty Event Count");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, rbResultPtr->faultyEventStats.getSampleCount() );
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    maker.addText(tableDiv, "Data Discard Count");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, rbResultPtr->dataDiscardStats.getSampleCount() );
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    maker.addText(tableDiv, "DQM Event Count");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, rbResultPtr->dqmEventStats.getSampleCount() );
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    maker.addText(tableDiv, "Faulty DQM Event Count");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, rbResultPtr->faultyDQMEventStats.getSampleCount() );
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    maker.addText(tableDiv, "DQM Discard Count");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, rbResultPtr->dqmDiscardStats.getSampleCount() );
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    maker.addText(tableDiv, "Ignored Discards Count");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, rbResultPtr->skippedDiscardStats.getSampleCount() );
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    maker.addText(tableDiv, "Last Event Number Received");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, rbResultPtr->lastEventNumber );
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    maker.addText(tableDiv, "Last Run Number Received");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt( tableDiv, rbResultPtr->lastRunNumber );
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    tmpDuration = rbResultPtr->eventStats.recentDuration.total_seconds();
    tmpText =  "Recent (" + boost::lexical_cast<std::string>(tmpDuration) +
      " sec) Event Rate (Hz)";
    maker.addText(tableDiv, tmpText);
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, rbResultPtr->eventStats.recentSampleRate );
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    tmpDuration = rbResultPtr->eventStats.fullDuration.total_seconds();
    tmpText =  "Full (" + boost::lexical_cast<std::string>(tmpDuration) +
      " sec) Event Rate (Hz)";
    maker.addText(tableDiv, tmpText);
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble( tableDiv, rbResultPtr->eventStats.fullSampleRate );
  }
  
  
  void SMWebPageHelper::addFilterUnitList
  (
    XHTMLMaker& maker,
    XHTMLMaker::Node *parent,
    long long uniqueRBID,
    DataSenderMonitorCollection const& dsmc
  ) const
  {
    DataSenderMonitorCollection::FilterUnitResultsList fuResultsList =
      dsmc.getFilterUnitResultsForRB(uniqueRBID);
    
    XHTMLMaker::AttrMap colspanAttr;
    colspanAttr[ "colspan" ] = "13";
    
    XHTMLMaker::AttrMap tableSuspiciousValueAttr = tableValueAttr_;
    tableSuspiciousValueAttr[ "style" ] = "background-color: yellow;";
    
    XHTMLMaker::Node* table = maker.addNode("table", parent, tableAttr_);
    
    XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
    maker.addText(tableDiv, "Filter Units");
    
    // Header
    tableRow = maker.addNode("tr", table, specialRowAttr_);
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Process ID");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of INIT messages");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of events");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of error events");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of faulty events");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of outstanding data discards");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of DQM events");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of faulty DQM events");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of outstanding DQM discards");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "# of ignored discards");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Recent event rate (Hz)");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Last event number received");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "Last run number received");
    
    if (fuResultsList.empty())
    {
      XHTMLMaker::AttrMap messageAttr = colspanAttr;
      messageAttr[ "align" ] = "center";
      
      tableRow = maker.addNode("tr", table, rowAttr_);
      tableDiv = maker.addNode("td", tableRow, messageAttr);
      maker.addText(tableDiv, "No filter units have registered yet.");
      return;
    }
    else
    {
      for (unsigned int idx = 0; idx < fuResultsList.size(); ++idx)
      {
        tableRow = maker.addNode("tr", table, rowAttr_);
        
        tableDiv = maker.addNode("td", tableRow);
        maker.addInt( tableDiv, fuResultsList[idx]->key.fuProcessId );
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addInt( tableDiv, fuResultsList[idx]->initMsgCount );
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addInt( tableDiv, fuResultsList[idx]->shortIntervalEventStats.getSampleCount() );
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addInt( tableDiv, fuResultsList[idx]->errorEventStats.getSampleCount() );
        
        if (fuResultsList[idx]->faultyEventStats.getSampleCount() != 0)
        {
          tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
        }
        else
        {
          tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        }
        maker.addInt( tableDiv, fuResultsList[idx]->faultyEventStats.getSampleCount() );
        
        if (fuResultsList[idx]->outstandingDataDiscardCount != 0)
        {
          tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
        }
        else
        {
          tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        }
        maker.addInt( tableDiv, fuResultsList[idx]->outstandingDataDiscardCount );
        
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addInt( tableDiv, fuResultsList[idx]->dqmEventStats.getSampleCount() );
        
        if (fuResultsList[idx]->faultyDQMEventStats.getSampleCount() != 0)
        {
          tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
        }
        else
        {
          tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        }
        maker.addInt( tableDiv, fuResultsList[idx]->faultyDQMEventStats.getSampleCount() );
        
        if (fuResultsList[idx]->outstandingDQMDiscardCount != 0)
        {
          tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
        }
        else
        {
          tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        }
        maker.addInt( tableDiv, fuResultsList[idx]->outstandingDQMDiscardCount );
        
        const int skippedDiscards = fuResultsList[idx]->skippedDiscardStats.getSampleCount();
        if (skippedDiscards != 0)
        {
          tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
        }
        else
        {
          tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        }
        maker.addInt( tableDiv, skippedDiscards );
        
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addDouble( tableDiv, fuResultsList[idx]->shortIntervalEventStats.
          getSampleRate(MonitoredQuantity::RECENT) );
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addInt( tableDiv, fuResultsList[idx]->lastEventNumber );
        tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
        maker.addInt( tableDiv, fuResultsList[idx]->lastRunNumber );
      }
    }
  }
 
} // namespace stor
 

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
