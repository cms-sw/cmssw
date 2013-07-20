// $Id: SMPSWebPageHelper.cc,v 1.3 2011/05/09 11:03:34 mommsen Exp $
/// @file: SMPSWebPageHelper.cc

#include "EventFilter/SMProxyServer/interface/SMPSWebPageHelper.h"
#include "EventFilter/StorageManager/interface/RegistrationCollection.h"
#include "EventFilter/StorageManager/interface/Utils.h"
#include "EventFilter/StorageManager/interface/XHTMLMonitor.h"
#include "EventFilter/StorageManager/src/ConsumerWebPageHelper.icc"

#include <boost/pointer_cast.hpp>


namespace smproxy
{
  SMPSWebPageHelper::SMPSWebPageHelper
  (
    xdaq::ApplicationDescriptor* appDesc,
    StateMachinePtr stateMachine
  ) :
  stor::WebPageHelper<SMPSWebPageHelper>(appDesc, "$Name: CMSSW_6_2_0 $", this, &smproxy::SMPSWebPageHelper::addDOMforHyperLinks),
  stateMachine_(stateMachine),
  consumerWebPageHelper_(appDesc, "$Name: CMSSW_6_2_0 $", this, &smproxy::SMPSWebPageHelper::addDOMforHyperLinks)
  { }
  
  
  void SMPSWebPageHelper::defaultWebPage(xgi::Output* out) const
  {
    stor::XHTMLMonitor theMonitor;
    stor::XHTMLMaker maker;

    stor::XHTMLMaker::Node* body = createWebPageBody(maker,
      "Main",
      stateMachine_->getExternallyVisibleStateName(),
      stateMachine_->getStateName(),
      stateMachine_->getReasonForFailed()
    );

    DataRetrieverMonitorCollection::SummaryStats summaryStats;
    stateMachine_->getStatisticsReporter()->
      getDataRetrieverMonitorCollection().getSummaryStats(summaryStats);

    addDOMforConnectionInfo(maker, body, summaryStats);

    maker.addNode("br", body);

    addDOMforThroughputPerEventType(maker, body, summaryStats);
    
    addDOMforHyperLinks(maker, body);
    
    // Dump the webpage to the output stream
    maker.out(*out);
  }
  
  
  void SMPSWebPageHelper::dataRetrieverWebPage(xgi::Output* out) const
  {
    stor::XHTMLMonitor theMonitor;
    stor::XHTMLMaker maker;

    stor::XHTMLMaker::Node* body = createWebPageBody(maker,
      "Data Retrieval",
      stateMachine_->getExternallyVisibleStateName(),
      stateMachine_->getStateName(),
      stateMachine_->getReasonForFailed()
    );

    addDOMforEventServers(maker, body);
    
    maker.addNode("hr", body);
    
    addDOMforDQMEventServers(maker, body);
    
    addDOMforHyperLinks(maker, body);
    
    // Dump the webpage to the output stream
    maker.out(*out);    
  } 
  
  
  void SMPSWebPageHelper::consumerStatisticsWebPage(xgi::Output* out) const
  {
    consumerWebPageHelper_.consumerStatistics(out,
      stateMachine_->getExternallyVisibleStateName(),
      stateMachine_->getStateName(),
      stateMachine_->getReasonForFailed(),
      stateMachine_->getStatisticsReporter(),
      stateMachine_->getRegistrationCollection(),
      stateMachine_->getEventQueueCollection(),
      stateMachine_->getDQMEventQueueCollection()
    );
  }
  
  
  void SMPSWebPageHelper::dqmEventStatisticsWebPage(xgi::Output* out) const
  {
    stor::XHTMLMonitor theMonitor;
    stor::XHTMLMaker maker;

    stor::XHTMLMaker::Node* body = createWebPageBody(maker,
      "DQM Event Processor",
      stateMachine_->getExternallyVisibleStateName(),
      stateMachine_->getStateName(),
      stateMachine_->getReasonForFailed()
    );
    
    const stor::DQMEventMonitorCollection& demc =
      stateMachine_->getStatisticsReporter()->getDQMEventMonitorCollection();
    addDOMforProcessedDQMEvents(maker, body, demc);
    addDOMforDQMEventStatistics(maker, body, demc);

    addDOMforHyperLinks(maker, body);
    
    // Dump the webpage to the output stream
    maker.out(*out);    
  } 
  
  
  void SMPSWebPageHelper::addDOMforHyperLinks
  (
    stor::XHTMLMaker& maker,
    stor::XHTMLMaker::Node *parent
  ) const
  {
    std::string url = appDescriptor_->getContextDescriptor()->getURL()
      + "/" + appDescriptor_->getURN();
    
    stor::XHTMLMaker::AttrMap linkAttr;
    stor::XHTMLMaker::Node *link;
    
    maker.addNode("hr", parent);
    
    linkAttr[ "href" ] = url;
    link = maker.addNode("a", parent, linkAttr);
    maker.addText(link, "Main web page");
    
    maker.addNode("hr", parent);
    
    linkAttr[ "href" ] = url + "/dataRetriever";
    link = maker.addNode("a", parent, linkAttr);
    maker.addText(link, "Data retriever web page");
    
    maker.addNode("hr", parent);
    
    linkAttr[ "href" ] = url + "/dqmEventStatistics";
    link = maker.addNode("a", parent, linkAttr);
    maker.addText(link, "DQM event processor statistics");
    
    maker.addNode("hr", parent);
    
    linkAttr[ "href" ] = url + "/consumerStatistics";
    link = maker.addNode("a", parent, linkAttr);
    maker.addText(link, "Consumer Statistics");
    
    maker.addNode("hr", parent);
  }
  
  
  void SMPSWebPageHelper::addDOMforConnectionInfo
  (
    stor::XHTMLMaker& maker,
    stor::XHTMLMaker::Node* parent,
    const DataRetrieverMonitorCollection::SummaryStats& summaryStats
  ) const
  {
    stor::XHTMLMaker::AttrMap colspanAttr;
    colspanAttr[ "colspan" ] = "2";

    stor::XHTMLMaker::AttrMap tableAttr = tableAttr_;
    tableAttr[ "width" ] = "50%";

    stor::XHTMLMaker::AttrMap widthAttr;
    widthAttr[ "width" ] = "70%";
 
    stor::XHTMLMaker::Node* table = maker.addNode("table", parent, tableAttr);
    
    stor::XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    stor::XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
    maker.addText(tableDiv, "Connection Information");

    // # of configured SMs
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, widthAttr);
    maker.addText(tableDiv, "# of configured StorageManagers");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    const size_t configuredSMs = stateMachine_->getConfiguration()->
      getDataRetrieverParams().smRegistrationList_.size();
    maker.addInt(tableDiv, configuredSMs);

    // # of requested SMs connections
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, widthAttr);
    maker.addText(tableDiv, "# of requested SM connections");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt(tableDiv, summaryStats.registeredSMs);

    // # of active SMs connections
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, widthAttr);
    maker.addText(tableDiv, "# of active SM connections");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addInt(tableDiv, summaryStats.activeSMs);

    // # of connected event consumers
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, widthAttr);
    maker.addText(tableDiv, "# of connected event consumers");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    stor::RegistrationCollection::ConsumerRegistrations consumers;
    stateMachine_->getRegistrationCollection()->
      getEventConsumers(consumers);
    maker.addInt(tableDiv, consumers.size());

    // # of connected DQM event (histogram) consumers
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("td", tableRow, widthAttr);
    maker.addText(tableDiv, "# of connected histogram consumers");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    stor::RegistrationCollection::DQMConsumerRegistrations dqmConsumers;
    stateMachine_->getRegistrationCollection()->
      getDQMEventConsumers(dqmConsumers);
    maker.addInt(tableDiv, dqmConsumers.size());
  }
  
  
  void SMPSWebPageHelper::addDOMforThroughputPerEventType
  (
    stor::XHTMLMaker& maker,
    stor::XHTMLMaker::Node* parent,
    const DataRetrieverMonitorCollection::SummaryStats& summaryStats
  ) const
  {
    stor::XHTMLMaker::AttrMap colspanAttr;
    colspanAttr[ "colspan" ] = "13";
    
    stor::XHTMLMaker::AttrMap subColspanAttr;
    
    stor::XHTMLMaker::AttrMap rowspanAttr;
    rowspanAttr[ "rowspan" ] = "3";
    rowspanAttr[ "width" ] = "30%";

    stor::XHTMLMaker::AttrMap noWrapAttr; 
    noWrapAttr[ "style" ] = "white-space: nowrap;";

    stor::XHTMLMaker::Node* table = maker.addNode("table", parent, tableAttr_);
    stor::XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    stor::XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
    maker.addText(tableDiv, "Throughput");
    
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("th", tableRow, rowspanAttr);
    maker.addText(tableDiv, "Requested Event Type");
    subColspanAttr[ "colspan" ] = "2";
    tableDiv = maker.addNode("th", tableRow, subColspanAttr);
    maker.addText(tableDiv, "");
    subColspanAttr[ "colspan" ] = "6";
    tableDiv = maker.addNode("th", tableRow, subColspanAttr);
    maker.addText(tableDiv, "Input");
    tableDiv = maker.addNode("th", tableRow, subColspanAttr);
    maker.addText(tableDiv, "Output");

    subColspanAttr[ "colspan" ] = "2";
    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("th", tableRow, subColspanAttr);
    maker.addText(tableDiv, "Average Event Size (kB)");
    tableDiv = maker.addNode("th", tableRow, subColspanAttr);
    maker.addText(tableDiv, "Event Rate (Hz)");
    tableDiv = maker.addNode("th", tableRow, subColspanAttr);
    maker.addText(tableDiv, "Bandwidth (kB/s)");
    tableDiv = maker.addNode("th", tableRow, subColspanAttr);
    maker.addText(tableDiv, "Corrupted Event Rate (Hz)");
    tableDiv = maker.addNode("th", tableRow, subColspanAttr);
    maker.addText(tableDiv, "Event Rate (Hz)");
    tableDiv = maker.addNode("th", tableRow, subColspanAttr);
    maker.addText(tableDiv, "Bandwidth (kB/s)");

    tableRow = maker.addNode("tr", table, rowAttr_);
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "overall");
    tableDiv = maker.addNode("th", tableRow, noWrapAttr);
    maker.addText(tableDiv, "last 60 s");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "overall");
    tableDiv = maker.addNode("th", tableRow, noWrapAttr);
    maker.addText(tableDiv, "last 60 s");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "overall");
    tableDiv = maker.addNode("th", tableRow, noWrapAttr);
    maker.addText(tableDiv, "last 60 s");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "overall");
    tableDiv = maker.addNode("th", tableRow, noWrapAttr);
    maker.addText(tableDiv, "last 60 s");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "overall");
    tableDiv = maker.addNode("th", tableRow, noWrapAttr);
    maker.addText(tableDiv, "last 60 s");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "overall");
    tableDiv = maker.addNode("th", tableRow, noWrapAttr);
    maker.addText(tableDiv, "last 60 s");

    if ( summaryStats.eventTypeStats.empty() )
    {
      stor::XHTMLMaker::AttrMap messageAttr = colspanAttr;
      messageAttr[ "align" ] = "center";
      
      stor::XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
      stor::XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow, messageAttr);
      maker.addText(tableDiv, "No data flowing, yet.");
      return;
    }
    
    bool evenRow = false;
    
    for (DataRetrieverMonitorCollection::SummaryStats::EventTypeStatList::const_iterator
           it = summaryStats.eventTypeStats.begin(), itEnd = summaryStats.eventTypeStats.end();
         it != itEnd; ++it)
    {
      stor::XHTMLMaker::AttrMap rowAttr = rowAttr_;
      if( evenRow )
      {
        rowAttr[ "style" ] = "background-color:#e0e0e0;";
        evenRow = false;
      }
       else
      {
        evenRow = true;
      }
      stor::XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr);
      addRowForEventType(maker, tableRow, *it);
    }

    addSummaryRowForThroughput(maker, table, summaryStats);
  }
  
  
  void SMPSWebPageHelper::addRowForEventType
  (
    stor::XHTMLMaker& maker,
    stor::XHTMLMaker::Node* tableRow,
    DataRetrieverMonitorCollection::SummaryStats::EventTypeStats const& stats
  ) const
  {
    // Event type
    stor::XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
    stor::XHTMLMaker::Node* pre = maker.addNode("pre", tableDiv);
    std::ostringstream eventType;
    stats.first->eventType(eventType);
    maker.addText(pre, eventType.str());
    
    // Average event size
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, stats.second.sizeStats.getValueAverage(stor::MonitoredQuantity::FULL));
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, stats.second.sizeStats.getValueAverage(stor::MonitoredQuantity::RECENT));

    // Input event rate
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, stats.second.sizeStats.getSampleRate(stor::MonitoredQuantity::FULL));
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, stats.second.sizeStats.getSampleRate(stor::MonitoredQuantity::RECENT));

    // Input bandwidth
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, stats.second.sizeStats.getValueRate(stor::MonitoredQuantity::FULL));
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, stats.second.sizeStats.getValueRate(stor::MonitoredQuantity::RECENT));

    // Input corrupted events rate
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv,
      stats.second.corruptedEventsStats.getSampleRate(stor::MonitoredQuantity::FULL));
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv,
      stats.second.corruptedEventsStats.getSampleRate(stor::MonitoredQuantity::RECENT));

    // Get statistics for consumers requesting this event type
    stor::QueueIDs queueIDs;
    bool isEventConsumer =
      stateMachine_->getDataManager()->getQueueIDsFromDataEventRetrievers(
        boost::dynamic_pointer_cast<stor::EventConsumerRegistrationInfo>(stats.first),
        queueIDs
      );
    if ( ! isEventConsumer)
    {
      stateMachine_->getDataManager()->getQueueIDsFromDQMEventRetrievers(
        boost::dynamic_pointer_cast<stor::DQMEventConsumerRegistrationInfo>(stats.first),
        queueIDs
      );
    }

    if ( queueIDs.empty() )
    {
      stor::XHTMLMaker::AttrMap noConsumersAttr = tableLabelAttr_;
      noConsumersAttr[ "colspan" ] = "4";
      tableDiv = maker.addNode("td", tableRow, noConsumersAttr);
      maker.addText(tableDiv, "no consumers connected");
      return;
    }
    
    const stor::EventConsumerMonitorCollection& ecmc =
      stateMachine_->getStatisticsReporter()->getEventConsumerMonitorCollection();
    const stor::DQMConsumerMonitorCollection& dcmc =
      stateMachine_->getStatisticsReporter()->getDQMConsumerMonitorCollection();
    const stor::ConsumerMonitorCollection& cmc =
      isEventConsumer ?
      static_cast<const stor::ConsumerMonitorCollection&>(ecmc) :
      static_cast<const stor::ConsumerMonitorCollection&>(dcmc);
    
    double rateOverall = 0;
    double rateRecent = 0;
    double bandwidthOverall = 0;
    double bandwidthRecent = 0;

    for ( stor::QueueIDs::const_iterator it = queueIDs.begin(),
            itEnd = queueIDs.end(); it != itEnd; ++it)
    {
      stor::MonitoredQuantity::Stats result;
      if ( cmc.getServed(*it, result) )
      {
        rateOverall += result.getSampleRate(stor::MonitoredQuantity::FULL);
        rateRecent += result.getSampleRate(stor::MonitoredQuantity::RECENT);
        bandwidthOverall += result.getValueRate(stor::MonitoredQuantity::FULL) / 1024;
        bandwidthRecent += result.getValueRate(stor::MonitoredQuantity::RECENT) / 1024;
      }
    }

    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, rateOverall);
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, rateRecent);
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, bandwidthOverall);
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, bandwidthRecent);
  }
  
  
  void SMPSWebPageHelper::addSummaryRowForThroughput
  (
    stor::XHTMLMaker& maker,
    stor::XHTMLMaker::Node* table,
    const DataRetrieverMonitorCollection::SummaryStats& summaryStats
  ) const
  {
    stor::XHTMLMaker::Node* tableRow = maker.addNode("tr", table, specialRowAttr_);
    stor::XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, "Total");

    // Average event size
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, summaryStats.totals.sizeStats.getValueAverage(stor::MonitoredQuantity::FULL));
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, summaryStats.totals.sizeStats.getValueAverage(stor::MonitoredQuantity::RECENT));

    // Input event rate
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, summaryStats.totals.sizeStats.getSampleRate(stor::MonitoredQuantity::FULL));
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, summaryStats.totals.sizeStats.getSampleRate(stor::MonitoredQuantity::RECENT));

    // Input bandwidth
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, summaryStats.totals.sizeStats.getValueRate(stor::MonitoredQuantity::FULL));
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, summaryStats.totals.sizeStats.getValueRate(stor::MonitoredQuantity::RECENT));

    // Input corrupted events rate
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, summaryStats.totals.corruptedEventsStats.getValueRate(stor::MonitoredQuantity::FULL));
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, summaryStats.totals.corruptedEventsStats.getValueRate(stor::MonitoredQuantity::RECENT));
    
    const stor::EventConsumerMonitorCollection& ecmc =
      stateMachine_->getStatisticsReporter()->getEventConsumerMonitorCollection();
    const stor::DQMConsumerMonitorCollection& dcmc =
      stateMachine_->getStatisticsReporter()->getDQMConsumerMonitorCollection();
    stor::ConsumerMonitorCollection::TotalStats ecmcStats, dcmcStats;
    ecmc.getTotalStats(ecmcStats);
    dcmc.getTotalStats(dcmcStats);

    // Output event rate
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv,
      ecmcStats.servedStats.getSampleRate(stor::MonitoredQuantity::FULL) +
      dcmcStats.servedStats.getSampleRate(stor::MonitoredQuantity::FULL)
    );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv,
      ecmcStats.servedStats.getSampleRate(stor::MonitoredQuantity::RECENT) +
      dcmcStats.servedStats.getSampleRate(stor::MonitoredQuantity::RECENT)
    );
    
    // Output bandwidth
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv,
      (ecmcStats.servedStats.getValueRate(stor::MonitoredQuantity::FULL) +
        dcmcStats.servedStats.getValueRate(stor::MonitoredQuantity::FULL)) / 1024
    );
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv,
      (ecmcStats.servedStats.getValueRate(stor::MonitoredQuantity::RECENT) +
        dcmcStats.servedStats.getValueRate(stor::MonitoredQuantity::RECENT)) / 1024
    );
  }
  
  
  void SMPSWebPageHelper::addDOMforEventServers
  (
    stor::XHTMLMaker& maker,
    stor::XHTMLMaker::Node* parent
  ) const
  {
    stor::XHTMLMaker::AttrMap colspanAttr;
    colspanAttr[ "colspan" ] = "14";
    
    stor::XHTMLMaker::Node* table = maker.addNode("table", parent, tableAttr_);
    
    stor::XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr_);
    stor::XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
    maker.addText(tableDiv, "Event Servers");
    
    stor::XHTMLMaker::AttrMap rowspanAttr;
    rowspanAttr[ "rowspan" ] = "2";
    
    stor::XHTMLMaker::AttrMap subColspanAttr;
    subColspanAttr[ "colspan" ] = "2";

    stor::XHTMLMaker::AttrMap noWrapAttr; 
    noWrapAttr[ "style" ] = "white-space: nowrap;";
   
    // Header
    tableRow = maker.addNode("tr", table, specialRowAttr_);
    tableDiv = maker.addNode("th", tableRow, rowspanAttr);
    maker.addText(tableDiv, "Hostname");
    tableDiv = maker.addNode("th", tableRow, rowspanAttr);
    maker.addText(tableDiv, "Status");
    tableDiv = maker.addNode("th", tableRow, rowspanAttr);
    maker.addText(tableDiv, "Requested Event Type");
    tableDiv = maker.addNode("th", tableRow, rowspanAttr);
    maker.addText(tableDiv, "Max Request Rate (Hz)");
    tableDiv = maker.addNode("th", tableRow, subColspanAttr);
    maker.addText(tableDiv, "Event Rate (Hz)");
    tableDiv = maker.addNode("th", tableRow, subColspanAttr);
    maker.addText(tableDiv, "Average Event Size (kB)");
    tableDiv = maker.addNode("th", tableRow, subColspanAttr);
    maker.addText(tableDiv, "Bandwidth (kB/s)");
    tableDiv = maker.addNode("th", tableRow, subColspanAttr);
    maker.addText(tableDiv, "Corrupted Events");
    tableDiv = maker.addNode("th", tableRow, subColspanAttr);
    maker.addText(tableDiv, "Corrupted Event Rate (Hz)");

    tableRow = maker.addNode("tr", table, specialRowAttr_);
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "overall");
    tableDiv = maker.addNode("th", tableRow, noWrapAttr);
    maker.addText(tableDiv, "last 60 s");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "overall");
    tableDiv = maker.addNode("th", tableRow, noWrapAttr);
    maker.addText(tableDiv, "last 60 s");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "overall");
    tableDiv = maker.addNode("th", tableRow, noWrapAttr);
    maker.addText(tableDiv, "last 60 s");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "overall");
    tableDiv = maker.addNode("th", tableRow, noWrapAttr);
    maker.addText(tableDiv, "last 60 s");
    tableDiv = maker.addNode("th", tableRow);
    maker.addText(tableDiv, "overall");
    tableDiv = maker.addNode("th", tableRow, noWrapAttr);
    maker.addText(tableDiv, "last 60 s");

    DataRetrieverMonitorCollection::EventTypePerConnectionStatList eventTypePerConnectionStats;
    stateMachine_->getStatisticsReporter()->getDataRetrieverMonitorCollection()
      .getStatsByEventTypesPerConnection(eventTypePerConnectionStats);

    DataRetrieverMonitorCollection::ConnectionStats connectionStats;
    stateMachine_->getStatisticsReporter()->getDataRetrieverMonitorCollection()
      .getStatsByConnection(connectionStats);

    if ( eventTypePerConnectionStats.empty() )
    {
      stor::XHTMLMaker::AttrMap messageAttr = colspanAttr;
      messageAttr[ "align" ] = "center";

      tableRow = maker.addNode("tr", table, rowAttr_);
      tableDiv = maker.addNode("td", tableRow, messageAttr);
      maker.addText(tableDiv, "Not registered to any event servers yet");
      return;
    }

    bool evenRow = false;

    for (DataRetrieverMonitorCollection::EventTypePerConnectionStatList::const_iterator
           it = eventTypePerConnectionStats.begin(), itEnd = eventTypePerConnectionStats.end();
         it != itEnd; ++it)
    {
      stor::XHTMLMaker::AttrMap rowAttr = rowAttr_;
      if( evenRow )
      {
        rowAttr[ "style" ] = "background-color:#e0e0e0;";
        evenRow = false;
      }
      else
      {
        evenRow = true;
      }
      stor::XHTMLMaker::Node* tableRow = maker.addNode("tr", table, rowAttr);
      addRowForEventServer(maker, tableRow, *it);
      
      const std::string currentSourceURL = it->regPtr->sourceURL();

      if ( (it+1) == eventTypePerConnectionStats.end() ||
        (it+1)->regPtr->sourceURL() != currentSourceURL )
      {
        addSummaryRowForEventServer(maker, table,
          connectionStats.find(currentSourceURL));
      }
    }
  }
  
  
  void SMPSWebPageHelper::addRowForEventServer
  (
    stor::XHTMLMaker& maker,
    stor::XHTMLMaker::Node* tableRow,
    DataRetrieverMonitorCollection::EventTypePerConnectionStats const& stats
  ) const
  {
    stor::XHTMLMaker::Node* tableDiv;

    // Hostname
    addDOMforSMhost(maker, tableRow, stats.regPtr->sourceURL());

    // Status
    if ( stats.connectionStatus == DataRetrieverMonitorCollection::CONNECTED )
    {
      tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
      maker.addText(tableDiv, "Connected");
    }
    else
    {
      stor::XHTMLMaker::AttrMap statusAttr = tableLabelAttr_;
      statusAttr[ "style" ] = "color:brown;";
      tableDiv = maker.addNode("td", tableRow, statusAttr);
      std::ostringstream status;
      status << stats.connectionStatus;
      maker.addText(tableDiv, status.str());
    }

    // Requested event type
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    stor::XHTMLMaker::Node* pre = maker.addNode("pre", tableDiv);
    std::ostringstream eventType;
    stats.regPtr->eventType(eventType);
    maker.addText(pre, eventType.str());

    // Max request rate
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    const stor::utils::Duration_t interval =
      stats.regPtr->minEventRequestInterval();
    if ( interval.is_not_a_date_time() )
      maker.addText(tableDiv, "unlimited");
    else
      maker.addDouble(tableDiv, 1 / stor::utils::durationToSeconds(interval), 1);

    // Event rate
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, stats.eventStats.sizeStats.getSampleRate(stor::MonitoredQuantity::FULL));
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, stats.eventStats.sizeStats.getSampleRate(stor::MonitoredQuantity::RECENT));

    // Average event size
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, stats.eventStats.sizeStats.getValueAverage(stor::MonitoredQuantity::FULL));
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, stats.eventStats.sizeStats.getValueAverage(stor::MonitoredQuantity::RECENT));
    
    // Bandwidth
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, stats.eventStats.sizeStats.getValueRate(stor::MonitoredQuantity::FULL));
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, stats.eventStats.sizeStats.getValueRate(stor::MonitoredQuantity::RECENT));
    
    // Corrupted events counts
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv,
      stats.eventStats.corruptedEventsStats.getValueSum(stor::MonitoredQuantity::FULL), 0);
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv,
      stats.eventStats.corruptedEventsStats.getValueSum(stor::MonitoredQuantity::RECENT), 0);
    
    // Corrupted event rate
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv,
      stats.eventStats.corruptedEventsStats.getValueRate(stor::MonitoredQuantity::FULL));
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv,
      stats.eventStats.corruptedEventsStats.getValueRate(stor::MonitoredQuantity::RECENT));
  }
  
  
  void SMPSWebPageHelper::addSummaryRowForEventServer
  (
    stor::XHTMLMaker& maker,
    stor::XHTMLMaker::Node* table,
    DataRetrieverMonitorCollection::ConnectionStats::const_iterator pos
  ) const
  {
    stor::XHTMLMaker::Node* tableRow = maker.addNode("tr", table, specialRowAttr_);
    
    // Hostname
    addDOMforSMhost(maker, tableRow, pos->first);

    // Status, requst type, and max request rate not filled
    stor::XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
    tableDiv = maker.addNode("td", tableRow);
    tableDiv = maker.addNode("td", tableRow);
    
    // Event rate
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, pos->second.sizeStats.getSampleRate(stor::MonitoredQuantity::FULL));
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, pos->second.sizeStats.getSampleRate(stor::MonitoredQuantity::RECENT));

    // Average event size
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, pos->second.sizeStats.getValueAverage(stor::MonitoredQuantity::FULL));
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, pos->second.sizeStats.getValueAverage(stor::MonitoredQuantity::RECENT));
    
    // Bandwidth
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, pos->second.sizeStats.getValueRate(stor::MonitoredQuantity::FULL));
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv, pos->second.sizeStats.getValueRate(stor::MonitoredQuantity::RECENT));
    
    // Corrupted events counts
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv,
      pos->second.corruptedEventsStats.getValueSum(stor::MonitoredQuantity::FULL), 0);
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv,
      pos->second.corruptedEventsStats.getValueSum(stor::MonitoredQuantity::RECENT), 0);
    
    // Corrupted event rate
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv,
      pos->second.corruptedEventsStats.getValueRate(stor::MonitoredQuantity::FULL));
    tableDiv = maker.addNode("td", tableRow, tableValueAttr_);
    maker.addDouble(tableDiv,
      pos->second.corruptedEventsStats.getValueRate(stor::MonitoredQuantity::RECENT));
  }
  
  
  void SMPSWebPageHelper::addDOMforSMhost
  (
    stor::XHTMLMaker& maker,
    stor::XHTMLMaker::Node* tableRow,
    const std::string& sourceURL
  ) const
  {
    stor::XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow, tableLabelAttr_);
    std::string::size_type startPos = sourceURL.find("//");
    if ( startPos == std::string::npos )
      startPos = 0; 
    else
      startPos += 2;
    const std::string::size_type endPos = sourceURL.find('.');
    const std::string hostname = sourceURL.substr(startPos,(endPos-startPos));
    stor::XHTMLMaker::AttrMap linkAttr;
    linkAttr[ "href" ] = sourceURL + "/consumerStatistics";
    stor::XHTMLMaker::Node* link = maker.addNode("a", tableDiv, linkAttr);
    maker.addText(link, hostname);
  }
  
  
  void SMPSWebPageHelper::addDOMforDQMEventServers
  (
    stor::XHTMLMaker& maker,
    stor::XHTMLMaker::Node* parent
  ) const
  {
  }
  
} // namespace smproxy


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
