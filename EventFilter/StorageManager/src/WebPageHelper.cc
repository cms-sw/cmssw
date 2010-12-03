// $Id: WebPageHelper.cc,v 1.52 2010/12/03 16:01:14 mommsen Exp $
/// @file: WebPageHelper.cc

#ifdef __APPLE__
#include <sys/param.h>
#include <sys/mount.h>
#else
#include <sys/statfs.h>
#endif

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdio.h>

#include "boost/lexical_cast.hpp"

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/DQMConsumerMonitorCollection.h"
#include "EventFilter/StorageManager/interface/DQMEventMonitorCollection.h"
#include "EventFilter/StorageManager/interface/EventConsumerMonitorCollection.h"
#include "EventFilter/StorageManager/interface/FilesMonitorCollection.h"
#include "EventFilter/StorageManager/interface/FragmentMonitorCollection.h"
#include "EventFilter/StorageManager/interface/MonitoredQuantity.h"
#include "EventFilter/StorageManager/interface/RegistrationCollection.h"
#include "EventFilter/StorageManager/interface/ResourceMonitorCollection.h"
#include "EventFilter/StorageManager/interface/RunMonitorCollection.h"
#include "EventFilter/StorageManager/interface/StreamsMonitorCollection.h"
#include "EventFilter/StorageManager/interface/WebPageHelper.h"
#include "EventFilter/StorageManager/interface/XHTMLMonitor.h"

using namespace stor;


boost::mutex WebPageHelper::_xhtmlMakerMutex;

WebPageHelper::WebPageHelper
(
  xdaq::ApplicationDescriptor* appDesc
) :
_appDescriptor(appDesc)
{
  // set application icon for hyperdaq
  appDesc->setAttribute("icon", "/evf/images/smicon.jpg");

  _tableAttr[ "frame" ] = "void";
  _tableAttr[ "rules" ] = "group";
  _tableAttr[ "class" ] = "states";
  _tableAttr[ "cellspacing" ] = "0";
  _tableAttr[ "cellpadding" ] = "2";
  _tableAttr[ "width" ] = "100%";
  _tableAttr[ "valign" ] = "top";

  _rowAttr[ "valign" ] = "top";
  
  _specialRowAttr = _rowAttr;
  _specialRowAttr[ "class" ] = "special";

  _alarmColors[ AlarmHandler::OKAY ] = "#FFFFFF";
  _alarmColors[ AlarmHandler::WARNING ] = "#FFE635";
  _alarmColors[ AlarmHandler::ERROR ] = "#FF9F36";
  _alarmColors[ AlarmHandler::FATAL ] = "#FF2338";

  _tableLabelAttr[ "align" ] = "left";

  _tableValueAttr[ "align" ] = "right";
}


void WebPageHelper::defaultWebPage
(
  xgi::Output *out, 
  const SharedResourcesPtr sharedResources
)
{
  boost::mutex::scoped_lock lock(_xhtmlMakerMutex);
  XHTMLMonitor theMonitor;
  XHTMLMaker maker;
  
  StatisticsReporterPtr statReporter = sharedResources->_statisticsReporter;

  // Create the body with the standard header
  XHTMLMaker::Node* body = createWebPageBody(maker, "Main", sharedResources);

  // Run and event summary
  addDOMforRunMonitor(maker, body, statReporter->getRunMonitorCollection());
  
  // Resource usage
  addDOMforResourceUsage(maker, body, 
    statReporter->getResourceMonitorCollection(),
    statReporter->getThroughputMonitorCollection());
  
  // Add the received data statistics table
  addDOMforFragmentMonitor(maker, body,
                           statReporter->getFragmentMonitorCollection());

  addDOMforSMLinks(maker, body);
  
  // Dump the webpage to the output stream
  maker.out(*out);
}


void WebPageHelper::storedDataWebPage
(
  xgi::Output *out,
  const SharedResourcesPtr sharedResources
)
{
  boost::mutex::scoped_lock lock(_xhtmlMakerMutex);
  XHTMLMonitor theMonitor;
  XHTMLMaker maker;
  
  StatisticsReporterPtr statReporter = sharedResources->_statisticsReporter;
  
  // Create the body with the standard header
  XHTMLMaker::Node* body = createWebPageBody(maker, "Stored Data", sharedResources); 

  addDOMforStoredData(maker, body, statReporter->getStreamsMonitorCollection());

  maker.addNode("hr", body);

  addDOMforConfigString(maker, body, sharedResources->_configuration->getDiskWritingParams());  
  
  addDOMforSMLinks(maker, body);
  
   // Dump the webpage to the output stream
  maker.out(*out);
}


void WebPageHelper::filesWebPage
(
  xgi::Output *out,
  const SharedResourcesPtr sharedResources
)
{
  boost::mutex::scoped_lock lock(_xhtmlMakerMutex);
  XHTMLMonitor theMonitor;
  XHTMLMaker maker;
  
  StatisticsReporterPtr statReporter = sharedResources->_statisticsReporter;
  
  // Create the body with the standard header
  XHTMLMaker::Node* body = createWebPageBody(maker, "Files", sharedResources);

  addDOMforFiles(maker, body, statReporter->getFilesMonitorCollection());  

  addDOMforSMLinks(maker, body);
  
  // Dump the webpage to the output stream
  maker.out(*out);
}


//////////////////////////////
//// Consumer statistics: ////
//////////////////////////////
void WebPageHelper::consumerStatistics
(
  xgi::Output* out,
  const SharedResourcesPtr sharedResources
)
{

  // Get lock, initialize maker:
  boost::mutex::scoped_lock lock(_xhtmlMakerMutex);
  XHTMLMonitor theMonitor;
  XHTMLMaker maker;

  // Create the body with the standard header
  XHTMLMaker::Node* body = createWebPageBody(maker, "Consumer Statistics", sharedResources);

  addDOMforEventConsumers(maker, body, sharedResources);

  maker.addNode("hr", body);

  addDOMforDQMEventConsumers(maker, body, sharedResources);

  addDOMforSMLinks(maker, body);
  
  // Dump the webpage to the output stream
  maker.out(*out);
}


void WebPageHelper::resourceBrokerOverview
(
  xgi::Output *out,
  const SharedResourcesPtr sharedResources
)
{
  boost::mutex::scoped_lock lock(_xhtmlMakerMutex);
  XHTMLMonitor theMonitor;
  XHTMLMaker maker;
  
  StatisticsReporterPtr statReporter = sharedResources->_statisticsReporter;
  
  // Create the body with the standard header
  XHTMLMaker::Node* body = createWebPageBody(maker, "Resource Broker Overview", sharedResources);

  addOutputModuleTables(maker, body,
                        statReporter->getDataSenderMonitorCollection());  

  maker.addNode("hr", body);

  addResourceBrokerList(maker, body,
                        statReporter->getDataSenderMonitorCollection());

  addDOMforSMLinks(maker, body);
  
   // Dump the webpage to the output stream
  maker.out(*out);
}


void WebPageHelper::resourceBrokerDetail
(
  xgi::Output *out,
  const SharedResourcesPtr sharedResources,
  long long uniqueRBID
)
{
  boost::mutex::scoped_lock lock(_xhtmlMakerMutex);
  XHTMLMonitor theMonitor;
  XHTMLMaker maker;
  
  StatisticsReporterPtr statReporter = sharedResources->_statisticsReporter;
  
  // Create the body with the standard header
  std::ostringstream pageTitle;
  pageTitle << "Resource Broker " << uniqueRBID << " Detail";
  XHTMLMaker::Node* body = createWebPageBody(maker, pageTitle.str(), sharedResources);

  addResourceBrokerDetails(maker, body, uniqueRBID,
                           statReporter->getDataSenderMonitorCollection());  

  addOutputModuleStatistics(maker, body, uniqueRBID,
                            statReporter->getDataSenderMonitorCollection());  

  addFilterUnitList(maker, body, uniqueRBID,
                    statReporter->getDataSenderMonitorCollection());  

  addDOMforSMLinks(maker, body);
  
   // Dump the webpage to the output stream
  maker.out(*out);
}


void WebPageHelper::dqmEventWebPage
(
  xgi::Output *out,
  const SharedResourcesPtr sharedResources
)
{
  boost::mutex::scoped_lock lock(_xhtmlMakerMutex);
  XHTMLMonitor theMonitor;
  XHTMLMaker maker;
  
  StatisticsReporterPtr statReporter = sharedResources->_statisticsReporter;
  
  // Create the body with the standard header
  XHTMLMaker::Node* body = createWebPageBody(maker, "DQM Event Processor", sharedResources);

  addDOMforProcessedDQMEvents(maker, body, statReporter->getDQMEventMonitorCollection());  
  addDOMforDQMEventStatistics(maker, body, statReporter->getDQMEventMonitorCollection());  

  addDOMforSMLinks(maker, body);
  
   // Dump the webpage to the output stream
  maker.out(*out);
}


void WebPageHelper::throughputWebPage
(
  xgi::Output *out,
  const SharedResourcesPtr sharedResources
)
{
  boost::mutex::scoped_lock lock(_xhtmlMakerMutex);
  XHTMLMonitor theMonitor;
  XHTMLMaker maker;

  StatisticsReporterPtr statReporter = sharedResources->_statisticsReporter;

  // Create the body with the standard header
  XHTMLMaker::Node* body = createWebPageBody(maker, "Throughput", sharedResources);

  addDOMforThroughputStatistics(maker, body, statReporter->getThroughputMonitorCollection());  

  addDOMforSMLinks(maker, body);

  // Dump the webpage to the output stream
  maker.out(*out);
}


///////////////////////
//// Get base URL: ////
///////////////////////
std::string WebPageHelper::baseURL() const
{
  return _appDescriptor->getContextDescriptor()->getURL() + "/" + _appDescriptor->getURN();
}


XHTMLMaker::Node* WebPageHelper::createWebPageBody
(
  XHTMLMaker& maker,
  const std::string& pageTitle,
  const SharedResourcesPtr sharedResources
)
{
  StatisticsReporterPtr statReporter = sharedResources->_statisticsReporter;

  std::ostringstream title;
  title << _appDescriptor->getClassName()
    << " instance " << _appDescriptor->getInstance()
    << " - " << pageTitle;
  XHTMLMaker::Node* body = maker.start(title.str());
  
  std::ostringstream stylesheetLink;
  stylesheetLink << "/" << _appDescriptor->getURN()
    << "/styles.css";
  XHTMLMaker::AttrMap stylesheetAttr;
  stylesheetAttr[ "rel" ] = "stylesheet";
  stylesheetAttr[ "type" ] = "text/css";
  stylesheetAttr[ "href" ] = stylesheetLink.str();
  maker.addNode("link", maker.getHead(), stylesheetAttr);
  
  XHTMLMaker::AttrMap tableAttr;
  tableAttr[ "border" ] = "0";
  tableAttr[ "cellspacing" ] = "7";
  tableAttr[ "width" ] = "100%";
  XHTMLMaker::Node* table = maker.addNode("table", body, tableAttr);
  
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  
  XHTMLMaker::AttrMap tableDivAttr;
  tableDivAttr[ "align" ] = "left";
  tableDivAttr[ "width" ] = "64";
  XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow, tableDivAttr);

  XHTMLMaker::AttrMap smLinkAttr;
  smLinkAttr[ "href" ] = _appDescriptor->getContextDescriptor()->getURL()
    + "/" + _appDescriptor->getURN();
  XHTMLMaker::Node* smLink = maker.addNode("a", tableDiv, smLinkAttr);
  
  XHTMLMaker::AttrMap smImgAttr;
  smImgAttr[ "align" ] = "middle";
  smImgAttr[ "src" ] = "/evf/images/smicon.jpg"; // $XDAQ_DOCUMENT_ROOT is prepended to this path
  smImgAttr[ "alt" ] = "main";
  smImgAttr[ "width" ] = "64";
  smImgAttr[ "height" ] = "64";
  smImgAttr[ "border" ] = "0";
  maker.addNode("img", smLink, smImgAttr);

  tableDiv = maker.addNode("td", tableRow);
  tableAttr[ "cellspacing" ] = "1";
  XHTMLMaker::Node* instanceTable = maker.addNode("table", tableDiv, tableAttr);
  XHTMLMaker::Node* instanceTableRow = maker.addNode("tr", instanceTable, _rowAttr);
  tableDivAttr[ "width" ] = "30%";
  XHTMLMaker::Node* instanceTableDiv = maker.addNode("td", instanceTableRow, tableDivAttr);
  XHTMLMaker::AttrMap fontAttr;
  fontAttr[ "size" ] = "+2";
  XHTMLMaker::Node* header = maker.addNode("font", instanceTableDiv, fontAttr);
  header = maker.addNode("b", header);
  maker.addText(header, _appDescriptor->getClassName());

  const std::string cvsVersion = "$Name:  $";
  if ( cvsVersion.length() > 9 ) {
    const std::string smVersion = "(" + cvsVersion.substr(7, cvsVersion.length()-9) + ")";
    maker.addText(instanceTableDiv, smVersion);
  }

  tableDivAttr[ "width" ] = "30%";
  instanceTableDiv = maker.addNode("td", instanceTableRow, tableDivAttr);
  header = maker.addNode("font", instanceTableDiv, fontAttr);
  header = maker.addNode("b", header);
  std::ostringstream instance;
  instance << "Instance " << _appDescriptor->getInstance();
  maker.addText(header, instance.str());
  
  tableDivAttr[ "width" ] = "40%";
  instanceTableDiv = maker.addNode("td", instanceTableRow, tableDivAttr);
  header = maker.addNode("font", instanceTableDiv, fontAttr);
  header = maker.addNode("b", header);
  maker.addText(header, 
    statReporter->getStateMachineMonitorCollection().externallyVisibleState());

  instanceTableRow = maker.addNode("tr", instanceTable, _rowAttr);

  instanceTableDiv = maker.addNode("td", instanceTableRow);
  fontAttr[ "size" ] = "-1";
  XHTMLMaker::Node* timestamp = maker.addNode("font", instanceTableDiv, fontAttr);
  maker.addText(timestamp,
    "Page last updated: " + utils::asctimeUTC(utils::getCurrentTime()) );

  instanceTableDiv = maker.addNode("td", instanceTableRow);
  XHTMLMaker::Node* hostname = maker.addNode("font", instanceTableDiv, fontAttr);
  maker.addText(hostname, "on " +
    sharedResources->_configuration->getDiskWritingParams()._hostName );
 
  instanceTableDiv = maker.addNode("td", instanceTableRow);
  XHTMLMaker::Node* innerState = maker.addNode("font", instanceTableDiv, fontAttr);
  maker.addText(innerState, 
    statReporter->getStateMachineMonitorCollection().innerStateName());

  tableDivAttr[ "align" ] = "right";
  tableDivAttr[ "width" ] = "64";
  tableDiv = maker.addNode("td", tableRow, tableDivAttr);
  
  XHTMLMaker::AttrMap xdaqLinkAttr;
  xdaqLinkAttr[ "href" ] = "/urn:xdaq-application:lid=3";
  XHTMLMaker::Node* xdaqLink = maker.addNode("a", tableDiv, xdaqLinkAttr);
  
  XHTMLMaker::AttrMap xdaqImgAttr;
  xdaqImgAttr[ "align" ] = "middle";
  xdaqImgAttr[ "src" ] = "/hyperdaq/images/HyperDAQ.jpg"; // $XDAQ_DOCUMENT_ROOT is prepended to this path
  xdaqImgAttr[ "alt" ] = "HyperDAQ";
  xdaqImgAttr[ "width" ] = "64";
  xdaqImgAttr[ "height" ] = "64";
  xdaqImgAttr[ "border" ] = "0";
  maker.addNode("img", xdaqLink, xdaqImgAttr);

  // Status message box (reason for failed state, etc.):
  std::string msg = "";
  if( statReporter->getStateMachineMonitorCollection().statusMessage( msg ) )
    {
      maker.addNode( "hr", body );
      XHTMLMaker::Node* msg_box = maker.addNode( "p", body );
      maker.addText( msg_box, msg );
    }

  maker.addNode( "hr", body );
  
  return body;

}


void WebPageHelper::addDOMforSMLinks
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *parent
)
{
  std::string url = _appDescriptor->getContextDescriptor()->getURL()
    + "/" + _appDescriptor->getURN();

  XHTMLMaker::AttrMap linkAttr;
  XHTMLMaker::Node *link;

  maker.addNode("hr", parent);

  linkAttr[ "href" ] = url;
  link = maker.addNode("a", parent, linkAttr);
  maker.addText(link, "Main web page");

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


void WebPageHelper::addDOMforResourceUsage
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *parent,
  ResourceMonitorCollection const& rmc,
  ThroughputMonitorCollection const& tmc
)
{
  ResourceMonitorCollection::Stats rmcStats;
  rmc.getStats(rmcStats);
  MonitoredQuantity::Stats poolUsageStats;
  tmc.getPoolUsageMQ().getStats(poolUsageStats);

  XHTMLMaker::AttrMap halfWidthAttr;
  halfWidthAttr[ "width" ] = "50%";
  
  XHTMLMaker::Node* table = maker.addNode("table", parent, _tableAttr);
  
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);

  XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow, halfWidthAttr);
  addTableForResourceUsages(maker, tableDiv, rmcStats, poolUsageStats);

  tableDiv = maker.addNode("td", tableRow, halfWidthAttr);
  addTableForDiskUsages(maker, tableDiv, rmcStats);
}


void WebPageHelper::addTableForResourceUsages
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *parent,
  ResourceMonitorCollection::Stats const& rmcStats,
  MonitoredQuantity::Stats const& poolUsageStats
)
{
  XHTMLMaker::AttrMap colspanAttr;
  colspanAttr[ "colspan" ] = "2";

  XHTMLMaker::Node* table = maker.addNode("table", parent, _tableAttr);
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
  maker.addText(tableDiv, "Resource Usage");
  
  addRowsForMemoryUsage(maker, table, poolUsageStats);
  addRowsForWorkers(maker, table, rmcStats);
  addRowsForSataBeast(maker, table, rmcStats);
}

   
void WebPageHelper::addRowsForMemoryUsage
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *table,
  MonitoredQuantity::Stats const& stats
)
{
  XHTMLMaker::AttrMap colspanAttr;
  colspanAttr[ "colspan" ] = "2";

  XHTMLMaker::AttrMap tableLabelAttr = _tableLabelAttr;
  tableLabelAttr[ "width" ] = "54%";

  XHTMLMaker::AttrMap tableValueAttr = _tableValueAttr;
  tableValueAttr[ "width" ] = "46%";

  // Memory pool usage
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv;
  if ( stats.getSampleCount() > 0 )
  {
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr);
    maker.addText(tableDiv, "Memory pool used (bytes)");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addDouble( tableDiv, stats.getLastSampleValue(), 0 );
  }
  else
  {
    tableDiv = maker.addNode("td", tableRow, colspanAttr);
    maker.addText(tableDiv, "Memory pool pointer not yet available");
  }
}

 
void WebPageHelper::addRowsForWorkers
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *table,
  ResourceMonitorCollection::Stats const& stats
)
{
  XHTMLMaker::AttrMap tableLabelAttr = _tableLabelAttr;
  tableLabelAttr[ "width" ] = "54%";

  XHTMLMaker::AttrMap tableValueAttr = _tableValueAttr;
  tableValueAttr[ "width" ] = "46%";

  // # copy worker
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# CopyWorker");
  tableDiv = maker.addNode("td", tableRow, tableValueAttr);
  maker.addInt( tableDiv, stats.numberOfCopyWorkers );

  // # inject worker
  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# InjectWorker");
  tableDiv = maker.addNode("td", tableRow, tableValueAttr);
  maker.addInt( tableDiv, stats.numberOfInjectWorkers );
}


void WebPageHelper::addRowsForSataBeast
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *table,
  ResourceMonitorCollection::Stats const& stats
)
{
  XHTMLMaker::AttrMap tableLabelAttr = _tableLabelAttr;
  tableLabelAttr[ "width" ] = "54%";

  XHTMLMaker::AttrMap tableValueAttr = _tableValueAttr;
  tableValueAttr[ "width" ] = "46%";

  XHTMLMaker::Node *tableRow, *tableDiv;

  XHTMLMaker::AttrMap warningAttr = _rowAttr;

  if (stats.sataBeastStatus < 0 )
  {
    warningAttr[ "bgcolor" ] = _alarmColors[ AlarmHandler::WARNING ];

    XHTMLMaker::AttrMap colspanAttr = _tableLabelAttr;
    colspanAttr[ "colspan" ] = "2";

    tableRow = maker.addNode("tr", table, warningAttr);
    tableDiv = maker.addNode("td", tableRow, colspanAttr);
    maker.addText(tableDiv, "No SATA disks found");
  }
  else
  {
    if ( stats.sataBeastStatus > 0 )
      warningAttr[ "bgcolor" ] = _alarmColors[ AlarmHandler::ERROR ];
    tableRow = maker.addNode("tr", table, warningAttr);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr);
    maker.addText(tableDiv, "SATA beast status");
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    maker.addInt( tableDiv, stats.sataBeastStatus );
  }
}


void WebPageHelper::addTableForDiskUsages
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *parent,
  ResourceMonitorCollection::Stats const& stats
)
{
  XHTMLMaker::AttrMap colspanAttr;
  colspanAttr[ "colspan" ] = "2";
  
  XHTMLMaker::AttrMap tableLabelAttr = _tableLabelAttr;
  tableLabelAttr[ "width" ] = "54%";

  XHTMLMaker::AttrMap tableValueAttr = _tableValueAttr;
  tableValueAttr[ "width" ] = "46%";
  
  XHTMLMaker::AttrMap warningAttr = _rowAttr;

  XHTMLMaker::Node* table = maker.addNode("table", parent, _tableAttr);
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
  maker.addText(tableDiv, "Disk Space Usage");


  for (ResourceMonitorCollection::DiskUsageStatsPtrList::const_iterator
         it = stats.diskUsageStatsList.begin(),
         itEnd = stats.diskUsageStatsList.end();
       it != itEnd;
       ++it)
  {
    warningAttr[ "bgcolor" ] = _alarmColors[ (*it)->alarmState ];
    tableRow = maker.addNode("tr", table, warningAttr);
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr);
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


void WebPageHelper::addDOMforFragmentMonitor
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *parent,
  FragmentMonitorCollection const& fmc
)
{
  FragmentMonitorCollection::FragmentStats stats;
  fmc.getStats(stats);

  XHTMLMaker::AttrMap colspanAttr;
  colspanAttr[ "colspan" ] = "4";

  XHTMLMaker::Node* table = maker.addNode("table", parent, _tableAttr);

  // Received Data Statistics header
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
  maker.addText(tableDiv, "Received I2O Frames");

  // Parameter/Value header
  tableRow = maker.addNode("tr", table, _rowAttr);
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

void WebPageHelper::addFragmentStats
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *table,
  FragmentMonitorCollection::FragmentStats const& stats,
  const MonitoredQuantity::DataSetType dataSet
)
{
  // Mean performance header
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
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


void WebPageHelper::addDurationToTableHead
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *tableRow,
  const utils::duration_t duration
)
{
  XHTMLMaker::AttrMap tableValueWidth;
  tableValueWidth[ "width" ] = "23%";

  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, tableValueWidth);
  std::ostringstream tmpString;
  tmpString << std::fixed << std::setprecision(0) <<
      duration << " s";
  maker.addText(tableDiv, tmpString.str());
}


void WebPageHelper::addRowForFramesReceived
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *table,
  FragmentMonitorCollection::FragmentStats const& stats,
  const MonitoredQuantity::DataSetType dataSet
)
{
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
  maker.addText(tableDiv, "Frames Received");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, stats.allFragmentSizeStats.getSampleCount(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, stats.eventFragmentSizeStats.getSampleCount(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, stats.dqmEventFragmentSizeStats.getSampleCount(dataSet) );
}


void WebPageHelper::addRowForBandwidth
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *table,
  FragmentMonitorCollection::FragmentStats const& stats,
  const MonitoredQuantity::DataSetType dataSet
)
{
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
  maker.addText(tableDiv, "Bandwidth (MB/s)");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.allFragmentSizeStats.getValueRate(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.eventFragmentSizeStats.getValueRate(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.dqmEventFragmentSizeStats.getValueRate(dataSet) );
}


void WebPageHelper::addRowForRate
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *table,
  FragmentMonitorCollection::FragmentStats const& stats,
  const MonitoredQuantity::DataSetType dataSet
)
{
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
  maker.addText(tableDiv, "Rate (frames/s)");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.allFragmentSizeStats.getSampleRate(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.eventFragmentSizeStats.getSampleRate(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.dqmEventFragmentSizeStats.getSampleRate(dataSet) );
}


void WebPageHelper::addRowForLatency
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *table,
  FragmentMonitorCollection::FragmentStats const& stats,
  const MonitoredQuantity::DataSetType dataSet
)
{
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
  maker.addText(tableDiv, "Latency (us/frame)");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.allFragmentSizeStats.getSampleLatency(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.eventFragmentSizeStats.getSampleLatency(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.dqmEventFragmentSizeStats.getSampleLatency(dataSet) );
}


void WebPageHelper::addRowForTotalVolume
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *table,
  FragmentMonitorCollection::FragmentStats const& stats,
  const MonitoredQuantity::DataSetType dataSet
)
{
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
  maker.addText(tableDiv, "Total volume received (MB)");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.allFragmentSizeStats.getValueSum(dataSet), 3 );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.eventFragmentSizeStats.getValueSum(dataSet), 3 );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.dqmEventFragmentSizeStats.getValueSum(dataSet), 3 );
}


void WebPageHelper::addRowForMaxBandwidth
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *table,
  FragmentMonitorCollection::FragmentStats const& stats,
  const MonitoredQuantity::DataSetType dataSet
)
{
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
  maker.addText(tableDiv, "Maximum Bandwidth (MB/s)");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.allFragmentBandwidthStats.getValueMax(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.eventFragmentBandwidthStats.getValueMax(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.dqmEventFragmentBandwidthStats.getValueMax(dataSet) );
}


void WebPageHelper::addRowForMinBandwidth
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *table,
  FragmentMonitorCollection::FragmentStats const& stats,
  const MonitoredQuantity::DataSetType dataSet
)
{
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
  maker.addText(tableDiv, "Minimum Bandwidth (MB/s)");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.allFragmentBandwidthStats.getValueMin(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.eventFragmentBandwidthStats.getValueMin(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.dqmEventFragmentBandwidthStats.getValueMin(dataSet) );
}


void WebPageHelper::addDOMforRunMonitor
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *parent,
  RunMonitorCollection const& rmc
)
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
  
  XHTMLMaker::AttrMap tableLabelAttr = _tableLabelAttr;
  tableLabelAttr[ "width" ] = "18%";

  XHTMLMaker::AttrMap tableValueAttr = _tableValueAttr;
  tableValueAttr[ "width" ] = "16%";

  XHTMLMaker::Node* table = maker.addNode("table", parent, _tableAttr);

  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
  maker.addText(tableDiv, "Storage Manager Statistics");

  // Run number and lumi section
  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Run number");
  tableDiv = maker.addNode("td", tableRow, tableValueAttr);
  maker.addDouble( tableDiv, runNumbersSeenStats.getLastSampleValue(), 0 );
  tableDiv = maker.addNode("td", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Current lumi section");
  tableDiv = maker.addNode("td", tableRow, tableValueAttr);
  maker.addDouble( tableDiv, lumiSectionsSeenStats.getLastSampleValue(), 0 );
  tableDiv = maker.addNode("td", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Last EoLS");
  tableDiv = maker.addNode("td", tableRow, tableValueAttr);
  maker.addDouble( tableDiv, eolsSeenStats.getLastSampleValue(), 0 );

  // Total events received
  tableRow = maker.addNode("tr", table, _specialRowAttr);
  tableDiv = maker.addNode("td", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Events received (non-unique)");
  tableDiv = maker.addNode("td", tableRow, tableValueAttr);
  maker.addInt( tableDiv, eventIDsReceivedStats.getSampleCount() );
  tableDiv = maker.addNode("td", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Error events received");
  tableDiv = maker.addNode("td", tableRow, tableValueAttr);
  maker.addInt( tableDiv, errorEventIDsReceivedStats.getSampleCount() );
  tableDiv = maker.addNode("td", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Unwanted events received");
  tableDiv = maker.addNode("td", tableRow, tableValueAttr);
  maker.addInt( tableDiv, unwantedEventIDsReceivedStats.getSampleCount() );

  // Last event IDs
  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Last event ID");
  tableDiv = maker.addNode("td", tableRow, tableValueAttr);
  maker.addDouble( tableDiv, eventIDsReceivedStats.getLastSampleValue(), 0 );
  tableDiv = maker.addNode("td", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Last error event ID");
  tableDiv = maker.addNode("td", tableRow, tableValueAttr);
  maker.addDouble( tableDiv, errorEventIDsReceivedStats.getLastSampleValue(), 0 );
  tableDiv = maker.addNode("td", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Last unwanted event ID");
  tableDiv = maker.addNode("td", tableRow, tableValueAttr);
  maker.addDouble( tableDiv, unwantedEventIDsReceivedStats.getLastSampleValue(), 0 );

}


void WebPageHelper::addDOMforStoredData
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *parent,
  StreamsMonitorCollection const& smc
)
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

  XHTMLMaker::Node* table = maker.addNode("table", parent, _tableAttr);

  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
  maker.addText(tableDiv, "Stored Data Statistics");

  // Header
  tableRow = maker.addNode("tr", table, _specialRowAttr);
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

  tableRow = maker.addNode("tr", table, _specialRowAttr);
  tableDiv = maker.addNode("th", tableRow, tableValueWidthAttr);
  maker.addText(tableDiv, "average");
  tableDiv = maker.addNode("th", tableRow, tableValueWidthAttr);
  maker.addText(tableDiv, "min");
  tableDiv = maker.addNode("th", tableRow, tableValueWidthAttr);
  maker.addText(tableDiv, "max");
  
  if (smc.getStreamRecordsMQ().empty())
  {
    tableRow = maker.addNode("tr", table, _rowAttr);
    tableDiv = maker.addNode("td", tableRow, colspanAttr);
    maker.addText(tableDiv, "no streams available yet");
    return;
  }
  // Mean performance
  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("th", tableRow, colspanAttr);
  {
    std::ostringstream tmpString;
    tmpString << "Mean performance for " << std::fixed << std::setprecision(0) <<
      allStreamsVolumeStats.getDuration() << " s";
    maker.addText(tableDiv, tmpString.str());
  }
  listStreamRecordsStats(maker, table, smc, MonitoredQuantity::FULL);
  
  
  // Recent performance
  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("th", tableRow, colspanAttr);
  {
    std::ostringstream tmpString;
    tmpString << "Recent performance for the last " << std::fixed << std::setprecision(0) <<
      allStreamsVolumeStats.getDuration(MonitoredQuantity::RECENT) << " s";
    maker.addText(tableDiv, tmpString.str());
  }
  listStreamRecordsStats(maker, table, smc, MonitoredQuantity::RECENT);
}


void WebPageHelper::addDOMforConfigString
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *parent,
  DiskWritingParams const& dwParams
)
{
  XHTMLMaker::Node* table = maker.addNode("table", parent);
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _specialRowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "SM Configuration");
  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow);
  XHTMLMaker::AttrMap textareaAttr;
  textareaAttr[ "rows" ] = "10";
  textareaAttr[ "cols" ] = "100";
  textareaAttr[ "scroll" ] = "yes";
  textareaAttr[ "readonly" ];
  textareaAttr[ "title" ] = "SM config";
  XHTMLMaker::Node* textarea = maker.addNode("textarea", tableDiv, textareaAttr);
  maker.addText(textarea, dwParams._streamConfiguration);
}  


void WebPageHelper::listStreamRecordsStats
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *table,
  StreamsMonitorCollection const& smc,
  const MonitoredQuantity::DataSetType dataSet
)
{
  StreamsMonitorCollection::StreamRecordList const& streamRecords =
    smc.getStreamRecordsMQ();
  MonitoredQuantity::Stats allStreamsFileCountStats;
  smc.getAllStreamsFileCountMQ().getStats(allStreamsFileCountStats);
  MonitoredQuantity::Stats allStreamsVolumeStats;
  smc.getAllStreamsVolumeMQ().getStats(allStreamsVolumeStats);
  MonitoredQuantity::Stats allStreamsBandwidthStats;
  smc.getAllStreamsBandwidthMQ().getStats(allStreamsBandwidthStats);
 
  XHTMLMaker::Node *tableRow, *tableDiv;

  XHTMLMaker::AttrMap tableValueAttr = _tableValueAttr;
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
    
    
    tableRow = maker.addNode("tr", table, _rowAttr);
    tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, (*it)->streamName);
    tableDiv = maker.addNode("td", tableRow);
    maker.addDouble(tableDiv, (*it)->fractionToDisk, 2);
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addInt( tableDiv, streamFileCountStats.getSampleCount(dataSet) );
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addInt( tableDiv, streamVolumeStats.getSampleCount(dataSet) );
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble( tableDiv, streamVolumeStats.getSampleRate(dataSet), 1 );
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble( tableDiv, streamVolumeStats.getValueSum(dataSet), 1 );
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble( tableDiv, streamBandwidthStats.getValueRate(dataSet) );
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble( tableDiv, streamBandwidthStats.getValueMin(dataSet) );
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble( tableDiv, streamBandwidthStats.getValueMax(dataSet) );
  }
  
  tableRow = maker.addNode("tr", table, _specialRowAttr);
  tableDiv = maker.addNode("td", tableRow);
  maker.addText(tableDiv, "Total");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, allStreamsFileCountStats.getSampleCount(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, allStreamsVolumeStats.getSampleCount(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, allStreamsVolumeStats.getSampleRate(dataSet), 1 );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, allStreamsVolumeStats.getValueSum(dataSet), 1 );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, allStreamsBandwidthStats.getValueRate(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, allStreamsBandwidthStats.getValueMin(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, allStreamsBandwidthStats.getValueMax(dataSet) );
 
}


void WebPageHelper::addDOMforFiles(XHTMLMaker& maker,
                                   XHTMLMaker::Node *parent,
                                   FilesMonitorCollection const& fmc)
{
  FilesMonitorCollection::FileRecordList fileRecords;
  fmc.getFileRecords(fileRecords);

  XHTMLMaker::AttrMap colspanAttr;
  colspanAttr[ "colspan" ] = "5";
  
  XHTMLMaker::AttrMap tableLabelAttr = _tableLabelAttr;
  tableLabelAttr[ "align" ] = "center";

  XHTMLMaker::AttrMap tableValueWidthAttr;
  tableValueWidthAttr[ "width" ] = "11%";

  XHTMLMaker::AttrMap tableCounterWidthAttr;
  tableCounterWidthAttr[ "width" ] = "5%";

  XHTMLMaker::Node* table = maker.addNode("table", parent, _tableAttr);

  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
  maker.addText(tableDiv, "File Statistics (most recent first)");

  // Header
  tableRow = maker.addNode("tr", table, _specialRowAttr);
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

  // File list
  if (fileRecords.empty())
  {
    tableRow = maker.addNode("tr", table, _rowAttr);
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
    tableRow = maker.addNode("tr", table, _rowAttr);
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addInt( tableDiv, (*it)->entryCounter );
    tableDiv = maker.addNode("td", tableRow);
    maker.addText(tableDiv, (*it)->completeFileName());
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addInt( tableDiv, (*it)->eventCount );
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addInt( tableDiv, (*it)->fileSize );
    tableDiv = maker.addNode("td", tableRow, tableLabelAttr);
    maker.addText(tableDiv, (*it)->closingReason());
  }
}


void WebPageHelper::addDOMforEventConsumers
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *parent,
  const SharedResourcesPtr sharedResources
)
{
  RegistrationCollection::ConsumerRegistrations consumers;
  sharedResources->_registrationCollection->getEventConsumers(consumers);
  
  EventConsumerMonitorCollection& eventConsumerCollection =
    sharedResources->_statisticsReporter->getEventConsumerMonitorCollection();
  
  
  XHTMLMaker::Node* table = maker.addNode("table", parent, _tableAttr);
  
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);

  XHTMLMaker::AttrMap colspanAttr;
  colspanAttr[ "colspan" ] = "20";
  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
  maker.addText(tableDiv, "Consumer Statistics");

  XHTMLMaker::AttrMap rowspanAttr;
  rowspanAttr[ "rowspan" ] = "2";

  XHTMLMaker::AttrMap subColspanAttr;
  subColspanAttr[ "colspan" ] = "2";

  //Header
  tableRow = maker.addNode("tr", table, _specialRowAttr);
  tableDiv = maker.addNode("th", tableRow, rowspanAttr);
  maker.addText(tableDiv, "ID");
  tableDiv = maker.addNode("th", tableRow, rowspanAttr);
  maker.addText(tableDiv, "Name");
  tableDiv = maker.addNode("th", tableRow, rowspanAttr);
  maker.addText(tableDiv, "Consumer Host" );
  tableDiv = maker.addNode("th", tableRow, rowspanAttr);
  maker.addText(tableDiv, "Status");
  tableDiv = maker.addNode("th", tableRow, rowspanAttr);
  maker.addText(tableDiv, "HLT Output Module");
  tableDiv = maker.addNode("th", tableRow, rowspanAttr);
  maker.addText(tableDiv, "Filters");
  tableDiv = maker.addNode("th", tableRow, rowspanAttr);
  maker.addText(tableDiv, "Enquing Policy");
  tableDiv = maker.addNode("th", tableRow, rowspanAttr);
  maker.addText(tableDiv, "Queue Size");
  tableDiv = maker.addNode("th", tableRow, rowspanAttr);
  maker.addText(tableDiv, "Events In Queue");

  tableDiv = maker.addNode("th", tableRow, subColspanAttr);
  maker.addText(tableDiv, "Enqueued Event Rate (Hz)");
  tableDiv = maker.addNode("th", tableRow, subColspanAttr);
  maker.addText(tableDiv, "Served Event Rate (Hz)");
  tableDiv = maker.addNode("th", tableRow, subColspanAttr);
  maker.addText(tableDiv, "Events Served");
  tableDiv = maker.addNode("th", tableRow, subColspanAttr);
  maker.addText(tableDiv, "Average Event Size (kB)");
  tableDiv = maker.addNode("th", tableRow, subColspanAttr);
  maker.addText(tableDiv, "Bandwidth (kB/s)");

  tableRow = maker.addNode("tr", table, _specialRowAttr);
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "overall");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "recent");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "overall");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "recent");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "overall");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "recent");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "overall");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "recent");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "overall");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "recent");

  if ( consumers.empty() )
  {
    XHTMLMaker::AttrMap messageAttr = colspanAttr;
    messageAttr[ "align" ] = "center";

    tableRow = maker.addNode("tr", table, _rowAttr);
    tableDiv = maker.addNode("td", tableRow, messageAttr);
    maker.addText(tableDiv, "No event consumers have registered yet.");
    return;
  }
  
  // Loop over consumers
  bool evenRow = false;
  
  for( RegistrationCollection::ConsumerRegistrations::const_iterator
         it = consumers.begin(), itEnd = consumers.end();
       it != itEnd; ++it )
  {
    // Row:
    XHTMLMaker::AttrMap rowAttr = _rowAttr;
    if( evenRow )
    {
      rowAttr[ "style" ] = "background-color:#e0e0e0;";
      evenRow = false;
    }
    else
    {
      evenRow = true;
    }
    tableRow = maker.addNode("tr", table, rowAttr);

    // ID:
    std::ostringstream cid;
    cid << (*it)->consumerID();
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addText(tableDiv, cid.str());
    
    // Name:
    tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
    if ( (*it)->isProxyServer() )
      maker.addText(tableDiv, "Proxy Server");
    else
      maker.addText(tableDiv, (*it)->consumerName());

    // Host:
    tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
    maker.addText(tableDiv, (*it)->remoteHost());
    
    // Status:
    XHTMLMaker::AttrMap statusAttr = _tableLabelAttr;
    std::string statusMessage;
    if( (*it)->isStale() )
    {
      statusAttr[ "style" ] = "color:brown;";
      statusMessage = "Stale";
    }
    else
    {
      statusAttr[ "style" ] = "color:green;";
      statusMessage = "Active";
    }
    tableDiv = maker.addNode("td", tableRow, statusAttr);
    maker.addText(tableDiv, statusMessage);
    
    // HLT output module:
    tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
    maker.addText(tableDiv, (*it)->outputModuleLabel());
    
    // Filter list:
    std::string filters = (*it)->triggerSelection();
    if ( filters.empty() )
    {
      const EventConsumerRegistrationInfo::FilterList fl = (*it)->selEvents();
      for( EventConsumerRegistrationInfo::FilterList::const_iterator
             lit = fl.begin(), litEnd = fl.end();
           lit != litEnd; ++lit )
      {
        if( lit != fl.begin() ) filters += "&nbsp;&nbsp;";
        filters += *lit;
      }
    }
    tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
    maker.addText(tableDiv, filters);
    
    // Policy:
    std::ostringstream policy;
    policy << (*it)->queuePolicy();
    tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
    maker.addText(tableDiv, policy.str());
    
    // Queue size:
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addInt(tableDiv, (*it)->queueSize());
    
    // Events in queue:
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addInt(tableDiv, sharedResources->_eventConsumerQueueCollection->size( (*it)->queueId() ));
    
    // Events enqueued:
    MonitoredQuantity::Stats enqueuedStats;
    const bool enqueuedFound = eventConsumerCollection.getQueued( (*it)->queueId(), enqueuedStats );
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble(tableDiv,
      enqueuedFound ? enqueuedStats.getSampleRate(MonitoredQuantity::FULL) : 0
    );
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble(tableDiv,
      enqueuedFound ? enqueuedStats.getSampleRate(MonitoredQuantity::RECENT) : 0
    );

    // Events served:
    MonitoredQuantity::Stats servedStats;
    const bool servedFound = eventConsumerCollection.getServed( (*it)->queueId(), servedStats );

    // rate
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble(tableDiv,
      servedFound ? servedStats.getSampleRate(MonitoredQuantity::FULL) : 0
    );
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble(tableDiv,
      servedFound ? servedStats.getSampleRate(MonitoredQuantity::RECENT) : 0
    );
      
    // event counts
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addInt(tableDiv,
      servedFound ? servedStats.getSampleCount(MonitoredQuantity::FULL) : 0
    );
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addInt(tableDiv,
      servedFound ? servedStats.getSampleCount(MonitoredQuantity::RECENT) : 0
    );
      
    // event size
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble(tableDiv,
      servedFound ? servedStats.getValueAverage(MonitoredQuantity::FULL)/1024 : 0,
      1);
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble(tableDiv,
      servedFound ? servedStats.getValueAverage(MonitoredQuantity::RECENT)/1024 : 0,
      1);

    // bandwidth
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble(tableDiv,
      servedFound ? servedStats.getValueRate(MonitoredQuantity::FULL)/1024 : 0
    );
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble(tableDiv,
      servedFound ? servedStats.getValueRate(MonitoredQuantity::RECENT)/1024 : 0
    );
  }
}


void WebPageHelper::addDOMforDQMEventConsumers
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *parent,
  const SharedResourcesPtr sharedResources
)
{
  RegistrationCollection::DQMConsumerRegistrations consumers;
  sharedResources->_registrationCollection->getDQMEventConsumers( consumers );
  
  DQMConsumerMonitorCollection& dqmConsumerCollection =
    sharedResources->_statisticsReporter->getDQMConsumerMonitorCollection();
  
  
  XHTMLMaker::Node* table = maker.addNode("table", parent, _tableAttr);
  
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);

  XHTMLMaker::AttrMap colspanAttr;
  colspanAttr[ "colspan" ] = "20";
  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
  maker.addText(tableDiv, "DQM Consumer Statistics");

  XHTMLMaker::AttrMap rowspanAttr;
  rowspanAttr[ "rowspan" ] = "2";

  XHTMLMaker::AttrMap subColspanAttr;
  subColspanAttr[ "colspan" ] = "2";

  //Header
  tableRow = maker.addNode("tr", table, _specialRowAttr);
  tableDiv = maker.addNode("th", tableRow, rowspanAttr);
  maker.addText(tableDiv, "ID");
  tableDiv = maker.addNode("th", tableRow, rowspanAttr);
  maker.addText(tableDiv, "Name");
  tableDiv = maker.addNode("th", tableRow, rowspanAttr);
  maker.addText(tableDiv, "Consumer Host" );
  tableDiv = maker.addNode("th", tableRow, rowspanAttr);
  maker.addText(tableDiv, "Status");
  tableDiv = maker.addNode("th", tableRow, rowspanAttr);
  maker.addText(tableDiv, "Top Level Folder");
  tableDiv = maker.addNode("th", tableRow, rowspanAttr);
  maker.addText(tableDiv, "Enquing Policy");
  tableDiv = maker.addNode("th", tableRow, rowspanAttr);
  maker.addText(tableDiv, "Queue Size");
  tableDiv = maker.addNode("th", tableRow, rowspanAttr);
  maker.addText(tableDiv, "Events In Queue");

  tableDiv = maker.addNode("th", tableRow, subColspanAttr);
  maker.addText(tableDiv, "Enqueued Event Rate (Hz)");
  tableDiv = maker.addNode("th", tableRow, subColspanAttr);
  maker.addText(tableDiv, "Served Event Rate (Hz)");
  tableDiv = maker.addNode("th", tableRow, subColspanAttr);
  maker.addText(tableDiv, "Events Served");
  tableDiv = maker.addNode("th", tableRow, subColspanAttr);
  maker.addText(tableDiv, "Average Event Size (kB)");
  tableDiv = maker.addNode("th", tableRow, subColspanAttr);
  maker.addText(tableDiv, "Bandwidth (kB/s)");

  tableRow = maker.addNode("tr", table, _specialRowAttr);
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "overall");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "recent");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "overall");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "recent");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "overall");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "recent");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "overall");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "recent");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "overall");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "recent");

  if ( consumers.empty() )
  {
    XHTMLMaker::AttrMap messageAttr = colspanAttr;
    messageAttr[ "align" ] = "center";

    tableRow = maker.addNode("tr", table, _rowAttr);
    tableDiv = maker.addNode("td", tableRow, messageAttr);
    maker.addText(tableDiv, "No DQM consumers have registered yet.");
    return;
  }

  // Loop over consumers
  bool evenRow = false;

  for( RegistrationCollection::DQMConsumerRegistrations::const_iterator
         it = consumers.begin(), itEnd = consumers.end();
       it != itEnd; ++it )
  {
    // Row:
    XHTMLMaker::AttrMap rowAttr = _rowAttr;
    if( evenRow )
    {
      rowAttr[ "style" ] = "background-color:#e0e0e0;";
      evenRow = false;
    }
    else
    {
      evenRow = true;
    }
    tableRow = maker.addNode("tr", table, rowAttr);

    // ID:
    std::ostringstream cid;
    cid << (*it)->consumerID();
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addText(tableDiv, cid.str());
    
    // Name:
    tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
    if ( (*it)->isProxyServer() )
      maker.addText(tableDiv, "Proxy Server");
    else
      maker.addText(tableDiv, (*it)->consumerName());
    
    // Host:
    tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
    maker.addText(tableDiv, (*it)->remoteHost());
    
    // Status:
    XHTMLMaker::AttrMap statusAttr = _tableLabelAttr;
    std::string statusMessage;
    if( (*it)->isStale() )
    {
      statusAttr[ "style" ] = "color:brown;";
      statusMessage = "Stale";
    }
    else
    {
      statusAttr[ "style" ] = "color:green;";
      statusMessage = "Active";
    }
    tableDiv = maker.addNode("td", tableRow, statusAttr);
    maker.addText(tableDiv, statusMessage);
    
    // Top level folder:
    tableDiv = maker.addNode("td", tableRow, statusAttr);
    maker.addText(tableDiv, (*it)->topLevelFolderName());
    
    // Policy:
    std::ostringstream policy;
    policy << (*it)->queuePolicy();
    tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
    maker.addText(tableDiv, policy.str());
    
    // Queue size:
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addInt(tableDiv, (*it)->queueSize());
    
    // Events in queue:
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addInt(tableDiv, sharedResources->_dqmEventConsumerQueueCollection->size( (*it)->queueId() ));

    // Events enqueued:
    MonitoredQuantity::Stats enqueuedStats;
    const bool enqueuedFound = dqmConsumerCollection.getQueued( (*it)->queueId(), enqueuedStats );
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble(tableDiv,
      enqueuedFound ? enqueuedStats.getSampleRate(MonitoredQuantity::FULL) : 0
    );
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble(tableDiv,
      enqueuedFound ? enqueuedStats.getSampleRate(MonitoredQuantity::RECENT) : 0
    );

    // Events served:
    MonitoredQuantity::Stats servedStats;
    const bool servedFound = dqmConsumerCollection.getServed( (*it)->queueId(), servedStats );

    // rate
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble(tableDiv,
      servedFound ? servedStats.getSampleRate(MonitoredQuantity::FULL) : 0
    );
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble(tableDiv,
      servedFound ? servedStats.getSampleRate(MonitoredQuantity::RECENT) : 0
    );
      
    // event counts
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addInt(tableDiv,
      servedFound ? servedStats.getSampleCount(MonitoredQuantity::FULL) : 0
    );
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addInt(tableDiv,
      servedFound ? servedStats.getSampleCount(MonitoredQuantity::RECENT) : 0
    );
      
    // event size
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble(tableDiv,
      servedFound ? servedStats.getValueAverage(MonitoredQuantity::FULL)/1024 : 0,
      1);
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble(tableDiv,
      servedFound ? servedStats.getValueAverage(MonitoredQuantity::RECENT)/1024 : 0,
      1);

    // bandwidth
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble(tableDiv,
      servedFound ? servedStats.getValueRate(MonitoredQuantity::FULL)/1024 : 0
    );
    tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
    maker.addDouble(tableDiv,
      servedFound ? servedStats.getValueRate(MonitoredQuantity::RECENT)/1024 : 0
    );
  }
}


void WebPageHelper::addDOMforProcessedDQMEvents(XHTMLMaker& maker,
                                                XHTMLMaker::Node *parent,
                                                DQMEventMonitorCollection const& dmc)
{
  DQMEventMonitorCollection::DQMEventStats stats;
  dmc.getStats(stats);

  XHTMLMaker::AttrMap colspanAttr;
  colspanAttr[ "colspan" ] = "4";

  XHTMLMaker::Node* table = maker.addNode("table", parent, _tableAttr);

  // Received Data Statistics header
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
  maker.addText(tableDiv, "Processed DQM events");

  // Parameter/Value header
  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "Parameter");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "Received");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "Served to consumers");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "Written to disk");

  addDQMEventStats(maker, table, stats,  MonitoredQuantity::FULL);

  addDQMEventStats(maker, table, stats,  MonitoredQuantity::RECENT);
}


void WebPageHelper::addDOMforDQMEventStatistics(XHTMLMaker& maker,
                                                XHTMLMaker::Node *parent,
                                                DQMEventMonitorCollection const& dmc)
{
  DQMEventMonitorCollection::DQMEventStats stats;
  dmc.getStats(stats);

  XHTMLMaker::AttrMap colspanAttr;
  colspanAttr[ "colspan" ] = "3";

  XHTMLMaker::Node* table = maker.addNode("table", parent, _tableAttr);

  // Received Data Statistics header
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
  maker.addText(tableDiv, "DQM Event Statistics");

  // Parameter/Value header
  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "Parameter");
  {
    tableDiv = maker.addNode("th", tableRow);
    std::ostringstream tmpString;
    tmpString << "Full run (" <<
      std::fixed << std::setprecision(0) <<
      stats.dqmEventSizeStats.getDuration(MonitoredQuantity::FULL) <<
      " s)";
    maker.addText(tableDiv, tmpString.str());
  }
  {
    tableDiv = maker.addNode("th", tableRow);
    std::ostringstream tmpString;
    tmpString << "Recent (" <<
      std::fixed << std::setprecision(0) <<
      stats.dqmEventSizeStats.getDuration(MonitoredQuantity::RECENT) <<
      " s)";
    maker.addText(tableDiv, tmpString.str());
  }


  // DQM events processed
  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "DQM events processed");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, stats.dqmEventSizeStats.getSampleCount(MonitoredQuantity::FULL) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, stats.dqmEventSizeStats.getSampleCount(MonitoredQuantity::RECENT) );

  // DQM events lost
  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "DQM events discarded");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.discardedDQMEventCountsStats.getValueSum(MonitoredQuantity::FULL), 0 );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.discardedDQMEventCountsStats.getValueSum(MonitoredQuantity::RECENT), 0 );

  // Average updates/folder
  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "Updates/folder (average)");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.numberOfUpdatesStats.getValueAverage(MonitoredQuantity::FULL) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.numberOfUpdatesStats.getValueAverage(MonitoredQuantity::RECENT) );

  // Min updates/folder
  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "Updates/folder (min)");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.numberOfUpdatesStats.getValueMin(MonitoredQuantity::FULL) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.numberOfUpdatesStats.getValueMin(MonitoredQuantity::RECENT) );

  // Max updates/folder
  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "Updates/folder (max)");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.numberOfUpdatesStats.getValueMax(MonitoredQuantity::FULL) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.numberOfUpdatesStats.getValueMax(MonitoredQuantity::RECENT) );

  // RMS updates/folder
  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "Updates/folder (RMS)");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.numberOfUpdatesStats.getValueRMS(MonitoredQuantity::FULL) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.numberOfUpdatesStats.getValueRMS(MonitoredQuantity::RECENT) );
}


void WebPageHelper::addDOMforThroughputStatistics(XHTMLMaker& maker,
                                                  XHTMLMaker::Node *parent,
                                                  ThroughputMonitorCollection const& tmc)
{
  XHTMLMaker::AttrMap colspanAttr;
  colspanAttr[ "colspan" ] = "21";

  XHTMLMaker::AttrMap tableLabelAttr = _tableLabelAttr;
  tableLabelAttr[ "align" ] = "center";

  XHTMLMaker::Node* table = maker.addNode("table", parent, _tableAttr);

  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
  maker.addText(tableDiv, "Throughput Statistics");

  // Header
  tableRow = maker.addNode("tr", table, _specialRowAttr);
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "Time (UTC)");
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "Memory pool usage (bytes)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Instantaneous Number of Fragments in Fragment Queue");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Memory used in Fragment Queue (MB)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Number of Fragments Popped from Fragment Queue (Hz)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Data Rate Popped from Fragment Queue (MB/sec)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Fragment Processor Thread Busy Percentage");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Instantaneous Number of Events in Fragment Store");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Memory used in Fragment Store (MB)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Instantaneous Number of Events in Stream Queue");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Memory used in Stream Queue (MB)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Number of Events Popped from Stream Queue (Hz)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Data Rate Popped from Stream Queue (MB/sec)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Disk Writer Thread Busy Percentage");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Number of Events Written to Disk (Hz)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Data  Rate to Disk (MB/sec)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Instantaneous Number of DQMEvents in DQMEvent Queue");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Memory used in DQMEvent Queue (MB)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Number of DQMEvents Popped from DQMEvent Queue (Hz)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Data Rate Popped from DQMEvent Queue (MB/sec)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
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


void WebPageHelper::addRowForThroughputStatistics
(
  XHTMLMaker& maker,
  XHTMLMaker::Node* table,
  const ThroughputMonitorCollection::Stats::Snapshot& snapshot,
  const bool isAverage
)
{
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv;
  XHTMLMaker::AttrMap tableValueAttr = _tableValueAttr;

  if (isAverage)
  {
    tableValueAttr[ "style" ] = "background-color: yellow;";
    tableDiv = maker.addNode("td", tableRow, tableValueAttr);
    std::ostringstream avg;
    avg << "<" << std::fixed << std::setprecision(0) << snapshot.relativeTime << "s>";
    maker.addText(tableDiv, avg.str());
  }
  else
  {
    tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
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


void WebPageHelper::addOutputModuleTables(XHTMLMaker& maker,
                                          XHTMLMaker::Node *parent,
                           DataSenderMonitorCollection const& dsmc)
{
  DataSenderMonitorCollection::OutputModuleResultsList resultsList =
    dsmc.getTopLevelOutputModuleResults();

  addOutputModuleSummary(maker, parent, resultsList);
  addOutputModuleStatistics(maker, parent, resultsList);
}


void WebPageHelper::addOutputModuleStatistics(XHTMLMaker& maker,
                                              XHTMLMaker::Node *parent,
                                              long long uniqueRBID,
                               DataSenderMonitorCollection const& dsmc)
{
  DataSenderMonitorCollection::OutputModuleResultsList resultsList =
    dsmc.getOutputModuleResultsForRB(uniqueRBID);

  addOutputModuleStatistics(maker, parent, resultsList);
}


void WebPageHelper::addOutputModuleStatistics(XHTMLMaker& maker,
                                              XHTMLMaker::Node *parent,
    DataSenderMonitorCollection::OutputModuleResultsList const& resultsList)
{
  XHTMLMaker::AttrMap colspanAttr;
  colspanAttr[ "colspan" ] = "7";

  XHTMLMaker::AttrMap tableLabelAttr = _tableLabelAttr;
  tableLabelAttr[ "align" ] = "center";

  XHTMLMaker::Node* table = maker.addNode("table", parent, _tableAttr);

  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
  maker.addText(tableDiv, "Received Data Statistics (by output module)");

  // Header
  tableRow = maker.addNode("tr", table, _specialRowAttr);
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "Output Module");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Events");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Size (MB)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Size/Evt (KB)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "RMS (KB)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Min (KB)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Max (KB)");

  if (resultsList.empty())
  {
    XHTMLMaker::AttrMap messageAttr = colspanAttr;
    messageAttr[ "align" ] = "center";

    tableRow = maker.addNode("tr", table, _rowAttr);
    tableDiv = maker.addNode("td", tableRow, messageAttr);
    maker.addText(tableDiv, "No output modules are available yet.");
    return;
  }
  else
  {
    for (unsigned int idx = 0; idx < resultsList.size(); ++idx)
    {
      std::string outputModuleLabel = resultsList[idx]->name;

      tableRow = maker.addNode("tr", table, _rowAttr);
      tableDiv = maker.addNode("td", tableRow);
      maker.addText(tableDiv, outputModuleLabel);
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addInt( tableDiv, resultsList[idx]->eventStats.getSampleCount() );
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addDouble( tableDiv,
                       resultsList[idx]->eventStats.getValueSum()/(double)0x100000 );
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addDouble( tableDiv,
                       resultsList[idx]->eventStats.getValueAverage()/(double)0x400 );
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addDouble( tableDiv,
                       resultsList[idx]->eventStats.getValueRMS()/(double)0x400 );
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addDouble( tableDiv,
                       resultsList[idx]->eventStats.getValueMin()/(double)0x400 );
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addDouble( tableDiv,
                       resultsList[idx]->eventStats.getValueMax()/(double)0x400 );
    }
  }
}


void WebPageHelper::addOutputModuleSummary(XHTMLMaker& maker,
                                           XHTMLMaker::Node *parent,
    DataSenderMonitorCollection::OutputModuleResultsList const& resultsList)
{
  XHTMLMaker::AttrMap colspanAttr;
  colspanAttr[ "colspan" ] = "3";

  XHTMLMaker::AttrMap tableLabelAttr = _tableLabelAttr;
  tableLabelAttr[ "align" ] = "center";

  XHTMLMaker::Node* table = maker.addNode("table", parent, _tableAttr);

  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
  maker.addText(tableDiv, "Output Module Summary");

  // Header
  tableRow = maker.addNode("tr", table, _specialRowAttr);
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "Name");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "ID");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Header Size (bytes)");

  if (resultsList.empty())
  {
    XHTMLMaker::AttrMap messageAttr = colspanAttr;
    messageAttr[ "align" ] = "center";

    tableRow = maker.addNode("tr", table, _rowAttr);
    tableDiv = maker.addNode("td", tableRow, messageAttr);
    maker.addText(tableDiv, "No output modules are available yet.");
    return;
  }
  else
  {
    for (unsigned int idx = 0; idx < resultsList.size(); ++idx)
    {
      tableRow = maker.addNode("tr", table, _rowAttr);
      tableDiv = maker.addNode("td", tableRow);
      maker.addText(tableDiv, resultsList[idx]->name);
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addInt( tableDiv, resultsList[idx]->id );
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addInt( tableDiv, resultsList[idx]->initMsgSize );
    }
  }
}


void WebPageHelper::addResourceBrokerList(XHTMLMaker& maker,
                                          XHTMLMaker::Node *parent,
                           DataSenderMonitorCollection const& dsmc)
{
  DataSenderMonitorCollection::ResourceBrokerResultsList rbResultsList =
    dsmc.getAllResourceBrokerResults();
  std::sort(rbResultsList.begin(), rbResultsList.end(), compareRBResultPtrValues);

  XHTMLMaker::AttrMap colspanAttr;
  colspanAttr[ "colspan" ] = "15";

  XHTMLMaker::AttrMap tableLabelAttr = _tableLabelAttr;
  tableLabelAttr[ "align" ] = "center";

  XHTMLMaker::AttrMap tableSuspiciousValueAttr = _tableValueAttr;
  tableSuspiciousValueAttr[ "style" ] = "background-color: yellow;";

  XHTMLMaker::Node* table = maker.addNode("table", parent, _tableAttr);

  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
  maker.addText(tableDiv, "Data Sender Overview");

  // Header
  tableRow = maker.addNode("tr", table, _specialRowAttr);
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "Resource Broker URL");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "RB instance");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "RB TID");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of FUs");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of INIT messages");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of events");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of error events");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of faulty events");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of outstanding data discards");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of DQM events");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of faulty DQM events");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of outstanding DQM discards");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of ignored discards");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Recent event rate (Hz)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Last event number received");

  if (rbResultsList.empty())
  {
    XHTMLMaker::AttrMap messageAttr = colspanAttr;
    messageAttr[ "align" ] = "center";

    tableRow = maker.addNode("tr", table, _rowAttr);
    tableDiv = maker.addNode("td", tableRow, messageAttr);
    maker.addText(tableDiv, "No data senders have registered yet.");
    return;
  }
  else
  {
    for (unsigned int idx = 0; idx < rbResultsList.size(); ++idx)
    {
      tableRow = maker.addNode("tr", table, _rowAttr);

      tableDiv = maker.addNode("td", tableRow);
      XHTMLMaker::AttrMap linkAttr;
      linkAttr[ "href" ] = baseURL() + "/rbsenderdetail?id=" +
        boost::lexical_cast<std::string>(rbResultsList[idx]->uniqueRBID);
      XHTMLMaker::Node* link = maker.addNode("a", tableDiv, linkAttr);
      maker.addText(link, rbResultsList[idx]->key.hltURL);

      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addInt( tableDiv, rbResultsList[idx]->key.hltInstance );
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addInt( tableDiv, rbResultsList[idx]->key.hltTid );
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addInt( tableDiv, rbResultsList[idx]->filterUnitCount );
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addInt(tableDiv, rbResultsList[idx]->initMsgCount );
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addInt( tableDiv, rbResultsList[idx]->eventStats.getSampleCount() );
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addInt( tableDiv, rbResultsList[idx]->errorEventStats.getSampleCount() );

      if (rbResultsList[idx]->faultyEventStats.getSampleCount() != 0)
      {
        tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
      }
      else
      {
        tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      }
      maker.addInt( tableDiv, rbResultsList[idx]->faultyEventStats.getSampleCount() );

      if (rbResultsList[idx]->outstandingDataDiscardCount != 0)
      {
        tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
      }
      else
      {
        tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      }
      maker.addInt( tableDiv, rbResultsList[idx]->outstandingDataDiscardCount );

      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addInt( tableDiv, rbResultsList[idx]->dqmEventStats.getSampleCount() );

      if (rbResultsList[idx]->faultyDQMEventStats.getSampleCount() != 0)
      {
        tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
      }
      else
      {
        tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      }
      maker.addInt( tableDiv, rbResultsList[idx]->faultyDQMEventStats.getSampleCount() );

      if (rbResultsList[idx]->outstandingDQMDiscardCount != 0)
      {
        tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
      }
      else
      {
        tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      }
      maker.addInt( tableDiv, rbResultsList[idx]->outstandingDQMDiscardCount );

      const int skippedDiscards = rbResultsList[idx]->skippedDiscardStats.getSampleCount();
      if (skippedDiscards != 0)
      {
        tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
      }
      else
      {
        tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      }
      maker.addInt( tableDiv, skippedDiscards );

      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addDouble( tableDiv, rbResultsList[idx]->eventStats.
                       getSampleRate(MonitoredQuantity::RECENT) );
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addInt( tableDiv, rbResultsList[idx]->lastEventNumber );
    }
  }
}


void WebPageHelper::addResourceBrokerDetails(XHTMLMaker& maker,
                                             XHTMLMaker::Node *parent,
                                             long long uniqueRBID,
                              DataSenderMonitorCollection const& dsmc)
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

  XHTMLMaker::AttrMap tableAttr = _tableAttr;
  tableAttr[ "width" ] = "";

  XHTMLMaker::AttrMap tableLabelAttr = _tableLabelAttr;
  tableLabelAttr[ "align" ] = "center";

  XHTMLMaker::Node* table = maker.addNode("table", parent, tableAttr);

  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
  maker.addText(tableDiv, "Resource Broker Details");

  // Header
  tableRow = maker.addNode("tr", table, _specialRowAttr);
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Parameter");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Value");

  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "URL");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  XHTMLMaker::AttrMap linkAttr;
  linkAttr[ "href" ] = rbResultPtr->key.hltURL + "/urn:xdaq-application:lid=" +
    boost::lexical_cast<std::string>(rbResultPtr->key.hltLocalId);
  XHTMLMaker::Node* link = maker.addNode("a", tableDiv, linkAttr);
  maker.addText(link, rbResultPtr->key.hltURL);

  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "Class Name");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addText(tableDiv, rbResultPtr->key.hltClassName);

  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "Instance");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, rbResultPtr->key.hltInstance );

  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "Local ID");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, rbResultPtr->key.hltLocalId );

  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "Tid");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, rbResultPtr->key.hltTid );

  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "INIT Message Count");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, rbResultPtr->initMsgCount );

  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "Event Count");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, rbResultPtr->eventStats.getSampleCount() );

  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "Error Event Count");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, rbResultPtr->errorEventStats.getSampleCount() );

  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "Faulty Event Count");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, rbResultPtr->faultyEventStats.getSampleCount() );

  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "Data Discard Count");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, rbResultPtr->dataDiscardStats.getSampleCount() );

  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "DQM Event Count");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, rbResultPtr->dqmEventStats.getSampleCount() );

  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "Faulty DQM Event Count");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, rbResultPtr->faultyDQMEventStats.getSampleCount() );

  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "DQM Discard Count");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, rbResultPtr->dqmDiscardStats.getSampleCount() );

  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "Ignored Discards Count");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, rbResultPtr->skippedDiscardStats.getSampleCount() );

  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "Last Event Number Received");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, rbResultPtr->lastEventNumber );

  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  maker.addText(tableDiv, "Last Run Number Received");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, rbResultPtr->lastRunNumber );

  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  tmpDuration = static_cast<int>(rbResultPtr->eventStats.recentDuration);
  tmpText =  "Recent (" + boost::lexical_cast<std::string>(tmpDuration) +
    " sec) Event Rate (Hz)";
  maker.addText(tableDiv, tmpText);
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, rbResultPtr->eventStats.recentSampleRate );

  tableRow = maker.addNode("tr", table, _rowAttr);
  tableDiv = maker.addNode("td", tableRow, _tableLabelAttr);
  tmpDuration = static_cast<int>(rbResultPtr->eventStats.fullDuration);
  tmpText =  "Full (" + boost::lexical_cast<std::string>(tmpDuration) +
    " sec) Event Rate (Hz)";
  maker.addText(tableDiv, tmpText);
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, rbResultPtr->eventStats.fullSampleRate );
}


void WebPageHelper::addFilterUnitList(XHTMLMaker& maker,
                                      XHTMLMaker::Node *parent,
                                      long long uniqueRBID,
                                      DataSenderMonitorCollection const& dsmc)
{
  DataSenderMonitorCollection::FilterUnitResultsList fuResultsList =
    dsmc.getFilterUnitResultsForRB(uniqueRBID);

  XHTMLMaker::AttrMap colspanAttr;
  colspanAttr[ "colspan" ] = "13";

  XHTMLMaker::AttrMap tableLabelAttr = _tableLabelAttr;
  tableLabelAttr[ "align" ] = "center";

  XHTMLMaker::AttrMap tableSuspiciousValueAttr = _tableValueAttr;
  tableSuspiciousValueAttr[ "style" ] = "background-color: yellow;";

  XHTMLMaker::Node* table = maker.addNode("table", parent, _tableAttr);

  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow, colspanAttr);
  maker.addText(tableDiv, "Filter Units");

  // Header
  tableRow = maker.addNode("tr", table, _specialRowAttr);
  tableDiv = maker.addNode("th", tableRow);
  maker.addText(tableDiv, "Process ID");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of INIT messages");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of events");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of error events");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of faulty events");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of outstanding data discards");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of DQM events");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of faulty DQM events");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of outstanding DQM discards");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "# of ignored discards");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Recent event rate (Hz)");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Last event number received");
  tableDiv = maker.addNode("th", tableRow, tableLabelAttr);
  maker.addText(tableDiv, "Last run number received");

  if (fuResultsList.empty())
  {
    XHTMLMaker::AttrMap messageAttr = colspanAttr;
    messageAttr[ "align" ] = "center";

    tableRow = maker.addNode("tr", table, _rowAttr);
    tableDiv = maker.addNode("td", tableRow, messageAttr);
    maker.addText(tableDiv, "No filter units have registered yet.");
    return;
  }
  else
  {
    for (unsigned int idx = 0; idx < fuResultsList.size(); ++idx)
    {
      tableRow = maker.addNode("tr", table, _rowAttr);

      tableDiv = maker.addNode("td", tableRow);
      maker.addInt( tableDiv, fuResultsList[idx]->key.fuProcessId );
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addInt( tableDiv, fuResultsList[idx]->initMsgCount );
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addInt( tableDiv, fuResultsList[idx]->shortIntervalEventStats.getSampleCount() );
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addInt( tableDiv, fuResultsList[idx]->errorEventStats.getSampleCount() );

      if (fuResultsList[idx]->faultyEventStats.getSampleCount() != 0)
      {
        tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
      }
      else
      {
        tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      }
      maker.addInt( tableDiv, fuResultsList[idx]->faultyEventStats.getSampleCount() );

      if (fuResultsList[idx]->outstandingDataDiscardCount != 0)
      {
        tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
      }
      else
      {
        tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      }
      maker.addInt( tableDiv, fuResultsList[idx]->outstandingDataDiscardCount );

      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addInt( tableDiv, fuResultsList[idx]->dqmEventStats.getSampleCount() );

      if (fuResultsList[idx]->faultyDQMEventStats.getSampleCount() != 0)
      {
        tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
      }
      else
      {
        tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      }
      maker.addInt( tableDiv, fuResultsList[idx]->faultyDQMEventStats.getSampleCount() );

      if (fuResultsList[idx]->outstandingDQMDiscardCount != 0)
      {
        tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
      }
      else
      {
        tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      }
      maker.addInt( tableDiv, fuResultsList[idx]->outstandingDQMDiscardCount );

      const int skippedDiscards = fuResultsList[idx]->skippedDiscardStats.getSampleCount();
      if (skippedDiscards != 0)
      {
        tableDiv = maker.addNode("td", tableRow, tableSuspiciousValueAttr);
      }
      else
      {
        tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      }
      maker.addInt( tableDiv, skippedDiscards );

      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addDouble( tableDiv, fuResultsList[idx]->shortIntervalEventStats.
                       getSampleRate(MonitoredQuantity::RECENT) );
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addInt( tableDiv, fuResultsList[idx]->lastEventNumber );
      tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
      maker.addInt( tableDiv, fuResultsList[idx]->lastRunNumber );
    }
  }
}


void WebPageHelper::addDQMEventStats
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *table,
  DQMEventMonitorCollection::DQMEventStats const& stats,
  const MonitoredQuantity::DataSetType dataSet
)
{
  // Mean performance header
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("th", tableRow);
  if ( dataSet == MonitoredQuantity::FULL )
    maker.addText(tableDiv, "Mean performance for");
  else
    maker.addText(tableDiv, "Recent performance for last");

  addDurationToTableHead(maker, tableRow,
    stats.dqmEventSizeStats.getDuration(dataSet));
  addDurationToTableHead(maker, tableRow,
    stats.servedDQMEventSizeStats.getDuration(dataSet));
  addDurationToTableHead(maker, tableRow,
    stats.writtenDQMEventSizeStats.getDuration(dataSet));

  addRowForDQMEventsProcessed(maker, table, stats, dataSet);
  addRowForDQMEventBandwidth(maker, table, stats, dataSet);
  if ( dataSet == MonitoredQuantity::FULL )
  {
    addRowForTotalDQMEventVolume(maker, table, stats, dataSet);
  }
  else
  {
    addRowForMaxDQMEventBandwidth(maker, table, stats, dataSet);
    addRowForMinDQMEventBandwidth(maker, table, stats, dataSet);
  }
}


void WebPageHelper::addRowForDQMEventsProcessed
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *table,
  DQMEventMonitorCollection::DQMEventStats const& stats,
  const MonitoredQuantity::DataSetType dataSet
)
{
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
  maker.addText(tableDiv, "Top level folders");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.numberOfGroupsStats.getValueSum(dataSet), 0 );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addInt( tableDiv, stats.servedDQMEventSizeStats.getSampleCount(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.numberOfWrittenGroupsStats.getValueSum(dataSet), 0 );
}


void WebPageHelper::addRowForDQMEventBandwidth
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *table,
  DQMEventMonitorCollection::DQMEventStats const& stats,
  const MonitoredQuantity::DataSetType dataSet
)
{
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
  maker.addText(tableDiv, "Bandwidth (MB/s)");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.dqmEventSizeStats.getValueRate(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.servedDQMEventSizeStats.getValueRate(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.writtenDQMEventSizeStats.getValueRate(dataSet) );
}


void WebPageHelper::addRowForTotalDQMEventVolume
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *table,
  DQMEventMonitorCollection::DQMEventStats const& stats,
  const MonitoredQuantity::DataSetType dataSet
)
{
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
  maker.addText(tableDiv, "Total volume processed (MB)");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.dqmEventSizeStats.getValueSum(dataSet), 3 );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.servedDQMEventSizeStats.getValueSum(dataSet), 3 );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.writtenDQMEventSizeStats.getValueSum(dataSet), 3 );
}


void WebPageHelper::addRowForMaxDQMEventBandwidth
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *table,
  DQMEventMonitorCollection::DQMEventStats const& stats,
  const MonitoredQuantity::DataSetType dataSet
)
{
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
  maker.addText(tableDiv, "Maximum Bandwidth (MB/s)");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.dqmEventBandwidthStats.getValueMax(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.servedDQMEventBandwidthStats.getValueMax(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.writtenDQMEventBandwidthStats.getValueMax(dataSet) );
}


void WebPageHelper::addRowForMinDQMEventBandwidth
(
  XHTMLMaker& maker,
  XHTMLMaker::Node *table,
  DQMEventMonitorCollection::DQMEventStats const& stats,
  const MonitoredQuantity::DataSetType dataSet
)
{
  XHTMLMaker::Node* tableRow = maker.addNode("tr", table, _rowAttr);
  XHTMLMaker::Node* tableDiv = maker.addNode("td", tableRow);
  maker.addText(tableDiv, "Minimum Bandwidth (MB/s)");
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.dqmEventBandwidthStats.getValueMin(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.servedDQMEventBandwidthStats.getValueMin(dataSet) );
  tableDiv = maker.addNode("td", tableRow, _tableValueAttr);
  maker.addDouble( tableDiv, stats.writtenDQMEventBandwidthStats.getValueMin(dataSet) );
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
