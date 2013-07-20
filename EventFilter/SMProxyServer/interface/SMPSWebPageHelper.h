// $Id: SMPSWebPageHelper.h,v 1.4 2012/08/14 11:55:44 davidlt Exp $
/// @file: SMPSWebPageHelper.h

#ifndef EventFilter_SMProxyServer_SMPSWebPageHelper_h
#define EventFilter_SMProxyServer_SMPSWebPageHelper_h

#include "EventFilter/SMProxyServer/interface/EventQueueCollection.h"
#include "EventFilter/SMProxyServer/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/ConsumerWebPageHelper.h"
#include "EventFilter/StorageManager/interface/WebPageHelper.h"


namespace smproxy {

  /**
   * Helper class to handle SM proxy server web page requests
   *
   * $Author: davidlt $
   * $Revision: 1.4 $
   * $Date: 2012/08/14 11:55:44 $
   */
  
  class SMPSWebPageHelper : public stor::WebPageHelper<SMPSWebPageHelper>
  {
  public:

    SMPSWebPageHelper
    (
      xdaq::ApplicationDescriptor*,
      StateMachinePtr
    );

    virtual ~SMPSWebPageHelper() {};
    
    /**
       Generates the default web page
    */
    void defaultWebPage(xgi::Output*) const;
    
    /**
       Generates the data retriever web page
    */
    void dataRetrieverWebPage(xgi::Output*) const;
    
    /**
       Generates the data retriever web page
    */
    void dqmEventStatisticsWebPage(xgi::Output*) const;

    /**
       Generates consumer statistics page
    */
    void consumerStatisticsWebPage(xgi::Output*) const;
    
    
  private:
    
    /**
     * Adds the links for the other hyperdaq webpages
     */
    virtual void addDOMforHyperLinks(stor::XHTMLMaker&, stor::XHTMLMaker::Node* parent) const;

    /**
     * Adds the connection info to the parent DOM element
     */
    void addDOMforConnectionInfo
    (
      stor::XHTMLMaker&,
      stor::XHTMLMaker::Node* parent,
      const DataRetrieverMonitorCollection::SummaryStats&
    ) const;

    /**
     * Adds the summary throuphput per event type to the parent DOM element
     */
    void addDOMforThroughputPerEventType
    (
      stor::XHTMLMaker&,
      stor::XHTMLMaker::Node* parent,
      const DataRetrieverMonitorCollection::SummaryStats&
    ) const;
 
    /**
     * Adds a table row for each event type
     */
    void addRowForEventType
    (
      stor::XHTMLMaker&,
      stor::XHTMLMaker::Node* table,
      DataRetrieverMonitorCollection::SummaryStats::EventTypeStats const&
    ) const;
 
    /**
     * Adds a table row for the summary throughput
     */
    void addSummaryRowForThroughput
    (
      stor::XHTMLMaker&,
      stor::XHTMLMaker::Node* table,
      DataRetrieverMonitorCollection::SummaryStats const&
    ) const;

    /**
     * Adds the event servers to the parent DOM element
     */
    void addDOMforEventServers
    (
      stor::XHTMLMaker&,
      stor::XHTMLMaker::Node* parent
    ) const;
 
    /**
     * Adds a table row for each event server
     */
    void addRowForEventServer
    (
      stor::XHTMLMaker&,
      stor::XHTMLMaker::Node* table,
      DataRetrieverMonitorCollection::EventTypePerConnectionStats const&
    ) const;
 
    /**
     * Adds a summary table row for each event server
     */
    void addSummaryRowForEventServer
    (
      stor::XHTMLMaker&,
      stor::XHTMLMaker::Node* table,
      DataRetrieverMonitorCollection::ConnectionStats::const_iterator
    ) const;
 
    /**
     * Adds a table cell for the SM host
     */
    void addDOMforSMhost
    (
      stor::XHTMLMaker&,
      stor::XHTMLMaker::Node* tableRow,
      const std::string& sourceURL
    ) const;

    /**
     * Adds the DQM event (histogram) servers to the parent DOM element
     */
    void addDOMforDQMEventServers
    (
      stor::XHTMLMaker&,
      stor::XHTMLMaker::Node* parent
    ) const;
    
    //Prevent copying of the SMPSWebPageHelper
    SMPSWebPageHelper(SMPSWebPageHelper const&);
    SMPSWebPageHelper& operator=(SMPSWebPageHelper const&);

    StateMachinePtr stateMachine_;

    typedef stor::ConsumerWebPageHelper<SMPSWebPageHelper,
                                        EventQueueCollection,
                                        StatisticsReporter> ConsumerWebPageHelper_t;
    ConsumerWebPageHelper_t consumerWebPageHelper_;

  };

} // namespace smproxy

#endif // EventFilter_SMProxyServer_SMPSWebPageHelper_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
