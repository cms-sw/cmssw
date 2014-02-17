// $Id: ConsumerWebPageHelper.h,v 1.2 2011/03/07 15:31:31 mommsen Exp $
/// @file: ConsumerWebPageHelper.h

#ifndef EventFilter_StorageManager_ConsumerWebPageHelper_h
#define EventFilter_StorageManager_ConsumerWebPageHelper_h

#include "xdaq/ApplicationDescriptor.h"
#include "xgi/Output.h"

#include "EventFilter/StorageManager/interface/DQMConsumerMonitorCollection.h"
#include "EventFilter/StorageManager/interface/DQMEventQueueCollection.h"
#include "EventFilter/StorageManager/interface/EventConsumerMonitorCollection.h"
#include "EventFilter/StorageManager/interface/EventQueueCollection.h"
#include "EventFilter/StorageManager/interface/RegistrationCollection.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/interface/WebPageHelper.h"
#include "EventFilter/StorageManager/interface/XHTMLMaker.h"

#include <boost/function.hpp>

namespace stor
{

  /**
   * Helper class to handle consumer web page requests
   *
   * $Author: mommsen $
   * $Revision: 1.2 $
   * $Date: 2011/03/07 15:31:31 $
   */

  template<typename WebPageHelper_t, typename EventQueueCollection_t, typename StatisticsReporter_t>
  class ConsumerWebPageHelper : public WebPageHelper<WebPageHelper_t>
  {
  public:

    ConsumerWebPageHelper
    (
      xdaq::ApplicationDescriptor* appDesc,
      const std::string& cvsVersion,
      WebPageHelper_t* webPageHelper,
      void (WebPageHelper_t::*addHyperLinks)(XHTMLMaker&, XHTMLMaker::Node*) const
    );

    /**
       Generates consumer statistics page
    */
    void consumerStatistics
    (
      xgi::Output*,
      const std::string& externallyVisibleState,
      const std::string& innerStateName,
      const std::string& errorMsg,
      boost::shared_ptr<StatisticsReporter_t>,
      RegistrationCollectionPtr,
      boost::shared_ptr<EventQueueCollection_t>,
      DQMEventQueueCollectionPtr
    ) const;
    
    
  private:

    /**
     * Adds statistics for event consumers
     */
    void addDOMforEventConsumers
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      RegistrationCollectionPtr,
      boost::shared_ptr<EventQueueCollection_t>,
      const EventConsumerMonitorCollection&
    ) const;

    /**
     * Adds statistics for DQM event consumers
     */
    void addDOMforDQMEventConsumers
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      RegistrationCollectionPtr,
      DQMEventQueueCollectionPtr,
      const DQMConsumerMonitorCollection&
    ) const;

    /**
     * Add table cell with consumer name. If the consumer is
     * a proxy server, a hyperlink to it will be added.
     * Returns true if the consumer is a proxy server.
     */
    bool addDOMforConsumerName
    (
      stor::XHTMLMaker& maker,
      stor::XHTMLMaker::Node* tableRow,
      const std::string& consumerName
    ) const;
    

    //Prevent copying of the ConsumerWebPageHelper
    ConsumerWebPageHelper(ConsumerWebPageHelper const&);
    ConsumerWebPageHelper& operator=(ConsumerWebPageHelper const&);

    xdaq::ApplicationDescriptor* appDescriptor_;

  };

} // namespace stor

#endif // EventFilter_StorageManager_ConsumerWebPageHelper_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
