// $Id: WebPageHelper.h,v 1.13 2011/03/07 15:31:32 mommsen Exp $
/// @file: WebPageHelper.h

#ifndef EventFilter_StorageManager_WebPageHelper_h
#define EventFilter_StorageManager_WebPageHelper_h

#include <map>
#include <string>

#include "xdaq/ApplicationDescriptor.h"
#include "xgi/Input.h"
#include "xgi/Output.h"

#include "EventFilter/Utilities/interface/Css.h"

#include "EventFilter/StorageManager/interface/DQMEventMonitorCollection.h"
#include "EventFilter/StorageManager/interface/XHTMLMaker.h"


namespace stor {

  /**
   * Helper class to handle web page requests
   *
   * $Author: mommsen $
   * $Revision: 1.13 $
   * $Date: 2011/03/07 15:31:32 $
   */
  
  template<class T>
  class WebPageHelper
  {
  public:

    WebPageHelper
    (
      xdaq::ApplicationDescriptor*,
      const std::string& cvsVersion,
      T* callee,
      void (T::*addHyperLinks)(XHTMLMaker&, XHTMLMaker::Node*) const
    );

    /**
     * Create event filter style sheet
     */
    void css(xgi::Input *in, xgi::Output *out)
    { css_.css(in,out); }
        
    
  protected:

    /**
      Get base url
    */
    std::string baseURL() const;

    /**
     * Returns the webpage body with the standard header as XHTML node
     */
    XHTMLMaker::Node* createWebPageBody
    (
      XHTMLMaker&,
      const std::string& pageTitle,
      const std::string& externallyVisibleState,
      const std::string& innerStateName,
      const std::string& errorMsg
    ) const;
    
    /**
     * Adds the links for the other hyperdaq webpages
     */
    void addDOMforHyperLinks(XHTMLMaker& maker, XHTMLMaker::Node* parent) const
    { (callee_->*addHyperLinks_)(maker, parent); } 

    /**
     * Add header with integration duration
     */
    void addDurationToTableHead
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* tableRow,
      const utils::Duration_t
    ) const;

    /**
     * Adds DQM event processor statistics to the parent DOM element
     */
    void addDOMforProcessedDQMEvents
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      DQMEventMonitorCollection const&
    ) const;

    /**
     * Adds statistics for the DQM events to the parent DOM element
     */
    void addDOMforDQMEventStatistics
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* parent,
      DQMEventMonitorCollection const&
    ) const;

    /**
     * Add statistics for processed DQM events
     */
    void addDQMEventStats
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      DQMEventMonitorCollection::DQMEventStats const&,
      const MonitoredQuantity::DataSetType
    ) const;
    
    /**
     * Add a table row for number of DQM events processed
     */
    void addRowForDQMEventsProcessed
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      DQMEventMonitorCollection::DQMEventStats const&,
      const MonitoredQuantity::DataSetType
    ) const;

    /**
     * Add a table row for DQM event bandwidth
     */
    void addRowForDQMEventBandwidth
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      DQMEventMonitorCollection::DQMEventStats const&,
      const MonitoredQuantity::DataSetType
    ) const;

    /**
     * Add a table row for total fragment volume received
     */
    void addRowForTotalDQMEventVolume
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      DQMEventMonitorCollection::DQMEventStats const&,
      const MonitoredQuantity::DataSetType
    ) const;

    /**
     * Add a table row for maximum fragment bandwidth
     */
    void addRowForMaxDQMEventBandwidth
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      DQMEventMonitorCollection::DQMEventStats const&,
      const MonitoredQuantity::DataSetType
    ) const;

    /**
     * Add a table row for minimum fragment bandwidth
     */
    void addRowForMinDQMEventBandwidth
    (
      XHTMLMaker& maker,
      XHTMLMaker::Node* table,
      DQMEventMonitorCollection::DQMEventStats const&,
      const MonitoredQuantity::DataSetType
    ) const;


    xdaq::ApplicationDescriptor* appDescriptor_;

    XHTMLMaker::AttrMap tableAttr_;
    XHTMLMaker::AttrMap rowAttr_;
    XHTMLMaker::AttrMap tableLabelAttr_;
    XHTMLMaker::AttrMap tableValueAttr_;
    XHTMLMaker::AttrMap specialRowAttr_;

    std::map<unsigned int, std::string> alarmColors_;

  private:

    //Prevent copying of the WebPageHelper
    WebPageHelper(WebPageHelper const&);
    WebPageHelper& operator=(WebPageHelper const&);

    evf::Css css_;
    const std::string cvsVersion_;
    T* callee_;
    void (T::*addHyperLinks_)(XHTMLMaker&, XHTMLMaker::Node*) const;

  };

} // namespace stor

#endif // EventFilter_StorageManager_WebPageHelper_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
