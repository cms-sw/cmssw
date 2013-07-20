// $Id: EventStreamHttpReader.cc,v 1.54 2012/10/31 17:09:27 wmtan Exp $
/// @file: EventStreamHttpReader.cc

#include "DQMServices/Core/interface/MonitorElement.h"
#include "EventFilter/StorageManager/interface/CurlInterface.h"
#include "EventFilter/StorageManager/src/EventServerProxy.icc"
#include "EventFilter/StorageManager/src/EventStreamHttpReader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <string>


namespace edm
{  
  EventStreamHttpReader::EventStreamHttpReader
  (
    ParameterSet const& pset,
    InputSourceDescription const& desc
  ):
  StreamerInputSource(pset, desc),
  eventServerProxy_(pset),
  dqmStore_(0),
  dqmStoreAvailabiltyChecked_(false),
  dropOldLumisectionEvents_(pset.getUntrackedParameter<bool>("dropOldLumisectionEvents", false)),
  consumerName_(pset.getUntrackedParameter<std::string>("consumerName")),
  totalDroppedEvents_(0),
  lastLS_(0)
  {
    // Default in StreamerInputSource is 'false'
    bool inputFileTransitionsEachEvent = pset.getUntrackedParameter<bool>("inputFileTransitionsEachEvent", true);
    if(inputFileTransitionsEachEvent) {
      setInputFileTransitionsEachEvent();
    }

    readHeader();
  }
  
  bool EventStreamHttpReader::checkNextEvent()
  {
    initializeDQMStore();

    stor::CurlInterface::Content data;
    unsigned int currentLS(0);
    unsigned int droppedEvents(0);
    
    do
    {
      eventServerProxy_.getOneEvent(data);
      if ( data.empty() ) return false;
      
      HeaderView hdrView(&data[0]);
      if (hdrView.code() == Header::DONE)
      {
        return false;
      }
      
      EventMsgView eventView(&data[0]);
      currentLS = eventView.lumi();
      droppedEvents += eventView.droppedEventsCount();
    }
    while (
      dropOldLumisectionEvents_ &&
      lastLS_ > currentLS
    );
    
    lastLS_ = currentLS;

    if (dqmStore_)
    {
      MonitorElement* me = dqmStore_->get("SM_SMPS_Stats/droppedEventsCount_" + consumerName_ );
      if (!me){
        dqmStore_->setCurrentFolder("SM_SMPS_Stats");
        me = dqmStore_->bookInt("droppedEventsCount_" + consumerName_ );
      }
      totalDroppedEvents_ += droppedEvents;
      me->Fill(totalDroppedEvents_);
    }

    deserializeEvent(EventMsgView(&data[0]));
    return true;
  }
  
  
  void EventStreamHttpReader::readHeader()
  {
    stor::CurlInterface::Content data;
    
    eventServerProxy_.getInitMsg(data);
    InitMsgView initView(&data[0]);
    deserializeAndMergeWithRegistry(initView);
  }
  
  
  void EventStreamHttpReader::initializeDQMStore()
  {
    if ( ! dqmStoreAvailabiltyChecked_ )
    {   
      try
      {
        dqmStore_ = edm::Service<DQMStore>().operator->();
      }
      catch (cms::Exception& e)
      {
        edm::LogInfo("EventStreamHttpReader")
          << "Service DQMStore not defined. Will not record the number of dropped events.";
      }
      dqmStoreAvailabiltyChecked_ = true;
    }
  }


} //namespace edm


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
