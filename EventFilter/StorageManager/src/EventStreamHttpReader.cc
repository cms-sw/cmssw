// $Id: EventStreamHttpReader.cc,v 1.47 2011/04/04 16:05:37 mommsen Exp $
/// @file: EventStreamHttpReader.cc

#include "DQMServices/Core/interface/MonitorElement.h"
#include "EventFilter/StorageManager/interface/CurlInterface.h"
#include "EventFilter/StorageManager/src/EventServerProxy.icc"
#include "EventFilter/StorageManager/src/EventStreamHttpReader.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

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
  dropOldLumisectionEvents_(pset.getUntrackedParameter<bool>("dropOldLumisectionEvents", false)),
  lastLS_(0)
  {
    // Default in StreamerInputSource is 'false'
    inputFileTransitionsEachEvent_ =
      pset.getUntrackedParameter<bool>("inputFileTransitionsEachEvent", true);

    readHeader();
  }
  
  
  EventPrincipal* EventStreamHttpReader::read()
  {
    initializeDQMStore();

    stor::CurlInterface::Content data;
    unsigned int currentLS(0);
    unsigned int droppedEvents(0);
    
    do
    {
      eventServerProxy_.getOneEvent(data);
      if ( data.empty() ) return 0;
      
      HeaderView hdrView(&data[0]);
      if (hdrView.code() == Header::DONE)
      {
        setEndRun();
        return 0;
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

    std::cout << "droppedEventsCount: " << droppedEvents << std::endl;
    
    dqmStore_->cd();
    MonitorElement* me = dqmStore_->bookInt("droppedEventsCount");
    me->Fill(droppedEvents);

    return deserializeEvent(EventMsgView(&data[0]));
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
    if ( ! dqmStore_ )
      dqmStore_ = edm::Service<DQMStore>().operator->();
    
    if ( ! dqmStore_ )
      throw cms::Exception("read", "EventStreamHttpReader")
        << "Unable to lookup the DQMStore service!\n";
  }


} //namespace edm


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
