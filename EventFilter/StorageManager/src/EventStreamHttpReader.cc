// $Id: EventStreamHttpReader.cc,v 1.45 2011/03/24 17:19:25 mommsen Exp $
/// @file: EventStreamHttpReader.cc

#include "EventFilter/StorageManager/interface/CurlInterface.h"
#include "EventFilter/StorageManager/src/EventServerProxy.icc"
#include "EventFilter/StorageManager/src/EventStreamHttpReader.h"

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
    stor::CurlInterface::Content data;
    unsigned int currentLS(0);
    
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
    }
    while (
      dropOldLumisectionEvents_ &&
      lastLS_ > currentLS
    );
    
    lastLS_ = currentLS;
    return deserializeEvent(EventMsgView(&data[0]));
  }
  
  
  void EventStreamHttpReader::readHeader()
  {
    stor::CurlInterface::Content data;
    
    eventServerProxy_.getInitMsg(data);
    InitMsgView initView(&data[0]);
    deserializeAndMergeWithRegistry(initView);
  }

} //namespace edm


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
