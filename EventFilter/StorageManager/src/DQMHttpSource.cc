// $Id: DQMHttpSource.cc,v 1.22.4.1 2011/03/07 11:33:04 mommsen Exp $
/// @file: DQMHttpSource.cc

#include "EventFilter/StorageManager/interface/CurlInterface.h"
#include "EventFilter/StorageManager/src/DQMHttpSource.h"
#include "EventFilter/StorageManager/src/EventServerProxy.icc"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/StreamDQMDeserializer.h"

#include "TClass.h"

#include <string>
#include <vector>


namespace edm
{
  boost::mutex DQMHttpSource::mutex_;
  
  
  DQMHttpSource::DQMHttpSource
  (
    const ParameterSet& pset,
    const InputSourceDescription& desc
  ) :
  edm::RawInputSource(pset, desc),
  dqmStore_(0),
  dqmEventServerProxy_(pset)
  {}


  std::auto_ptr<Event> DQMHttpSource::readOneEvent()
  {
    initializeDQMStore();
    stor::CurlInterface::Content data;

    dqmEventServerProxy_.getOneEvent(data);
    if ( data.empty() ) return std::auto_ptr<edm::Event>();

    HeaderView hdrView(&data[0]);
    if (hdrView.code() == Header::DONE)
      return std::auto_ptr<edm::Event>();

    const DQMEventMsgView dqmEventMsgView(&data[0]);
    addEventToDQMBackend(dqmStore_, dqmEventMsgView, true);

    // make a fake event containing no data but the evId and runId from DQMEvent
    // and the time stamp from the event at update
    std::auto_ptr<Event> e = makeEvent(
      dqmEventMsgView.runNumber(),
      dqmEventMsgView.lumiSection(),
      dqmEventMsgView.eventNumberAtUpdate(),
      dqmEventMsgView.timeStamp()
    );

    return e;
  }
  
  
  void DQMHttpSource::addEventToDQMBackend
  (
    DQMStore* dqmStore,
    const DQMEventMsgView& dqmEventMsgView,
    const bool overwrite
  )
  {
    boost::mutex::scoped_lock sl(mutex_);

    edm::StreamDQMDeserializer deserializeWorker;
    std::auto_ptr<DQMEvent::TObjectTable> toTablePtr =
      deserializeWorker.deserializeDQMEvent(dqmEventMsgView);
    
    for (DQMEvent::TObjectTable::const_iterator tableIter = toTablePtr->begin(),
           tableIterEnd = toTablePtr->end(); tableIter != tableIterEnd; ++tableIter)
    {
      const std::string subFolderName = tableIter->first;
      std::vector<TObject*> toList = tableIter->second;
      dqmStore->setCurrentFolder(subFolderName); // creates dir if needed

      for (std::vector<TObject*>::const_iterator objectIter = toList.begin(),
             objectIterEnd = toList.end(); objectIter != objectIterEnd; ++objectIter)
      {
        dqmStore->extract(*objectIter, dqmStore->pwd(), overwrite);
        // TObject cloned into DQMStore. Thus, delete it here.
        delete *objectIter;
      }
    }
  }
  

  void DQMHttpSource::initializeDQMStore()
  {
    if ( ! dqmStore_ )
      dqmStore_ = edm::Service<DQMStore>().operator->();
    
    if ( ! dqmStore_ )
      throw cms::Exception("readOneEvent", "DQMHttpSource")
        << "Unable to lookup the DQMStore service!\n";
  }

} // namespace edm


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
