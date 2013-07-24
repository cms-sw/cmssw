// $Id: DQMHttpSource.cc,v 1.32 2012/11/01 17:08:57 wmtan Exp $
/// @file: DQMHttpSource.cc

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "EventFilter/StorageManager/interface/CurlInterface.h"
#include "EventFilter/StorageManager/src/DQMHttpSource.h"
#include "EventFilter/StorageManager/src/EventServerProxy.icc"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/StreamDQMDeserializer.h"

#include "TClass.h"

#include <memory>
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
  eventAuxiliary_(),
  dqmEventServerProxy_(pset),
  dqmStore_(0)
  {}


  bool DQMHttpSource::checkNextEvent() {
    initializeDQMStore();
    stor::CurlInterface::Content data;

    dqmEventServerProxy_.getOneEvent(data);
    if ( data.empty() ) return false;

    HeaderView hdrView(&data[0]);
    if (hdrView.code() == Header::DONE) return false;

    const DQMEventMsgView dqmEventMsgView(&data[0]);
    addEventToDQMBackend(dqmStore_, dqmEventMsgView, true);

    EventID id(dqmEventMsgView.runNumber(), dqmEventMsgView.lumiSection(), dqmEventMsgView.eventNumberAtUpdate());
  
    setEventAuxiliary(std::unique_ptr<EventAuxiliary>(new EventAuxiliary(id, std::string(), dqmEventMsgView.timeStamp(), true, EventAuxiliary::PhysicsTrigger)));

    if(!runAuxiliary() || runAuxiliary()->run() != eventAuxiliary().run()) {
      setRunAuxiliary(new RunAuxiliary(eventAuxiliary().run(), eventAuxiliary().time(), Timestamp::invalidTimestamp()));
    }
    if(!luminosityBlockAuxiliary() || luminosityBlockAuxiliary()->luminosityBlock() != eventAuxiliary().luminosityBlock()) {
      setLuminosityBlockAuxiliary(new LuminosityBlockAuxiliary(eventAuxiliary().run(), eventAuxiliary().luminosityBlock(), eventAuxiliary().time(), Timestamp::invalidTimestamp()));
    }
    setEventCached();
    return true;
  }

  EventPrincipal* DQMHttpSource::read(EventPrincipal& eventPrincipal)
  {
    // make a fake event principal containing no data but the evId and runId from DQMEvent
    // and the time stamp from the event at update
    EventPrincipal* e = makeEvent(
      eventPrincipal,
      eventAuxiliary()
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
    
    MonitorElement* me = dqmStore->get("SM_SMPS_Stats/mergeCount");
    if (!me){
      dqmStore->setCurrentFolder("SM_SMPS_Stats");
      me = dqmStore->bookInt("mergeCount");
    }
    me->Fill(dqmEventMsgView.mergeCount());

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
