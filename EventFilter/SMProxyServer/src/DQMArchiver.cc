// $Id: DQMArchiver.cc,v 1.3 2011/03/30 15:33:43 mommsen Exp $
/// @file: DQMArchiver.cc

#include "DQMServices/Core/interface/DQMStore.h"
#include "EventFilter/SMProxyServer/interface/DQMArchiver.h"
#include "EventFilter/SMProxyServer/interface/Exception.h"
#include "EventFilter/StorageManager/interface/ConsumerID.h"
#include "EventFilter/StorageManager/interface/DQMEventMonitorCollection.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/src/DQMHttpSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TObject.h"

#include <boost/foreach.hpp>

#include <memory>
#include <string>
#include <vector>

namespace smproxy
{
  DQMArchiver::DQMArchiver(StateMachine* stateMachine) :
  stateMachine_(stateMachine),
  dqmArchivingParams_(stateMachine->getConfiguration()->getDQMArchivingParams()),
  dqmEventQueueCollection_(stateMachine->getDQMEventQueueCollection())
  {
    if ( dqmArchivingParams_.archiveDQM_ )
    {
      createRegistration();
      thread_.reset(
        new boost::thread( boost::bind(&DQMArchiver::activity, this) )
      );
    }
  }

  DQMArchiver::~DQMArchiver()
  {
    if (thread_) thread_->join();
  }
  
  void DQMArchiver::activity()
  {
    try
    {
      doIt();
    }
    catch(xcept::Exception &e)
    {
      stateMachine_->moveToFailedState(e);
    }
    catch(std::exception &e)
    {
      XCEPT_DECLARE(exception::DQMArchival,
        sentinelException, e.what());
      stateMachine_->moveToFailedState(sentinelException);
    }
    catch(...)
    {
      std::string errorMsg = "Unknown exception in DQM archiving thread";
      XCEPT_DECLARE(exception::DQMArchival,
        sentinelException, errorMsg);
      stateMachine_->moveToFailedState(sentinelException);
    }
  }
  
  void DQMArchiver::doIt()
  {
    stor::RegistrationCollectionPtr registrationCollection =
      stateMachine_->getRegistrationCollection();
    const stor::ConsumerID cid = regPtr_->consumerId();
    
    while ( registrationCollection->registrationIsAllowed(cid) )
    {
      const stor::DQMEventQueueCollection::ValueType dqmEvent =
        dqmEventQueueCollection_->popEvent(cid);
      
      if ( dqmEvent.first.empty() )
        ::sleep(1);
      else
        handleDQMEvent(dqmEvent.first);
    }

    // run ended, write the last updates to file
    BOOST_FOREACH(
      const Records::value_type& pair,
      lastUpdateForFolders_
    )
    {
      writeDQMEventToFile(pair.second.getDQMEventMsgView(), true);
    }
  }

  void DQMArchiver::handleDQMEvent(const stor::DQMTopLevelFolder::Record& record)
  {
    const DQMEventMsgView view = record.getDQMEventMsgView();

    updateLastRecord(record);

    if (
      dqmArchivingParams_.archiveIntervalDQM_ > 0 &&
      ((view.updateNumber()+1) % dqmArchivingParams_.archiveIntervalDQM_) == 0
    )
    {
      writeDQMEventToFile(view, false);
    }
  }

  void DQMArchiver::updateLastRecord(const stor::DQMTopLevelFolder::Record& record)
  {
    const DQMEventMsgView view = record.getDQMEventMsgView();
    const std::string topFolderName = view.topFolderName();
    Records::iterator pos = lastUpdateForFolders_.lower_bound(topFolderName);

    if (pos != lastUpdateForFolders_.end() &&
      !(lastUpdateForFolders_.key_comp()(topFolderName, pos->first)))
    {
      // key already exists
      pos->second = record;
    }
    else
    {
      lastUpdateForFolders_.insert(pos, Records::value_type(topFolderName, record));
    }
  }

  void DQMArchiver::writeDQMEventToFile
  (
    const DQMEventMsgView& view,
    const bool endRun
  ) const
  {
    edm::ParameterSet dqmStorePSet;
    dqmStorePSet.addUntrackedParameter<bool>("collateHistograms", true);
    DQMStore dqmStore(dqmStorePSet);

    std::ostringstream fileName;
    fileName << dqmArchivingParams_.filePrefixDQM_
      << "/DQM_R"
      << std::setfill('0') << std::setw(9) << view.runNumber();
    if ( ! endRun ) fileName << "_L" << std::setw(6) << view.lumiSection();
    fileName << ".root";
    
    // don't require that the file exists
    dqmStore.load(fileName.str(), DQMStore::StripRunDirs, false);

    edm::DQMHttpSource::addEventToDQMBackend(&dqmStore, view, false);

    dqmStore.save(fileName.str());

    stor::DQMEventMonitorCollection& demc =
      stateMachine_->getStatisticsReporter()->getDQMEventMonitorCollection();
    demc.getWrittenDQMEventSizeMQ().addSample(
      static_cast<double>(view.size()) / 0x100000
    );
    demc.getNumberOfWrittenTopLevelFoldersMQ().addSample(1);
  }

  void DQMArchiver::createRegistration()
  {
    edm::ParameterSet pset;
    pset.addUntrackedParameter<std::string>("DQMconsumerName", "DQMArchiver");
    pset.addUntrackedParameter<std::string>("topLevelFolderName", 
      dqmArchivingParams_.archiveTopLevelFolder_);

    regPtr_.reset( new stor::DQMEventConsumerRegistrationInfo(pset,
        stateMachine_->getConfiguration()->getEventServingParams(),
        "internal")
    );
    
    stor::RegistrationCollectionPtr registrationCollection =
      stateMachine_->getRegistrationCollection();
    
    const stor::ConsumerID cid = registrationCollection->getConsumerId();
    regPtr_->setConsumerId(cid);

    const stor::QueueID qid = dqmEventQueueCollection_->createQueue(regPtr_);
    regPtr_->setQueueId(qid);

    registrationCollection->addRegistrationInfo(regPtr_);
  }

} // namespace smproxy
  
/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
