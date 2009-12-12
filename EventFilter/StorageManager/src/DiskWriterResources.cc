// $Id: DiskWriterResources.cc,v 1.4 2009/09/16 13:45:56 mommsen Exp $
/// @file: DiskWriterResources.cc

#include "EventFilter/StorageManager/interface/DiskWriterResources.h"

namespace stor
{
  DiskWriterResources::DiskWriterResources() :
    _configurationIsNeeded(false),
    _streamChangeIsNeeded(false),
    _fileClosingTestIsNeeded(false),
    _diskWriterIsBusy(false),
    _streamChangeInProgress(false)
  {
  }

  void DiskWriterResources::requestStreamConfiguration
  (
    EvtStrConfigListPtr evtStrConfig,
    ErrStrConfigListPtr errStrConfig,
    double timeoutValue
  )
  {
    boost::mutex::scoped_lock sl(_streamChangeMutex);

    _requestedEventStreamConfig = evtStrConfig;
    _requestedErrorStreamConfig = errStrConfig;
    _requestedTimeout = timeoutValue;
    _configurationIsNeeded = true;
    _streamChangeIsNeeded = true;
  }

  void DiskWriterResources::requestStreamDestruction()
  {
    boost::mutex::scoped_lock sl(_streamChangeMutex);
    _configurationIsNeeded = false;
    _streamChangeIsNeeded = true;
  }

  bool DiskWriterResources::streamChangeRequested
  (
    bool& doConfig,
    EvtStrConfigListPtr& evtStrConfig,
    ErrStrConfigListPtr& errStrConfig,
    double& timeoutValue
  )
  {
    boost::mutex::scoped_lock sl(_streamChangeMutex);

    if (! _streamChangeIsNeeded) {return false;}

    _streamChangeIsNeeded = false;

    doConfig = _configurationIsNeeded;
    if (_configurationIsNeeded)
    {
      _configurationIsNeeded = false;
      evtStrConfig = _requestedEventStreamConfig;
      errStrConfig = _requestedErrorStreamConfig;
      timeoutValue = _requestedTimeout;
    }

    _streamChangeInProgress = true;

    return true;
  }

  void DiskWriterResources::waitForStreamChange()
  {
    boost::mutex::scoped_lock sl(_streamChangeMutex);
    if (_streamChangeIsNeeded || _streamChangeInProgress)
      {
        _streamChangeCondition.wait(sl);
      }
  }

  bool DiskWriterResources::streamChangeOngoing()
  {
    boost::mutex::scoped_lock sl(_streamChangeMutex);
    return (_streamChangeIsNeeded || _streamChangeInProgress);
  }

  void DiskWriterResources::streamChangeDone()
  {
    boost::mutex::scoped_lock sl(_streamChangeMutex);
    if (_streamChangeInProgress)
      {
        _streamChangeCondition.notify_one();
      }
    _streamChangeInProgress = false;
  }

  void DiskWriterResources::requestLumiSectionClosure(const uint32_t lumiSection)
  {
    boost::mutex::scoped_lock sl(_lumiSectionMutex);

    lumiSectionsToClose.push_back(lumiSection);
  }

  bool DiskWriterResources::lumiSectionClosureRequested(uint32_t& lumiSection)
  {
    boost::mutex::scoped_lock sl(_lumiSectionMutex);

    if ( lumiSectionsToClose.empty() ) return false;

    lumiSection = lumiSectionsToClose.front();
    lumiSectionsToClose.pop_front();
    return true;
  }

  void DiskWriterResources::setBusy(bool isBusyFlag)
  {
    //boost::mutex::scoped_lock sl(_generalMutex);
    _diskWriterIsBusy = isBusyFlag;
  }

  bool DiskWriterResources::isBusy()
  {
    //boost::mutex::scoped_lock sl(_generalMutex);
    return _diskWriterIsBusy;
  }

} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
