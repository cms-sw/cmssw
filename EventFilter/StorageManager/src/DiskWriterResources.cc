// $Id: DiskWriterResources.cc,v 1.8 2011/03/07 15:31:32 mommsen Exp $
/// @file: DiskWriterResources.cc

#include "EventFilter/StorageManager/interface/DiskWriterResources.h"

namespace stor
{
  DiskWriterResources::DiskWriterResources() :
    configurationIsNeeded_(false),
    streamChangeIsNeeded_(false),
    fileClosingTestIsNeeded_(false),
    diskWriterIsBusy_(false),
    streamChangeInProgress_(false)
  {
  }

  void DiskWriterResources::requestStreamConfiguration
  (
    EvtStrConfigListPtr const evtStrConfig,
    ErrStrConfigListPtr const errStrConfig,
    DiskWritingParams const& dwParams,
    unsigned int const& runNumber,
    boost::posix_time::time_duration const& timeoutValue
  )
  {
    boost::mutex::scoped_lock sl(streamChangeMutex_);

    requestedEventStreamConfig_ = evtStrConfig;
    requestedErrorStreamConfig_ = errStrConfig;
    requestedDiskWritingParams_ = dwParams;
    requestedRunNumber_ = runNumber;
    requestedTimeout_ = timeoutValue;
    configurationIsNeeded_ = true;
    streamChangeIsNeeded_ = true;
  }

  void DiskWriterResources::requestStreamDestruction()
  {
    boost::mutex::scoped_lock sl(streamChangeMutex_);
    configurationIsNeeded_ = false;
    streamChangeIsNeeded_ = true;
  }

  bool DiskWriterResources::streamChangeRequested
  (
    bool& doConfig,
    EvtStrConfigListPtr& evtStrConfig,
    ErrStrConfigListPtr& errStrConfig,
    DiskWritingParams& dwParams,
    unsigned int& runNumber,
    boost::posix_time::time_duration& timeoutValue
  )
  {
    boost::mutex::scoped_lock sl(streamChangeMutex_);

    if (! streamChangeIsNeeded_) {return false;}

    streamChangeIsNeeded_ = false;

    doConfig = configurationIsNeeded_;
    if (configurationIsNeeded_)
    {
      configurationIsNeeded_ = false;
      evtStrConfig = requestedEventStreamConfig_;
      errStrConfig = requestedErrorStreamConfig_;
      dwParams = requestedDiskWritingParams_;
      runNumber = requestedRunNumber_;
      timeoutValue = requestedTimeout_;
    }

    streamChangeInProgress_ = true;

    return true;
  }

  void DiskWriterResources::waitForStreamChange()
  {
    boost::mutex::scoped_lock sl(streamChangeMutex_);
    if (streamChangeIsNeeded_ || streamChangeInProgress_)
      {
        streamChangeCondition_.wait(sl);
      }
  }

  bool DiskWriterResources::streamChangeOngoing()
  {
    boost::mutex::scoped_lock sl(streamChangeMutex_);
    return (streamChangeIsNeeded_ || streamChangeInProgress_);
  }

  void DiskWriterResources::streamChangeDone()
  {
    boost::mutex::scoped_lock sl(streamChangeMutex_);
    if (streamChangeInProgress_)
      {
        streamChangeCondition_.notify_one();
      }
    streamChangeInProgress_ = false;
  }

  void DiskWriterResources::setBusy(bool isBusyFlag)
  {
    //boost::mutex::scoped_lock sl(generalMutex_);
    diskWriterIsBusy_ = isBusyFlag;
  }

  bool DiskWriterResources::isBusy()
  {
    //boost::mutex::scoped_lock sl(generalMutex_);
    return diskWriterIsBusy_;
  }

} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
