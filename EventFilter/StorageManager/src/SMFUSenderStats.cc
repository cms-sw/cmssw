
// $Id: SMFUSenderStats.cc,v 1.1 2007/02/04 06:29:40 hcheung Exp $



#include "EventFilter/StorageManager/interface/SMFUSenderStats.h"

namespace stor {

SMFUSenderStats::SMFUSenderStats(boost::shared_ptr<std::vector<char> > hltURL,
                  boost::shared_ptr<std::vector<char> > hltClassName,
                  unsigned int  hltLocalId,
                  unsigned int  hltInstance,
                  unsigned int  hltTid,
                  unsigned int  registrySize,
                  bool          regAllReceived,
                  unsigned int  totFrames,
                  unsigned int  currFrames,
                  bool          regCheckedOK,
                  unsigned int  connectStatus,
                  double        lastLatency,
                  unsigned int  runNumber,
                  bool          isLocal,
                  unsigned int  framesReceived,
                  unsigned int  eventsReceived,
                  unsigned int  lastEventID,
                  unsigned int  lastRunID,
                  unsigned int  lastFrameNum,
                  unsigned int  lastTotalFrameNum,
                  unsigned int  totalOutOfOrder,
                  unsigned long long  totalSizeReceived,
                  unsigned int  totalBadEvents,
                  double        timewaited):
  hltURL_(hltURL), 
  hltClassName_(hltClassName),
  hltLocalId_(hltLocalId),
  hltInstance_(hltInstance),
  hltTid_(hltTid),
  registrySize_(registrySize),
  regAllReceived_(regAllReceived),
  totFrames_(totFrames),
  currFrames_(currFrames),
  regCheckedOK_(regCheckedOK),
  connectStatus_(connectStatus),
  lastLatency_(lastLatency),
  runNumber_(runNumber),
  isLocal_(isLocal),
  framesReceived_(framesReceived),
  eventsReceived_(eventsReceived),
  lastEventID_(lastEventID),
  lastRunID_(lastRunID),
  lastFrameNum_(lastFrameNum),
  lastTotalFrameNum_(lastTotalFrameNum),
  totalOutOfOrder_(totalOutOfOrder),
  totalSizeReceived_(totalSizeReceived),
  totalBadEvents_(totalBadEvents),
  timeWaited_(timewaited)
{
}

}
