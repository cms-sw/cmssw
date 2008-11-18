
// $Id: SMFUSenderStats.cc,v 1.3 2008/05/15 13:57:43 hcheung Exp $



#include "EventFilter/StorageManager/interface/SMFUSenderStats.h"

namespace stor {

SMFUSenderStats::SMFUSenderStats(boost::shared_ptr<std::vector<char> > hltURL,
                  boost::shared_ptr<std::vector<char> > hltClassName,
                  unsigned int  hltLocalId,
                  unsigned int  hltInstance,
                  unsigned int  hltTid,
                  uint32        fuID,
                  SMFUSenderRegCollection RegistryCollection,
                  SMFUSenderDatCollection DatCollection,
                  unsigned int  connectStatus,
                  double        lastLatency,
                  unsigned int  runNumber,
                  bool          isLocal,
                  unsigned int  framesReceived,
                  unsigned int  eventsReceived,
                  unsigned int  lastEventID,
                  unsigned int  lastRunID,
                  unsigned int  totalOutOfOrder,
                  unsigned long long  totalSizeReceived,
                  unsigned int  totalBadEvents,
                  double        timewaited):
  hltURL_(hltURL), 
  hltClassName_(hltClassName),
  hltLocalId_(hltLocalId),
  hltInstance_(hltInstance),
  hltTid_(hltTid),
  fuID_(fuID),
  registryCollection_(RegistryCollection),
  datCollection_(DatCollection),
  connectStatus_(connectStatus),
  lastLatency_(lastLatency),
  runNumber_(runNumber),
  isLocal_(isLocal),
  framesReceived_(framesReceived),
  eventsReceived_(eventsReceived),
  lastEventID_(lastEventID),
  lastRunID_(lastRunID),
  totalOutOfOrder_(totalOutOfOrder),
  totalSizeReceived_(totalSizeReceived),
  totalBadEvents_(totalBadEvents),
  timeWaited_(timewaited)
{
}

}
