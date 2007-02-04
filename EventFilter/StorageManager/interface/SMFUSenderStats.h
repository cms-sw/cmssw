#ifndef _smfusenderstats_h_
#define _smfusenderstats_h_

// $Id$ 

#include <vector>

#include "boost/shared_ptr.hpp"

namespace stor {

struct SMFUSenderStats // for FU sender statistics (from SMFUSenderEntry)
{
  SMFUSenderStats(boost::shared_ptr<std::vector<char> > hltURL,
                  boost::shared_ptr<std::vector<char> >  hltClassName,
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
                  double        timewaited);

  boost::shared_ptr<std::vector<char> >  hltURL_;       // FU+HLT identifiers
  boost::shared_ptr<std::vector<char> >  hltClassName_;
  unsigned int  hltLocalId_;
  unsigned int  hltInstance_;
  unsigned int  hltTid_;
  unsigned int  registrySize_;    // size of registry in bytes once received AND copied
  bool          regAllReceived_;  // All Registry fragments are received or not
  unsigned int  totFrames_;    // number of frames in this fragment
  unsigned int  currFrames_;   // current frames received for registry
  bool          regCheckedOK_;    // Registry checked to be same as configuration
  unsigned int  connectStatus_;   // FU+HLT connection status
  double        lastLatency_;     // Latency of last frame in microseconds
  unsigned int  runNumber_;
  bool          isLocal_;         // If detected a locally sent frame chain
  unsigned int  framesReceived_;
  unsigned int  eventsReceived_;
  unsigned int  lastEventID_;
  unsigned int  lastRunID_;
  unsigned int  lastFrameNum_;
  unsigned int  lastTotalFrameNum_;
  unsigned int  totalOutOfOrder_;
  unsigned long long  totalSizeReceived_;// For data only
  unsigned int  totalBadEvents_;   // Update meaning: include original size check?
  double        timeWaited_; // time since last frame in microseconds
};

}
#endif
