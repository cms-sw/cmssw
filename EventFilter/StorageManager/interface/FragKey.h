#ifndef EventFilter_StorageManager_FragKey_h
#define EventFilter_StorageManager_FragKey_h

// -*- C++ -*-

/*
 */

#include "IOPool/Streamer/interface/MsgTools.h"

namespace stor 
{

  struct FragKey
  {
    /** @deprecated */
    FragKey(uint8 msgcode, uint32 run, uint32 event, uint32 secondaryId):
      code_(msgcode), run_(run), event_(event), secondaryId_(secondaryId) {}
    FragKey(uint8 msgcode, uint32 run, uint32 event, uint32 secondaryId,
	    uint32 originatorPid, uint32 originatorGuid):
      code_(msgcode), run_(run), event_(event), secondaryId_(secondaryId),
      originatorPid_(originatorPid), originatorGuid_(originatorGuid) {}
    bool operator<(FragKey const& b) const {
      if(code_ != b.code_) return code_ < b.code_;
      if(run_ != b.run_) return run_ < b.run_;
      if(event_ != b.event_) return event_ < b.event_;
      if(originatorPid_ != b.originatorPid_) return originatorPid_ < b.originatorPid_;
      if(originatorGuid_ != b.originatorGuid_) return originatorGuid_ < b.originatorGuid_;
      return secondaryId_ < b.secondaryId_;
    }
    // the data for the key
    uint8 code_;
    uint32 run_;
    uint32 event_;
    // the secondary ID is populated with different values depending
    // on the context.  For EVENT messages, the output module ID is used.
    // For DQMEVENT messages, the folder ID is used.
    uint32 secondaryId_;
    uint32 originatorPid_;
    uint32 originatorGuid_;
  };

}

#endif
