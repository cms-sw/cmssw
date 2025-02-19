#ifndef POPCON_VerifyState_H
#define POPCON_VerifyState_H
//
// Author: Vincenzo Innocente
//

#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBOutputService/interface/TagInfo.h"
#include "CondCore/DBOutputService/interface/LogDBEntry.h"

#include<string>

namespace popcon {

  /** Verify the consistency between conddb and the log
   */
  class VerifyState {
  public:
    VerifyState(cond::TagInfo const & tagInfo, 
		cond::LogDBEntry const & logDBEntry) :
      m_tagInfo(tagInfo),
      m_logDBEntry(logDBEntry){}
    
    // verify that the last entry in the tag correspond to last entry in the log
    bool selfConsistent() const;

    // verify that last log entry is from this sourceId
    bool consistentWith(std::string & sourceId) const;

    
    

  private:
    
    cond::TagInfo const & m_tagInfo;
    
    cond::LogDBEntry const & m_logDBEntry;
    

  };

}
#endif // POPCON_VerifyState_H
