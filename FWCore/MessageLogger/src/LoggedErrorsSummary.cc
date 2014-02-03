#include "FWCore/MessageLogger/interface/LoggedErrorsSummary.h"
#include "FWCore/MessageLogger/interface/MessageSender.h"

// Change log
//
//  1  mf 8/25/08	First implementation
//			
//  2  mf 6/22/09	Change to use severity in the key by using entry as key
//			Also, LoggedErrorsOnlySummary()
namespace edm {

bool EnableLoggedErrorsSummary() {
  bool ret = MessageSender::errorSummaryIsBeingKept;
  MessageSender::errorSummaryIsBeingKept = true;
  return ret;
}

bool DisableLoggedErrorsSummary(){
  bool ret = MessageSender::errorSummaryIsBeingKept;
  MessageSender::errorSummaryIsBeingKept = false;
  return ret;
}

bool FreshErrorsExist() {
  return  MessageSender::freshError;
}

std::vector<ErrorSummaryEntry> LoggedErrorsSummary() {
  std::vector<ErrorSummaryEntry> v;
  ErrorSummaryEntry e;
  ErrorSummaryMapIterator end = MessageSender::errorSummaryMap.end();
  for (ErrorSummaryMapIterator i = MessageSender::errorSummaryMap.begin();
  	i != end; ++i) {
    e       = i->first;    // sets category, module and severity ChangeLog 2
    e.count = (i->second); // count is 0 in key; set it to correct value
    v.push_back(e);
  }
  MessageSender::freshError = false;
  MessageSender::errorSummaryMap.clear();
  return v;
}

std::vector<ErrorSummaryEntry> LoggedErrorsOnlySummary() {    //  ChangeLog 2
  std::vector<ErrorSummaryEntry> v;
  ErrorSummaryEntry e;
  ErrorSummaryMapIterator end = MessageSender::errorSummaryMap.end();
  for (ErrorSummaryMapIterator i = MessageSender::errorSummaryMap.begin();
  	i != end; ++i) {
    e = i->first;    
    if (e.severity >= edm::ELerror) {
      e.count = (i->second); 
      v.push_back(e);
    }
  }
  MessageSender::freshError = false;
  MessageSender::errorSummaryMap.clear();
  return v;
}

} // end namespace edm
