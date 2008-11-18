#include "FWCore/MessageLogger/interface/LoggedErrorsSummary.h"
#include "FWCore/MessageLogger/interface/MessageSender.h"

// Change log
//
//  1  mf 8/25/08	First implementation
//			


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
    e.category = (i->first).first;
    e.module   = (i->first).second;
    e.count    = (i->second);
    v.push_back(e);
  }
  MessageSender::freshError = false;
  MessageSender::errorSummaryMap.clear();
  return v;
}

} // end namespace edm
