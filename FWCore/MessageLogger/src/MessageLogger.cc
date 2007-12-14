#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Change Log
//
// 12/12/07  mf   elimination of dummyLogDebugObject_, dummyLogTraceObject_
//		 (see change log 8 in MessageLogger.h)
//
// 12/14/07  mf  Moved the static free function onlyLowestDirectory
//		 to a class member function of LogDebug_, changing
//		 name to a more descriptive stripLeadingDirectoryTree.
//		 Cures the 2600-copies-of-this-function complaint.
//		 Implementation of this is moved into this .cc file.
//
// ------------------------------------------------------------------------

namespace edm {

void LogStatistics() { 
  edm::MessageLoggerQ::MLqSUM ( ); // trigger summary info
}

bool isDebugEnabled() {
  return ( edm::MessageDrop::instance()->debugEnabled );
}

bool isInfoEnabled() {
  return( edm::MessageDrop::instance()->infoEnabled );
}

bool isWarningEnabled() {
  return( edm::MessageDrop::instance()->warningEnabled );
}

void HaltMessageLogging() {
  edm::MessageLoggerQ::MLqSHT ( ); // Shut the logger up
}

void FlushMessageLog() {
  edm::MessageLoggerQ::MLqFLS ( ); // Flush the message log queue
}

void GroupLogStatistics(std::string const & category) {
  std::string * cat_p = new std::string(category);
  edm::MessageLoggerQ::MLqGRP (cat_p); // Indicate a group summary category
  // Note that the scribe will be responsible for deleting cat_p
}

std::string
edm::LogDebug_::stripLeadingDirectoryTree(const std::string & file) const {
  std::string::size_type lastSlash = file.find_last_of('/');
  if (lastSlash == std::string::npos) return file;
  if (lastSlash == file.size()-1)     return file;
  return file.substr(lastSlash+1, file.size()-lastSlash-1);
}

}  // namespace edm
