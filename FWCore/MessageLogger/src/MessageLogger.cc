#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"

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
//  6/20/08  mf  Have flushMessageLog() check messageLoggerScribeIsRunning
//		 (in the message drop) to avoid hangs if that thread is not
//		 around.
//
//  11/18/08 wmtan  Use explicit non-inlined destructors
//
//  8/11/09  mf setStandAloneMessageThreshold() and
//		squelchStandAloneMessageCategory()
//
//  10/29/09 wmtan  Use explicit non-inlined constructors for LogDebug_ 
//                  and LogTrace_
//
//  8/11/09  mf setStandAloneMessageThreshold() and
//		squelchStandAloneMessageCategory()
//
//  9/23/10  mf Initialize debugEnabled according to 
// 		MessageDrop::debugAlwaysSuppressed, rather than
//		just true.  See change 21 of MessageLogger.h.
//
//  9/27/10  mf isDebugEnabled() - check that debugAlwaysSuppressed is
//              false before examining debugEnabled, which in principle 
//		ougth to be thread-specific thus more expensive to look at.
//
//  9/27/10b mf dtor for LogWarningThatSuppressesLikeLogInfo - see
//		change log 22 in MessageLogger.h
//
// ------------------------------------------------------------------------

namespace edm {

LogInfo::~LogInfo() {}
LogWarning::~LogWarning() {}
LogError::~LogError() {}
LogAbsolute::~LogAbsolute() {}
LogSystem::~LogSystem() {}
LogVerbatim::~LogVerbatim() {}
LogDebug_::~LogDebug_() {}
LogTrace_::~LogTrace_() {}
LogPrint::~LogPrint() {}
LogProblem::~LogProblem() {}
LogImportant::~LogImportant() {}
namespace edmmltest {						// 9/27/10b mf
 LogWarningThatSuppressesLikeLogInfo::~LogWarningThatSuppressesLikeLogInfo() {}
}

void LogStatistics() {
  edm::MessageLoggerQ::MLqSUM ( ); // trigger summary info
}

bool isDebugEnabled() {
  return ((!edm::MessageDrop::debugAlwaysSuppressed)		// 9/27/10 mf
         && edm::MessageDrop::debugEnabled );
}

bool isInfoEnabled() {
  return ((!edm::MessageDrop::infoAlwaysSuppressed)		// 9/27/10 mf
         && edm::MessageDrop::infoEnabled );
}

bool isWarningEnabled() {
  return ((!edm::MessageDrop::warningAlwaysSuppressed)		// 9/27/10 mf
         && edm::MessageDrop::warningEnabled );
}

void HaltMessageLogging() {
  edm::MessageLoggerQ::MLqSHT ( ); // Shut the logger up
}

void FlushMessageLog() {
  if (MessageDrop::instance()->messageLoggerScribeIsRunning !=
  			MLSCRIBE_RUNNING_INDICATOR) return; 	// 6/20/08 mf
  edm::MessageLoggerQ::MLqFLS ( ); // Flush the message log queue
}

bool isMessageProcessingSetUp() {				// 6/20/08 mf
//  std::cerr << "isMessageProcessingSetUp: \n";
//  std::cerr << "messageLoggerScribeIsRunning = "
//  	    << (int)MessageDrop::instance()->messageLoggerScribeIsRunning << "\n";
  return (MessageDrop::instance()->messageLoggerScribeIsRunning ==
  			MLSCRIBE_RUNNING_INDICATOR);
}

void GroupLogStatistics(std::string const & category) {
  std::string * cat_p = new std::string(category);
  edm::MessageLoggerQ::MLqGRP (cat_p); // Indicate a group summary category
  // Note that the scribe will be responsible for deleting cat_p
}

edm::LogDebug_::LogDebug_( std::string const & id, std::string const & file, int line )
  : ap( new MessageSender(ELsuccess,id) )
  , debugEnabled(!MessageDrop::debugAlwaysSuppressed)		//9/23/10 mf
{ *this
        << " "
        << stripLeadingDirectoryTree(file)
        << ":" << line << "\n"; }

std::string
edm::LogDebug_::stripLeadingDirectoryTree(const std::string & file) const {
  std::string::size_type lastSlash = file.find_last_of('/');
  if (lastSlash == std::string::npos) return file;
  if (lastSlash == file.size()-1)     return file;
  return file.substr(lastSlash+1, file.size()-lastSlash-1);
}

edm::LogTrace_::LogTrace_( std::string const & id )
  : ap( new MessageSender(ELsuccess,id,true) )
  , debugEnabled(!MessageDrop::debugAlwaysSuppressed)		//9/23/10 mf
  {  }

void setStandAloneMessageThreshold(std::string const & severity) {
  edm::MessageLoggerQ::standAloneThreshold(severity);
}
void squelchStandAloneMessageCategory(std::string const & category){
  edm::MessageLoggerQ::squelch(category);
}

}  // namespace edm
