#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"

// ------------------------------------------------------------------------

namespace edm {

  void LogStatistics() {
    edm::MessageLoggerQ::MLqSUM();  // trigger summary info
  }

  bool isDebugEnabled() {
    return ((!edm::MessageDrop::debugAlwaysSuppressed) && edm::MessageDrop::instance()->debugEnabled);
  }

  bool isInfoEnabled() {
    return ((!edm::MessageDrop::infoAlwaysSuppressed) && edm::MessageDrop::instance()->infoEnabled);
  }

  bool isFwkInfoEnabled() {
    return ((!edm::MessageDrop::fwkInfoAlwaysSuppressed) && edm::MessageDrop::instance()->fwkInfoEnabled);
  }

  bool isWarningEnabled() {
    return ((!edm::MessageDrop::warningAlwaysSuppressed) && edm::MessageDrop::instance()->warningEnabled);
  }

  bool isErrorEnabled() { return edm::MessageDrop::instance()->errorEnabled; }

  void HaltMessageLogging() {
    edm::MessageLoggerQ::MLqSHT();  // Shut the logger up
  }

  void FlushMessageLog() {
    if (MessageDrop::instance()->messageLoggerScribeIsRunning != MLSCRIBE_RUNNING_INDICATOR)
      return;
    edm::MessageLoggerQ::MLqFLS();  // Flush the message log queue
  }

  void clearMessageLog() { MessageDrop::instance()->clear(); }

  bool isMessageProcessingSetUp() {
    //  std::cerr << "isMessageProcessingSetUp: \n";
    //  std::cerr << "messageLoggerScribeIsRunning = "
    //            << (int)MessageDrop::instance()->messageLoggerScribeIsRunning << "\n";
    return (MessageDrop::instance()->messageLoggerScribeIsRunning == MLSCRIBE_RUNNING_INDICATOR);
  }

  void GroupLogStatistics(std::string_view category) {
    std::string* cat_p = new std::string(category);
    edm::MessageLoggerQ::MLqGRP(cat_p);  // Indicate a group summary category
    // Note that the scribe will be responsible for deleting cat_p
  }

  edm::LogDebug_::LogDebug_(std::string_view id, std::string_view file, int line) : Log<level::Debug, false>(id) {
    *this << " " << stripLeadingDirectoryTree(file) << ":" << line << "\n";
  }

  std::string_view edm::LogDebug_::stripLeadingDirectoryTree(const std::string_view file) const {
    std::string_view::size_type lastSlash = file.find_last_of('/');
    if (lastSlash == std::string_view::npos)
      return file;
    if (lastSlash == file.size() - 1)
      return file;
    return file.substr(lastSlash + 1, file.size() - lastSlash - 1);
  }

  void setStandAloneMessageThreshold(edm::ELseverityLevel const& severity) {
    edm::MessageLoggerQ::standAloneThreshold(severity);
  }
  void squelchStandAloneMessageCategory(std::string const& category) { edm::MessageLoggerQ::squelch(category); }

}  // namespace edm
