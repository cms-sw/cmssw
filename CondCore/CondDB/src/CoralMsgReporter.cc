// Include files
#include <cstdio>
#include <cstring>   // fix bug #58581
#include <iostream>  // fix bug #58581

// Local include files
#include "CondCore/CondDB/interface/Logger.h"
#include "CoralMsgReporter.h"

cond::persistency::MsgDispatcher::MsgDispatcher(Logger& logger) { m_recipient = &logger; }

void cond::persistency::MsgDispatcher::unsubscribe() { m_recipient = nullptr; }

bool cond::persistency::MsgDispatcher::hasRecipient() { return m_recipient != nullptr; }

cond::persistency::Logger& cond::persistency::MsgDispatcher::recipient() { return *m_recipient; }

/// Default constructor
cond::persistency::CoralMsgReporter::CoralMsgReporter()
    : m_dispatcher(), m_level(coral::Error), m_format(0), m_mutex() {
  // Use a non-default format?
  //char* msgformat = getenv ( "CORAL_MSGFORMAT" );
  if (getenv("CORAL_MESSAGEREPORTER_FORMATTED"))
    m_format = 1;

  // Use a non-default message level?
  if (getenv("CORAL_MSGLEVEL")) {
    // Check only the first char of the environment variable
    switch (*getenv("CORAL_MSGLEVEL")) {
      case '0':
      case 'n':
      case 'N':
        m_level = coral::Nil;
        break;

      case '1':
      case 'v':
      case 'V':
        m_level = coral::Verbose;
        break;

      case '2':
      case 'd':
      case 'D':
        m_level = coral::Debug;
        break;

      case '3':
      case 'i':
      case 'I':
        m_level = coral::Info;
        break;

      case '4':
      case 'w':
      case 'W':
        m_level = coral::Warning;
        break;

      case '5':
      case 'e':
      case 'E':
        m_level = coral::Error;
        break;

      case '6':
      case 'f':
      case 'F':
        m_level = coral::Fatal;
        break;

      case '7':
      case 'a':
      case 'A':
        m_level = coral::Always;
        break;

      default:
        break;  // keep the default
    }
  }
}

/// Access output level
coral::MsgLevel cond::persistency::CoralMsgReporter::outputLevel() const { return m_level; }

/// Modify output level
void cond::persistency::CoralMsgReporter::setOutputLevel(coral::MsgLevel lvl) { m_level = lvl; }

void reportToRecipient(const std::string& msg, int lvl, cond::persistency::Logger& recipient) {
  switch (lvl) {
    case coral::Nil:
    case coral::Verbose:
    case coral::Debug:
      recipient.logDebug() << "CORAL: " << msg;
      break;
    case coral::Info:
      recipient.logInfo() << "CORAL: " << msg;
      break;
    case coral::Warning:
      recipient.logWarning() << "CORAL: " << msg;
      break;
    case coral::Error:
      recipient.logError() << "CORAL: " << msg;
      break;
  }
}

/// Report message to stdout
void cond::persistency::CoralMsgReporter::report(int lvl, const std::string&, const std::string& msg) {
  if (lvl < m_level)
    return;
  std::lock_guard<std::recursive_mutex> lock(m_mutex);

  if (m_dispatcher.get() && m_dispatcher->hasRecipient()) {
    reportToRecipient(msg, lvl, m_dispatcher->recipient());
  }
  // Default CORAL reporter
  std::string level("");
  switch (lvl) {
    case coral::Nil:
      level = "Nil";
      break;
    case coral::Verbose:
      level = "Verbose";
      break;
    case coral::Debug:
      level = "Debug";
      break;
    case coral::Info:
      level = "Info";
      break;
    case coral::Warning:
      level = "Warning";
      break;
    case coral::Error:
      level = "Error";
      break;
  }
  std::cout << msg << " " << level << " " << msg << std::endl;
}

void cond::persistency::CoralMsgReporter::subscribe(Logger& logger) {
  m_dispatcher.reset(new MsgDispatcher(logger));
  std::weak_ptr<MsgDispatcher> callBack(m_dispatcher);
  logger.subscribeCoralMessages(callBack);
}
