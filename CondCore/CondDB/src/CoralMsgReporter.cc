// Include files
#include <cstdio>
#include <cstring>   // fix bug #58581
#include <iostream>  // fix bug #58581

// Local include files
#include "CondCore/CondDB/interface/Logger.h"
#include "CoralMsgReporter.h"

/// Default constructor
cond::persistency::CoralMsgReporter::CoralMsgReporter(Logger& logger)
    : m_logger(logger), m_level(coral::Error), m_format(0), m_mutex() {
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

/// Report message to stdout
void cond::persistency::CoralMsgReporter::report(int lvl, const std::string& src, const std::string& msg) {
  if (lvl < m_level)
    return;
  std::lock_guard<std::recursive_mutex> lock(m_mutex);

  std::stringstream out;
  /**
  if ( m_format == 1 ) // COOL format
  {
    // Formatted CORAL reporter (as in COOL)
    //std::ostream& out = std::cout;
    const std::string::size_type src_name_maxsize = 36;
    if ( src.size() <= src_name_maxsize )
    {
      out << src << std::string( src_name_maxsize-src.size(), ' ' );
    }
    else
    {
      out << src.substr( 0, src_name_maxsize-3 ) << "...";
    }
    switch ( lvl )
    {
    case 0:  out << " Nil      "; break;
    case 1:  out << " Verbose  "; break;
    case 2:  out << " Debug    "; break;
    case 3:  out << " Info     "; break;
    case 4:  out << " Warning  "; break;
    case 5:  out << " Error    "; break;
    case 6:  out << " Fatal    "; break;
    case 7:  out << " Always   "; break;
    default: out << " Unknown  "; break;
    }
    out << msg << std::endl;
  }
  else{
  **/
  // Default CORAL reporter
  switch (lvl) {
    case coral::Nil:
    case coral::Verbose:
    case coral::Debug:
      m_logger.logDebug() << "CORAL: " << msg;
      break;
    case coral::Info:
      m_logger.logInfo() << "CORAL: " << msg;
      break;
    case coral::Warning:
      m_logger.logWarning() << "CORAL: " << msg;
      break;
    case coral::Error:
      m_logger.logError() << "CORAL: " << msg;
      break;
  }
}
