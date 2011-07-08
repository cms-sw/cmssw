#include "SQLMonitoringService.h"
#include "RelationalAccess/MonitoringException.h"
#include "CoralBase/MessageStream.h"
#include "CoralBase/TimeStamp.h"
#include "CoralKernel/Context.h"
#include "CoralKernel/IHandle.h"
#include "CondCore/DBCommon/interface/CoralServiceMacros.h"

namespace cond
{
  SessionMonitor::SessionMonitor()
    : active( false ), level( coral::monitor::Default ), stream()
  {
  }
  
  SessionMonitor::SessionMonitor( bool act, coral::monitor::Level lvl )
    : active( act ), level( lvl ), stream()
  {
  }

  SQLMonitoringService::SQLMonitoringService( const std::string& key )
    : coral::Service( key ),
      m_events()
  {
  }
  
  SQLMonitoringService::~SQLMonitoringService()
  {
  }
  
  void SQLMonitoringService::setLevel( const std::string& contextKey, coral::monitor::Level level )
  {
    Repository::const_iterator rit;
    
    if( ( rit = m_events.find( contextKey ) ) == m_events.end() )
      {        
        m_events[contextKey] = SessionMonitor();
        m_monitoredDS.insert( contextKey );
      }
    
    m_events[contextKey].level = level;
    
    if( level == coral::monitor::Off )
      {
        m_events[contextKey].active = false;
      }
    else
      {
        m_events[contextKey].active = true;        
      }
    }
  
  coral::monitor::Level SQLMonitoringService::level( const std::string& contextKey ) const
  {
    Repository::const_iterator rit;
    
    if( ( rit = m_events.find( contextKey ) ) == m_events.end() )
      throw coral::MonitoringException( "Monitoring for session " + contextKey + " not initialized...", "MonitoringService::level", this->name() );
    
    return (*rit).second.level;
  }
  
  bool SQLMonitoringService::active( const std::string& contextKey ) const
  {
    Repository::const_iterator rit;
    
    if( ( rit = m_events.find( contextKey ) ) == m_events.end() )
      throw coral::MonitoringException( "Monitoring for session " + contextKey + " not initialized...", "MonitoringService::active", this->name() );
    
    return (*rit).second.active;
  }
  
  void SQLMonitoringService::enable( const std::string& contextKey )
    {
      Repository::iterator rit;
      
      if( ( rit = m_events.find( contextKey ) ) == m_events.end() )
        throw coral::MonitoringException( "Monitoring for session " + contextKey + " not initialized...", "MonitoringService::enable", this->name() );

      (*rit).second.active = true;
    }

  void SQLMonitoringService::disable( const std::string& contextKey )
    {
      Repository::iterator rit;
      
      if( ( rit = m_events.find( contextKey ) ) == m_events.end() )
        throw coral::MonitoringException( "Monitoring for session " + contextKey + " not initialized...", "MonitoringService::disable", this->name() );
      
      (*rit).second.active = false;
    }
  
  //relaxing filter on coral monitoring level due to a bug in the connection pool
  void SQLMonitoringService::record( const std::string& contextKey, coral::monitor::Source source, coral::monitor::Type type, const std::string& description )
    {
      Repository::iterator rit;
      
      if( ( rit = m_events.find( contextKey ) ) == m_events.end() )
        throw coral::MonitoringException( "Monitoring for session " + contextKey + " not initialized...", "MonitoringService::record( const std::string& , coral::monitor::Source, coral::monitor::Type, const std::string& )", this->name() );
      
      bool                  active = (*rit).second.active;
      //coral::monitor::Level level  = (*rit).second.level;
       
      if( active/* && (type & level)*/ )
      {
	(*rit).second.stream.push_back( coral::monitor::createEvent( source, type, description ) );
      }
    }

  void SQLMonitoringService::record( const std::string& contextKey, coral::monitor::Source source, coral::monitor::Type type, const std::string& description, int data )
  {
    Repository::iterator rit;

    if( ( rit = m_events.find( contextKey ) ) == m_events.end() )
      throw coral::MonitoringException( "Monitoring for session " + contextKey + " not initialized...", "MonitoringService::record( const std::string& , coral::monitor::Source, coral::monitor::Type, const std::string&, int )", this->name() );
    
    bool                  active = (*rit).second.active;
    //coral::monitor::Level level  = (*rit).second.level;
    
    if( active/* && (type & level)*/ )
      {
	(*rit).second.stream.push_back( coral::monitor::createEvent( source, type, description, data ) );
      }
  }

  void SQLMonitoringService::record( const std::string& contextKey, coral::monitor::Source source, coral::monitor::Type type, const std::string& description, long long data )
  {
    Repository::iterator rit;

    if( ( rit = m_events.find( contextKey ) ) == m_events.end() )
      throw coral::MonitoringException( "Monitoring for session " + contextKey + " not initialized...", "MonitoringService::record( const std::string& , coral::monitor::Source, coral::monitor::Type, const std::string&, long long  )", this->name() );
    
    bool                  active = (*rit).second.active;
    //coral::monitor::Level level  = (*rit).second.level;
    
    if( active/* && (type & level)*/ )
      {
	(*rit).second.stream.push_back( coral::monitor::createEvent( source, type, description, data ) );
      }
  }

  void SQLMonitoringService::record( const std::string& contextKey, coral::monitor::Source source, coral::monitor::Type type, const std::string& description, double data )
    {
      Repository::iterator rit;

      if( ( rit = m_events.find( contextKey ) ) == m_events.end() )
        throw coral::MonitoringException( "Monitoring for session " + contextKey + " not initialized...", "MonitoringService::record( const std::string& , coral::monitor::Source, coral::monitor::Type, const std::string&, double )", this->name() );

      bool                  active = (*rit).second.active;
      //coral::monitor::Level level  = (*rit).second.level;
      
      if( active/* && (type & level)*/ )
      {
	(*rit).second.stream.push_back( coral::monitor::createEvent( source, type, description, data ) );
      }
    }

  void SQLMonitoringService::record( const std::string& contextKey, coral::monitor::Source source, coral::monitor::Type type, const std::string& description, const std::string& data )
    {
      Repository::iterator rit;

      if( ( rit = m_events.find( contextKey ) ) == m_events.end() )
        throw coral::MonitoringException( "Monitoring for session " + contextKey + " not initialized...", "MonitoringService::record( const std::string& , coral::monitor::Source, coral::monitor::Type, const std::string&, const std::string& )", this->name() );

      bool                  active = (*rit).second.active;
      //coral::monitor::Level level  = (*rit).second.level;
      
      if( active /*&& (type & level)*/ )
      {
	(*rit).second.stream.push_back( coral::monitor::createEvent( source, type, description, data ) );
      }
    }

  const coral::IMonitoringReporter& SQLMonitoringService::reporter() const
    {
      return( static_cast<const coral::IMonitoringReporter&>(*this) );
    }
    
  // The coral::IMonitoringReporter interface implementation
  std::set< std::string > SQLMonitoringService::monitoredDataSources() const
    {
      return m_monitoredDS;
    }

  void SQLMonitoringService::report( unsigned int /*level*/ ) const
    {
      Repository::const_iterator rit;
      coral::MessageStream  log( "MonitoringService" );
      
      // Dummy reporting so far
      for( rit = m_events.begin(); rit != m_events.end(); ++rit )
        reportForSession( rit, log );
    }

  void SQLMonitoringService::report( const std::string& contextKey, unsigned int /* level */ ) const
    {
      Repository::const_iterator rit;
      
      if( ( rit = m_events.find( contextKey ) ) == m_events.end() )
        throw coral::MonitoringException( "Monitoring for session " + contextKey + " not initialized...", "MonitoringService::record", this->name() );

      // Dummy reporting so far
      coral::MessageStream log( "MonitoringService" );

      reportForSession( rit, log );
    }

  void SQLMonitoringService::reportToOutputStream( const std::string& contextKey, std::ostream& os, unsigned int /* level */ ) const
    {
      Repository::const_iterator rit;
      
      if( ( rit = m_events.find( contextKey ) ) == m_events.end() )
        throw coral::MonitoringException( "Monitoring for session " + contextKey + " not initialized...", "MonitoringService::record", this->name() );

      // Dummy reporting so far
      coral::MessageStream log( "MonitoringService" );

      reportForSession( rit, os );
    }

  void SQLMonitoringService::reportOnEvent( EventStream::const_iterator& it, std::ostream& os ) const
  {
    std::string source("");
    switch( it->m_source ) {
    case coral::monitor::Application:
      source = "Application";
      break;
    case coral::monitor::Session:
      source = "Session";
      break;
    case coral::monitor::Transaction:
      source = "Transaction";
      break;
    case coral::monitor::Statement:
      source = "Statement";
      break;
    default:
      source = "";
    };
    
    std::string type("");
    switch( it->m_type ) {
    case coral::monitor::Info:
      type = "Info";
      break;
    case coral::monitor::Time:
      type = "Time";
      break;
    case coral::monitor::Warning:
      type = "Warning";
      break;
    case coral::monitor::Error:
      type = "Error";
      break;
    case coral::monitor::Config:
      type = "Config";
      break;
    default:
      type = "";
    };
    
    if(it->m_source == coral::monitor::Statement || it->m_source == coral::monitor::Transaction)
      {
	os << boost::posix_time::to_iso_extended_string((*it).m_time.time()) << ": " 
	   << source << "; "
	   << type << "; " 
	   <<(*it).m_description << ";"<< std::endl;
      }
  }
  

  void SQLMonitoringService::reportOnEvent( EventStream::const_iterator& it,coral::MessageStream& os ) const
    {
      std::string source("");
      switch( it->m_source ) {
      case coral::monitor::Application:
	source = "Application";
	break;
      case coral::monitor::Session:
	source = "Session";
	break;
      case coral::monitor::Transaction:
	source = "Transaction";
	break;
      case coral::monitor::Statement:
	source = "Statement";
	break;
      default:
	source = "";
      };
      
      std::string type("");
      switch( it->m_type ) {
      case coral::monitor::Info:
	type = "Info";
	break;
      case coral::monitor::Time:
	type = "Time";
	break;
      case coral::monitor::Warning:
	type = "Warning";
	break;
      case coral::monitor::Error:
	type = "Error";
	break;
      case coral::monitor::Config:
	type = "Config";
	break;
      default:
	type = "";
      };
      
      if(it->m_source == coral::monitor::Statement || it->m_source == coral::monitor::Transaction)
	{
	  os << boost::posix_time::to_iso_extended_string((*it).m_time.time()) << ": " 
	     << source << "; "
	     << type << "; " 
	     << (*it).m_description <<coral::MessageStream::flush;
	}
    }

  void SQLMonitoringService::reportForSession( Repository::const_iterator& it, std::ostream& os ) const
    {
      os << "Session: " << (*it).first << std::endl;
      std::string lvl;
      switch( (*it).second.level ) {
      case (coral::monitor::Off)     : lvl = "Off"; break;
      case (coral::monitor::Minimal) : lvl = "Minimal"; break;
      case (coral::monitor::Default) : lvl = "Default"; break;
      case (coral::monitor::Debug)   : lvl = "Debug"; break;
      case (coral::monitor::Trace)   : lvl = "Trace"; break;
      default: lvl = "";
      };
      os << "Monitoring Level: " << lvl << std::endl;
      
      const EventStream& evsref = (*it).second.stream;
      os << " Recorded " << evsref.size() << " events" << std::endl;
      
      for( EventStream::const_iterator evit = evsref.begin(); evit != evsref.end(); ++evit )
      {
        reportOnEvent( evit, os );
      }
    }

  void SQLMonitoringService::reportForSession( Repository::const_iterator& it, coral::MessageStream& os ) const
    {
      os << "Session: " << (*it).first;
      std::string lvl;
      switch( (*it).second.level ) {
      case (coral::monitor::Off)     : lvl = "Off"; break;
      case (coral::monitor::Minimal) : lvl = "Minimal"; break;
      case (coral::monitor::Default) : lvl = "Default"; break;
      case (coral::monitor::Debug)   : lvl = "Debug"; break;
      case (coral::monitor::Trace)   : lvl = "Trace"; break;
      default: lvl = "";
      };
      os << " monitored at level: " << lvl;
      
      const EventStream& evsref = (*it).second.stream;
      os << lvl << " has recorded " << evsref.size() << " events" << coral::MessageStream::endmsg;
      
      for( EventStream::const_iterator evit = evsref.begin(); evit != evsref.end(); ++evit )
      {
        reportOnEvent( evit, os );
      }

    }
    
} // namespace cond

DEFINE_CORALSERVICE(cond::SQLMonitoringService,"COND/Services/SQLMonitoringService");
