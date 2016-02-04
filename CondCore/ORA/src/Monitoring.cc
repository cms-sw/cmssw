#include "CondCore/ORA/interface/Monitoring.h"
// externals
#include "uuid/uuid.h"
//
#include <string.h>
#include <cstdlib>
#include <fstream>

namespace ora {
  static char const* fmt_Guid = 
    "%08lX-%04hX-%04hX-%02hhX%02hhX-%02hhX%02hhX%02hhX%02hhX%02hhX%02hhX";  

  std::string guidFromTime(){
    uuid_t me_;
    ::uuid_generate_time(me_);
    unsigned int*  tmp = reinterpret_cast<unsigned int*>(me_);
    unsigned int d1 = *tmp;
    unsigned short d2 = *reinterpret_cast<unsigned short*>(me_+4);
    unsigned short d3 = *reinterpret_cast<unsigned short*>(me_+6);
    unsigned char  d4[8];
    for (int i = 0; i < 8; ++i){
      d4[i]=me_[i+8];
    }
    
    char text[128];
    ::sprintf(text, fmt_Guid,
	      d1, d2, d3, 
              d4[0], d4[1], d4[2], d4[3], 
	      d4[4], d4[5], d4[6], d4[7]);
    return std::string(text);
  }
}

ora::TransactionMonitoringData::TransactionMonitoringData( boost::posix_time::ptime start ):
  m_start(start),
  m_stop(boost::posix_time::neg_infin),
  m_status(false){
}

void ora::TransactionMonitoringData::stop(bool commit_status){
  m_stop = boost::posix_time::microsec_clock::universal_time();
  m_status = commit_status;
}

ora::SessionMonitoringData::SessionMonitoringData( boost::posix_time::ptime start, const std::string& connectionString ):
  m_connectionString( connectionString ),
  m_start(start),
  m_stop(boost::posix_time::neg_infin),
  m_currentTransaction(0),
  m_transactions(),
  m_ncur(0){
}

ora::SessionMonitoringData::~SessionMonitoringData(){
  for(std::map<std::string,TransactionMonitoringData*>::const_iterator iT = m_transactions.begin();
      iT != m_transactions.end();iT++ ){
    delete iT->second;
  }
}

size_t ora::SessionMonitoringData::newTransaction(){
  m_currentTransaction = new TransactionMonitoringData( boost::posix_time::microsec_clock::universal_time() );
  m_transactions.insert(std::make_pair(guidFromTime(),m_currentTransaction));
  m_ncur = m_transactions.size();
  return m_ncur;
}

size_t ora::SessionMonitoringData::stopTransaction( bool commit_status ){
  size_t ncur = 0;
  if(m_currentTransaction){
    m_currentTransaction->stop( commit_status );
    m_currentTransaction = 0;
    ncur = m_ncur;
    m_ncur = 0;
  }
  return ncur;
}

void ora::SessionMonitoringData::stop(){
  m_stop = boost::posix_time::microsec_clock::universal_time();
  m_currentTransaction = 0;
  m_ncur = 0;
}

size_t ora::SessionMonitoringData::numberOfTransactions() const{
  return m_transactions.size();
}

void ora::SessionMonitoringData::report( std::ostream& out ) const {
  size_t i=1;
  for(std::map<std::string,TransactionMonitoringData*>::const_iterator iT = m_transactions.begin();
      iT != m_transactions.end();iT++ ){
    TransactionMonitoringData& data = *iT->second;
    boost::posix_time::time_duration duration;
    if( !data.m_stop.is_neg_infinity() ){
      duration = data.m_stop-data.m_start;
    }
    out <<"   -> Transaction #"<<i<<" duration="<<boost::posix_time::to_simple_string(duration)<<" status="<<(data.m_status?std::string("COMMIT"):std::string("ROLLBACK"))<<std::endl;
    i++;
  }  
}

bool ora::Monitoring::s_enabled = false;

ora::Monitoring& ora::Monitoring::get(){
  static ora::Monitoring s_mon;
  return s_mon;
}

bool ora::Monitoring::isEnabled(){
  if(! s_enabled ){
    const char* envVar = ::getenv( "ORA_MONITORING_LEVEL" );
    if( envVar && ::strcmp(envVar,"SESSION")==0 ) s_enabled = true;
  }
  return s_enabled;
}

void ora::Monitoring::enable(){
  s_enabled = true;
}

std::string& ora::Monitoring::outFileName(){
  static std::string s_outFileName("");
  if( s_outFileName.empty() ){
    const char* fileEnvVar = ::getenv( "ORA_MONITORING_FILE" );
    if( fileEnvVar ){
      s_outFileName = fileEnvVar;
    }
  }
  return s_outFileName;
}

ora::Monitoring::~Monitoring() throw(){
  if( isEnabled() ){
    try {
      if( !outFileName().empty() ){
	std::ofstream outFile;
	outFile.open(  outFileName().c_str() );
	if(outFile.good()){
	  report( outFile );
	  outFile.flush();
	}
        outFile.close();
	
      } else {
	report( std::cout );
      }
    } catch ( const std::exception& e ){
      std::cout <<"ORA_MONITORING Error: "<<e.what()<<std::endl;
    }
  }
  // clean up memory
  for(std::map<std::string,SessionMonitoringData*>::const_iterator iS = m_sessions.begin();
      iS != m_sessions.end();iS++ ){
    delete iS->second;
  }
}

ora::SessionMonitoringData* ora::Monitoring::startSession( const std::string& connectionString ){
  ora::SessionMonitoringData* ret = new SessionMonitoringData( boost::posix_time::microsec_clock::universal_time(), connectionString );
  m_sessions.insert(std::make_pair(guidFromTime(),ret));
  return ret;
}

void ora::Monitoring::report( std::ostream& out ){
  out << "### ---------------------------------------------------------------------- "<<std::endl;
  out << "### ORA Monitoring Summary "<<std::endl;
  out << "### "<<m_sessions.size()<<" session(s) registered."<<std::endl;
  size_t j = 1;
  for( std::map<std::string,SessionMonitoringData*>::const_iterator iS = m_sessions.begin();
       iS != m_sessions.end(); ++iS ){
    SessionMonitoringData& data = *iS->second;
    boost::posix_time::time_duration duration;
    if( !data.m_stop.is_neg_infinity() ){
      duration = data.m_stop-data.m_start;
    }
    out <<" -> Session #"<<j<<": connection=\""<<data.m_connectionString<<"\" duration="<<boost::posix_time::to_simple_string(duration)<<" transactions="<<(iS->second)->numberOfTransactions()<<std::endl;
    (iS->second)->report(out);
    j++;
  }
  out << "### ---------------------------------------------------------------------- "<<std::endl;

}
 
ora::Monitoring::Monitoring():
  m_sessions(){
}
