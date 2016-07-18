#include "CondCore/PopCon/interface/PopCon.h"
#include "CondCore/PopCon/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include <iostream>

namespace popcon {

  PopCon::PopCon(const edm::ParameterSet& pset):
    m_targetSession(),
    m_targetConnectionString(pset.getUntrackedParameter< std::string >("targetDBConnectionString","")),
    m_authPath( pset.getUntrackedParameter<std::string>("authenticationPath","")),
    m_authSys( pset.getUntrackedParameter<int>("authenticationSystem",1)),
    m_record(pset.getParameter<std::string> ("record")),
    m_payload_name(pset.getUntrackedParameter<std::string> ("name","")),
    m_LoggingOn(pset.getUntrackedParameter< bool > ("loggingOn",true)),
    m_close(pset.getUntrackedParameter< bool > ("closeIOV",false)),
    m_lastTill(pset.getUntrackedParameter< bool > ("lastTill",0))
    {
      //TODO set the policy (cfg or global configuration?)
      //Policy if corrupted data found
      
      edm::LogInfo ("PopCon") << "This is PopCon (Populator of Condition) v" << s_version << ".\n"
                              << "Please report any problem and feature request through the JIRA project CMSCONDDB.\n" ; 
    }
  
  PopCon::~PopCon(){
    if( !m_targetConnectionString.empty() )  {
      m_targetSession.transaction().commit();
    }
  }
 

  cond::persistency::Session PopCon::initialize() {	
    edm::LogInfo ("PopCon")<<"payload name "<<m_payload_name<<std::endl;
    if(!m_dbService.isAvailable() ) throw Exception("DBService not available");
    const std::string & connectionStr = m_dbService->session().connectionString();
    m_tag = m_dbService->tag(m_record);
    m_tagInfo.name = m_tag;
    if( m_targetConnectionString.empty() ) m_targetSession = m_dbService->session();
    else {
      cond::persistency::ConnectionPool connPool;
      connPool.setAuthenticationPath( m_authPath );
      connPool.setAuthenticationSystem( m_authSys );
      connPool.configure();
      m_targetSession = connPool.createSession( m_targetConnectionString );
      m_targetSession.transaction().start();
    }
    if( m_targetSession.existsIov( m_tag ) ){
      cond::persistency::IOVProxy iov = m_targetSession.readIov( m_tag );
      m_tagInfo.name = m_tag;
      m_tagInfo.size = iov.sequenceSize();
      if( m_tagInfo.size>0 ){
        cond::Iov_t last = iov.getLast();
        m_tagInfo.lastInterval = cond::ValidityInterval( last.since, last.till );
        m_tagInfo.lastPayloadToken = last.payloadId;
      }

      edm::LogInfo ("PopCon") << "destination DB: " << connectionStr
                              << ", target DB: " << ( m_targetConnectionString.empty() ? connectionStr : m_targetConnectionString ) << "\n"
                              << "TAG: " << m_tag
                              << ", last since/till: " <<  m_tagInfo.lastInterval.first
                              << "/" << m_tagInfo.lastInterval.second
                              << ", size: " << m_tagInfo.size << "\n" << std::endl;
    } else {
      edm::LogInfo ("PopCon") << "destination DB: " << connectionStr
                              << ", target DB: " << ( m_targetConnectionString.empty() ? connectionStr : m_targetConnectionString ) << "\n"
                              << "TAG: " << m_tag
                              << "; First writer to this new tag." << std::endl;
    }
    return m_targetSession;
  }
  
  
  void PopCon::finalize(Time_t lastTill) {
    
    if (m_close) {
      // avoid to close it before lastSince
      if (m_lastTill>lastTill) lastTill=m_lastTill;
      m_dbService->closeIOV(lastTill,m_record);
    }
    if( !m_targetConnectionString.empty() )  {
      m_targetSession.transaction().commit();
    }
  }
  
}
