#include "CondCore/PopCon/interface/PopCon.h"
#include "CondCore/PopCon/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>

namespace popcon {

  PopCon::PopCon(const edm::ParameterSet& pset):
    m_record(pset.getParameter<std::string> ("record")),
    m_payload_name(pset.getUntrackedParameter<std::string> ("name","")),
    m_LoggingOn(pset.getUntrackedParameter< bool > ("loggingOn",true)),
    m_IsDestDbCheckedInQueryLog(pset.getUntrackedParameter< bool > ("IsDestDbCheckedInQueryLog",true)),
    m_close(pset.getUntrackedParameter< bool > ("closeIOV",false)),
    m_lastTill(pset.getUntrackedParameter< bool > ("lastTill",0))
    {
      //TODO set the policy (cfg or global configuration?)
      //Policy if corrupted data found
      
      edm::LogInfo ("PopCon") << "This is PopCon (Populator of Condition) V4.0\n"
			      << "Please report any problem and feature request through the savannah portal under the category conditions\n" ; 

    }
  
  PopCon::~PopCon(){}
 

  void PopCon::initialize() {	
    edm::LogInfo ("PopCon")<<"payload name "<<m_payload_name<<std::endl;
    if(!m_dbService.isAvailable() ) throw Exception("DBService not available");
    const std::string & connectionStr = m_dbService->session().connectionString();
    m_tag = m_dbService->tag(m_record);
    m_tagInfo.name = m_tag;
    if( !m_dbService->isNewTagRequest(m_record) ) {
      m_dbService->tagInfo(m_record,m_tagInfo);
      /**
      // m_dbService->queryLog().LookupLastEntryByTag(m_tag, m_logDBEntry);
      if(m_IsDestDbCheckedInQueryLog) { 
	m_dbService->queryLog().LookupLastEntryByTag(m_tag, connectionStr , m_logDBEntry);
	std::cout << " ------ log info searched in the same db: " << connectionStr << "------" <<std::endl;
      } else {
	m_dbService->queryLog().LookupLastEntryByTag(m_tag , m_logDBEntry);
	std::cout << " ------ log info found in another db "  << "------" <<std::endl;
      }
      **/

      edm::LogInfo ("PopCon") << "DB: " << connectionStr << "\n"
			      << "TAG: " << m_tag 
			      << ", last since/till: " <<  m_tagInfo.lastInterval.first
			      << "/" << m_tagInfo.lastInterval.second
			      << ", , size: " << m_tagInfo.size << "\n" ;
	//<< "Last writer: " <<  m_logDBEntry.provenance 
	//	      << ", size: " << m_logDBEntry.payloadIdx+1 << std::endl;
    } else {
      edm::LogInfo ("PopCon") << "DB: " << connectionStr << "\n"
			      << "TAG: " << m_tag 
			      << "; First writer to this new tag." << std::endl; 
    }
  }
  
  
  void PopCon::finalize(Time_t lastTill) {
    
    if (m_close) {
      // avoid to close it before lastSince
      if (m_lastTill>lastTill) lastTill=m_lastTill;
      m_dbService->closeIOV(lastTill,m_record);
    }
  }
  
}
