#include "CondCore/PopCon/interface/PopCon.h"
#include "CondCore/PopCon/interface/Exception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include<iostream>

namespace popcon {

  PopCon::PopCon(const edm::ParameterSet& pset):
    m_record(pset.getParameter<std::string> ("record")),
    m_payload_name(pset.getUntrackedParameter<std::string> ("name","")),
    m_since(pset.getParameter<bool> ("SinceAppendMode")),
    m_LoggingOn(pset.getUntrackedParameter< bool > ("loggingOn",true))
    {
    //TODO set the policy (cfg or global configuration?)
    //Policy if corrupted data found
    }
  
  PopCon::~PopCon(){}
 

  void PopCon::initialize() {	
    std::cerr<<"payload name "<<m_payload_name<<std::endl;
    if(!m_dbService.isAvailable() ) throw Exception("DBService not available");
    
    m_tag = m_dbService->tag(m_record);
    if (!m_dbService->isNewTagRequest(m_record) ) {
      m_dbService->tagInfo(m_record,m_tagInfo);
      m_dbService->queryLog().LookupLastEntryByTag(m_tag, m_logDBEntry);
      
      std::cerr << "TAG, last since/till, size " << m_tag 
		<< ", " <<  m_tagInfo.lastInterval.first
		<< "/" << m_tagInfo.lastInterval.second
		<< ", " << m_tagInfo.size << std::endl;
      std::cerr << "Last writer, size" <<  m_logDBEntry.provenance 
		<< ", " << m_logDBEntry.payloadIdx << std::endl;
    }
  }


  void PopCon::finalize() {
  }

}
