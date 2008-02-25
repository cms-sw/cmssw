#ifndef POPCON_POPCON_H
#define POPCON_POPCON_H
//
// Author: Vincenzo Innocente
// Original Author:  Marcin BOGUSZ
// 


#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondCore/DBOutputService/interface/TagInfo.h"
#include "CondCore/DBOutputService/interface/LogDBEntry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "CondCore/DBCommon/interface/Time.h"


#include <boost/bind.hpp>
#include <algorithm>
#include <vector>
#include <string>


#include<iostream>




namespace popcon {


  /* Populator of the Condition DB
   *
   */
  class PopCon {
  public:
    typedef cond::Time_t Time_t;

     PopCon(const edm::ParameterSet& pset);
     
     virtual ~PopCon();

     template<typename Source>
       void write(Source const & source);

     template<typename T>
       void writeOne(T * payload, Time_t time);

  private:
     void initialize();
     void finalize();


  private:

    edm::Service<cond::service::PoolDBOutputService> m_dbService;
    
    std::string  m_record;
    
    std::string m_payload_name;
    
    bool m_since;
    
    bool m_LoggingOn;
    
    std::string m_tag;
    
    cond::TagInfo m_tagInfo;
    
    cond::LogDBEntry m_logDBEntry;
    
    
  };


  template<typename T>
  void PopCon::writeOne(T * payload, Time_t time) {
    if (m_dbService->isNewTagRequest(m_record) ){
      edm::LogInfo ("PopCon") << "Creating new IOV " << time << std::endl;
      m_dbService->createNewIOV<T>(payload, 
				   m_since ? time : m_dbService->beginOfTime(),
				   m_since ?  m_dbService->endOfTime() : time, 
				   m_record, m_LoggingOn);
    }
    else{
      if (m_since){
	edm::LogInfo ("PopCon") << "Appending since time " <<  time << std::endl; 
	m_dbService->appendSinceTime<T>(payload, time, m_record, m_LoggingOn);
      } 
      else {
	edm::LogInfo ("PopCon") << "Appending till time "  <<  time << std::endl; 
	m_dbService->appendTillTime<T>(payload, time, m_record, m_LoggingOn);
      }
    }
    
  }

  template<typename Container>
  void displayHelper(Container const & payloads, bool sinceAppend) {
    typename Container::const_iterator it;
    for (it = payloads.begin(); it != payloads.end(); it++){
      edm::LogInfo ("PopCon")<< (sinceAppend ? "Since " :" Till ") << (*it).second << std::endl;
    }
  }

    template<typename Source>
    void PopCon::write(Source const & source) {
      typedef typename Source::value_type value_type;
      typedef typename Source::Container Container;

      initialize();
      m_dbService->setLogHeaderForRecord(m_record,source.id(),"something clever");

      Container const & payloads = source(&m_dbService->connection(),
					  m_tagInfo,m_logDBEntry);
      displayHelper(payloads,m_since);
      std::for_each(payloads.begin(),payloads.end(),
		    boost::bind(&popcon::PopCon::writeOne<value_type>,this,
				boost::bind(&Container::value_type::first,_1),
				boost::bind(&Container::value_type::second,_1)
				)
		    );
      

      finalize();
    }

  

}

#endif //  POPCON_POPCON_H
