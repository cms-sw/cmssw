
#ifndef POPCON_POPCON_H
#define POPCON_POPCON_H
//
// Author: Vincenzo Innocente
// Original Author:  Marcin BOGUSZ
// 


#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondCore/DBCommon/interface/TagInfo.h"
#include "CondCore/DBCommon/interface/LogDBEntry.h"
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
    
    bool m_IsDestDbCheckedInQueryLog;

    std::string m_tag;
    
    cond::TagInfo m_tagInfo;
    
    cond::LogDBEntry m_logDBEntry;
  
    
    
    
    
  };


  template<typename T>
  void PopCon::writeOne(T * payload, Time_t time) {
    m_dbService->writeOne(payload, time, m_record, m_LoggingOn, m_since);
  }

  template<typename Container>
  void displayHelper(Container const & payloads, bool sinceAppend) {
    typename Container::const_iterator it;
        for (it = payloads.begin(); it != payloads.end(); it++){
      edm::LogInfo ("PopCon")<< (sinceAppend ? "Since " :" Till ") << (*it).second << std::endl;
     }
  }     
  
 
  template<typename Container>
    const std::string displayIovHelper(Container const & payloads, bool sinceAppend) {
    typename Container::const_iterator it;
    size_t i =0, j=0;
    std::ostringstream s; std::ostringstream ss; 
    // when only 1 payload is transferred; 
    for (it = payloads.begin(); it != payloads.end(); it++){
      i++; 
      if (i ==1)   s <<(sinceAppend ? "Since " :" Till ") << (*it).second <<  "; " ;
    }
    // when more than one payload are transferred;  
    if (i>1) {
      ss << "\ntransferred " << i << " payloads:\n"  ;
      for (it = payloads.begin(); it != payloads.end(); it++){
	j++;
	if (j==1) ss <<   "first payload " << (sinceAppend ? "Since " :" Till ") << (*it).second <<  ";\n" ;
	if (j==i) ss<< "last payload " << (sinceAppend ? "Since " :" Till ") << (*it).second <<  ";\n" ;  
      }  
    } 
    return ( (i ==1)  ?  s.str(): ss.str() ) ; 
  }            
  
 
 
  
    template<typename Source>
      void PopCon::write(Source const & source) {
      typedef typename Source::value_type value_type;
      typedef typename Source::Container Container;
      
      initialize();
      std::pair<Container const *, std::string const> ret = source(&m_dbService->connection(),
								   m_tagInfo,m_logDBEntry); 
     Container const & payloads = *ret.first;
     

  // adding info about the popcon user
    std::ostringstream user_info;
    char * user= ::getenv("USER");
    char * hostname= ::getenv("HOSTNAME");
    char * pwd = ::getenv("PWD");
    if (user) { user_info<< "\nUSER = " << user <<";" ;} else { user_info<< "\n USER = "<< "??;";}
    if (hostname) {user_info<< "\nHOSTNAME= " << hostname <<";";} else { user_info<< "\n HOSTNAME = "<< "??;";}
    if (pwd) {user_info<< "\nPWD= " << pwd <<";";} else  {user_info<< "\n PWD = "<< "??;";}
    
       

     m_dbService->setLogHeaderForRecord(m_record,source.id(),"PopCon v2.1; " + user_info.str() + displayIovHelper(payloads,m_since ) +  ret.second);
     
     
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


