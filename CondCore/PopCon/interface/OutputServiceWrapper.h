#ifndef OUTPUT_HELPER_H
#define OUTPUT_HELPER_H


#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/PopCon/interface/Exception.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <algorithm>
#include <vector>
#include <string>

namespace popcon
{

  struct TimeLess
  {
    template <typename T1>
    bool operator()(const std::pair<T1,cond::Time_t>& x, const std::pair<T1,cond::Time_t>& y)
    {
    return (x.second < y.second);
    }
  };

  class OutputServiceWrapper {
  private:
    edm::Service<cond::service::PoolDBOutputService> & m_dbService;
    std::string  m_record;
    bool m_since;
    bool m_LoggingOn;
    std::string logMsg;
    
  public:
    OutputServiceWrapper(edm::Service<cond::service::PoolDBOutputService> & dbService,
			 std::string const & record, bool since, bool log) :
      m_dbService(dbService),
      m_record(record), m_since(since), m_LoggingOn(log){}
    
    std::string tag() const {
      return m_dbService->tag(m_record);
    }
    
    void setLogHeader(std::string const & sourceId, std::string const & comment) {
      m_dbService->setLogHeaderForRecord(m_record,sourceId,comment);
    }

  public:
    template <typename T>
    void write (std::vector<std::pair<T*,popcon::IOVPair> > &  payload_vect, Time_t lsc){
      
      typename std::vector<std::pair<T*,popcon::IOVPair> >::iterator it;
	//sort ascending so the since/till order is respected 
      std::sort(payload_vect.begin(), payload_vect.end(),TimeLess());
      
      //check if attempting to insert an object with lower since-time than the last existing IOV
      it = payload_vect.begin();
      ///try{
      if ((*it).second < lsc) {
	throw popcon::Exception("IOV sequence Exception");
      }
      
      
      if(m_dbService.isAvailable() ){
	std::cerr << "DBOutputService configured with the following Tag " << tag() << std::endl;
	std::cerr << "... and Record " << m_record << std::endl;
	
	for (it = payload_vect.begin(); it != payload_vect.end(); it++){
	  try{
	    if (m_dbService->isNewTagRequest(m_record) ){
	      std::cerr << "Creating new IOV " << (*it).second << std::endl;
	      m_dbService->createNewIOV<T>((*it).first, m_since ? (*it).second : Time_t(0), m_record, m_LoggingOn);
	    }
	    else{
	      if (m_since){
		std::cerr << "Appending since time " <<  (*it).second << std::endl; 
		m_dbService->appendSinceTime<T>((*it).first, (*it).second, m_record, m_LoggingOn);
	      } 
	      else {
		std::cerr << "Appending till time "  << (*it).second << std::endl; 
		m_dbService->appendTillTime<T>((*it).first, (*it).second, m_record, m_LoggingOn);
	      }
	    }
	  }catch(std::exception& er){
	    std::cerr << "DB output exception: " << er.what();
	    std::string os("Problem with output service ");
	    os+= payload_vect.size();
	    os+=" objects should have been written";
	    logMsg = os; 
	    throw er;
	  }
	}
      }else{
	logMsg = "DBService unavailable";
      }
    }
    
  };

}
#endif




