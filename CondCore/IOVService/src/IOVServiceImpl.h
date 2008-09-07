#ifndef CondCore_IOVService_IOVServiceImpl_h
#define CondCore_IOVService_IOVServiceImpl_h
#include <string>
#include <map>
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "IOV.h"
namespace cond{
  class PoolStorageManager;
  class IOVIterator;
  class IOVEditor;
  class IOVServiceImpl{
  public:
    IOVServiceImpl(cond::PoolTransaction& pooldb, 
		   cond::TimeType timetype);
    ~IOVServiceImpl();
    std::string payloadToken( const std::string& iovToken,
			      cond::Time_t currenttime );
    bool isValid( const std::string& iovToken,
		  cond::Time_t currenttime );
    std::pair<cond::Time_t, cond::Time_t> 
      validity( const std::string& iovToken, cond::Time_t currenttime );
    std::string payloadContainerName( const std::string& iovtoken );
 
   void deleteAll(bool withPayload);

    cond::TimeType timeType() const;
    cond::Time_t globalSince() const;
    cond::Time_t globalTill() const;

    std::string exportIOVWithPayload( cond::PoolTransaction& destDB,
			       const std::string& iovToken );


    std::string exportIOVRangeWithPayload( cond::PoolTransaction& destDB,
					   const std::string& iovToken,
					   const std::string& destToken,
					   cond::Time_t since,
					   cond::Time_t till );
  private:
    cond::PoolTransaction* m_pooldb;
    cond::TimeType m_timetype;
    std::map< std::string,cond::TypedRef<cond::IOV> > m_iovcache;
    cond::Time_t m_beginOftime;
    cond::Time_t m_endOftime;
  };
}//ns cond
#endif
