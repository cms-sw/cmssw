#ifndef CondCore_IOVService_IOVService_h
#define CondCore_IOVService_IOVService_h
#include <string>
#include "CondCore/DBCommon/interface/Time.h"
namespace cond{
  class PoolStorageManager;
  class IOVServiceImpl;
  class IOVIterator;
  class IOVEditor;
  class IOVService{
  public:
    IOVService( cond::PoolStorageManager& pooldb,cond::TimeType timetype=cond::runnumber);
    virtual ~IOVService();
    std::string payloadToken( const std::string& iovToken,
			      cond::Time_t currenttime );
    bool isValid( const std::string& iovToken,
		  cond::Time_t currenttime );
    std::pair<cond::Time_t, cond::Time_t> 
      validity( const std::string& iovToken, cond::Time_t currenttime );
    std::string payloadContainerName( const std::string& iovtoken );
    void deleteAll();
    IOVIterator* newIOVIterator( const std::string& iovToken );
    IOVEditor* newIOVEditor( const std::string& token );
    IOVEditor* newIOVEditor();
    cond::TimeType timeType() const;
    cond::Time_t globalSince() const;
    cond::Time_t globalTill() const;
    void exportIOV( cond::PoolStorageManager& destDB,
		    const std::string& iovToken );
    void exportIOVRange( cond::PoolStorageManager& destDB,
			 const std::string& iovToken,
			 cond::Time_t lowValue,
			 cond::Time_t highValue);
    void exportIOVWithPayload( cond::PoolStorageManager& destDB,
			       const std::string& iovToken,
			       const std::string& payloadObjectName );
    void exportIOVRangeWithPayload( cond::PoolStorageManager& destDB,
				    const std::string& iovToken,
				    cond::Time_t lowValue,
				    cond::Time_t highValue,
				    const std::string& payloadObjectName );
  private:
    cond::PoolStorageManager& m_pooldb;
    cond::IOVServiceImpl* m_impl;
  };
}//ns cond
#endif
