#ifndef CondCore_IOVService_IOVService_h
#define CondCore_IOVService_IOVService_h
#include <string>
#include "CondCore/DBCommon/interface/Time.h"
namespace cond{
  class PoolTransaction;
  class IOVServiceImpl;
  class IOVIterator;
  class IOVEditor;
  class IOVService{
  public:
    static const bool forwardIter=true;
    static const bool backwardIter=!forwardIter;


    IOVService( cond::PoolTransaction& pooldb,
		cond::TimeType timetype=cond::runnumber );

    virtual ~IOVService();

    std::string payloadToken( const std::string& iovToken,
			      cond::Time_t currenttime );

    bool isValid( const std::string& iovToken,
		  cond::Time_t currenttime );

    std::pair<cond::Time_t, cond::Time_t> 
      validity( const std::string& iovToken, cond::Time_t currenttime );

    std::string payloadContainerName( const std::string& iovtoken );

    void deleteAll( bool withPayload=false );

    /**
    create an iterator to on the iov selected by the token
    user aquires the ownership of the pointer. Need explicit delete after usage
    */
    IOVIterator* newIOVIterator( const std::string& iovToken,  bool forward=forwardIter);

    /**
    create an editor to the iov selected by the token
    user aquires the ownership of the pointer. Need explicit delete after usage
    */
    IOVEditor* newIOVEditor( const std::string& token="" );

    cond::TimeType timeType() const;
    cond::Time_t globalSince() const;
    cond::Time_t globalTill() const;

    /**
       export IOV selected by token and associated payload to another database
       return new iov token string 
    */

    std::string exportIOVWithPayload( cond::PoolTransaction& destDB,
				      const std::string& iovToken );
    /**
       export IOV selected by token within selected range and associated 
       payload to another database
       return new iov token string 
    */
    std::string exportIOVRangeWithPayload( cond::PoolTransaction& destDB,
					   const std::string& iovToken,
					   const std::string& destToken,
					   cond::Time_t since,
					   cond::Time_t till );
  private:
    cond::PoolTransaction* m_pooldb;
    cond::IOVServiceImpl* m_impl;
  };
}//ns cond
#endif
