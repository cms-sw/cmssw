#ifndef CondCore_IOVService_IOVService_h
#define CondCore_IOVService_IOVService_h
#include <string>
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondFormats/Common/interface/IOVSequence.h"

namespace cond{

  class IOVEditor;
  class IOVService{
  public:

    IOVService(cond::DbSession& dbSess);
    
    virtual ~IOVService();

    std::string payloadToken( const std::string& iovToken,
                              cond::Time_t currenttime );

    bool isValid( const std::string& iovToken,
                  cond::Time_t currenttime );

    std::pair<cond::Time_t, cond::Time_t>
    validity( const std::string& iovToken, cond::Time_t currenttime );
    
    std::set<std::string> payloadClasses( const std::string& iovtoken );

    int iovSize( const std::string& iovtoken );

    cond::TimeType timeType( const std::string& iovToken );
    
    void deleteAll( bool withPayload=false );


    /**
    create an editor to the iov selected by the token
    user aquires the ownership of the pointer. Need explicit delete after usage
    */
    IOVEditor* newIOVEditor( const std::string& token="" );

   /**
       export IOV selected by token and associated payload to another database
       return new iov token string 
    */

    std::string exportIOVWithPayload( cond::DbSession& destDB,
                                      const std::string& iovToken );
    /**
       export IOV selected by token within selected range and associated 
       payload to another database
       return new iov token string 
    */
    std::string exportIOVRangeWithPayload( cond::DbSession& destDB,
                                           const std::string& iovToken,
                                           const std::string& destToken,
                                           cond::Time_t since,
                                           cond::Time_t till,
                                           bool outOfOrder
					   );
    
  private:
    cond::IOVSequence const & iovSeq(const std::string& iovToken);
    
    cond::DbSession m_dbSess;
    std::string m_token;
    boost::shared_ptr<cond::IOVSequence> m_iov;
  };

}//ns cond
#endif
