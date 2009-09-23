#ifndef CondCore_IOVService_IOVServiceImpl_h
#define CondCore_IOVService_IOVServiceImpl_h
#include <string>
#include <map>
#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/Common/interface/IOVSequence.h"
#include "CondCore/DBCommon/interface/DbSession.h"

namespace cond{
  class IOVIterator;
  class IOVEditor;
  class IOVServiceImpl{
  public:
    IOVServiceImpl(cond::DbSession& pooldb);
    
    ~IOVServiceImpl();
    
    std::string payloadToken( const std::string& iovToken,
			      cond::Time_t currenttime );
    
    bool isValid( const std::string& iovToken,
		  cond::Time_t currenttime );

    std::pair<cond::Time_t, cond::Time_t> 
    validity( const std::string& iovToken, cond::Time_t currenttime );
    
    std::string payloadContainerName( const std::string& iovtoken );
    
    void loadDicts( const std::string& iovToken);

    void deleteAll(bool withPayload);
    
    cond::TimeType timeType() const;
    cond::Time_t globalSince() const;
    cond::Time_t globalTill() const;

    std::string exportIOVWithPayload( cond::DbSession& destDB,
                                      const std::string& iovToken );
    
    
    std::string exportIOVRangeWithPayload( cond::DbSession& destDB,
                                           const std::string& iovToken,
                                           const std::string& destToken,
                                           cond::Time_t since,
                                           cond::Time_t till,
                                           bool outOfOrder);
  private:

    cond::IOVSequence const & iovSeq(const std::string& iovToken) const;

    mutable cond::DbSession m_pooldb;
    typedef std::map< std::string,  pool::Ref<cond::IOVSequence> > Cache;
    mutable Cache m_iovcache;
  };

}//ns cond
#endif
