#ifndef CondCore_IOVService_IOVEditorImpl_h
#define CondCore_IOVService_IOVEditorImpl_h
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/DBCommon/interface/Ref.h"
namespace cond{
  class PoolStorageManager;
  class IOV;
  class IOVEditorImpl : virtual public cond::IOVEditor{
  public:
    explicit IOVEditorImpl( cond::PoolStorageManager& pooldb,
			    const std::string& token,
			    cond::Time_t globalSince, 
			    cond::Time_t globalTill);
    virtual ~IOVEditorImpl();
    void insert( cond::Time_t tillTime,
		 const std::string& payloadToken
		 );
    void bulkInsert( std::vector< std::pair<cond::Time_t,std::string> >& values );
    virtual void updateClosure( cond::Time_t newtillTime );
    virtual void append(  cond::Time_t sinceTime,
			  const std::string& payloadToken
			  );
    virtual void deleteEntries( bool withPayload=false );
    virtual void import( const std::string& sourceIOVtoken );
    std::string token() const {
      return m_token;
    }
  private:
    void init();
    cond::PoolStorageManager& m_pooldb;
    std::string m_token;
    cond::Time_t m_globalSince;
    cond::Time_t m_globalTill;
    bool m_isActive;
    cond::Ref<cond::IOV> m_iov;
  };
}//ns cond
#endif
