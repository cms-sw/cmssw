#ifndef CondCore_IOVService_IOVEditorImpl_h
#define CondCore_IOVService_IOVEditorImpl_h
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/DBCommon/interface/TypedRef.h"

namespace cond{
  class PoolTransaction;
  class IOVSequence;

  class IOVEditorImpl : virtual public cond::IOVEditor{
  public:
    // constructor from existing iov
    IOVEditorImpl( cond::PoolTransaction& pooldb,
		   const std::string& token);
    /// Destructor
    virtual ~IOVEditorImpl();

    /// create a new IOV
    void create(cond::TimeType timetype,cond::Time_t lastTill);

    /// Assign a payload with till time. Returns the payload index in the iov sequence
    virtual unsigned int insert( cond::Time_t tillTime,
				 const std::string& payloadToken
				 );

    void bulkAppend( std::vector< std::pair<cond::Time_t,std::string> >& values );
    void bulkAppend(std::vector< cond::IOVElement >& values);

    virtual void updateClosure( cond::Time_t newtillTime );


    // stamp iov
    virtual void stamp(std::string const & icomment, bool append=false);


    /// Append a payload with known since time. The previous last payload's till time will be adjusted to the new payload since time. Returns the payload index in the iov sequence
    virtual unsigned int append(  cond::Time_t sinceTime,
				  const std::string& payloadToken
				  );
    
    /// insert a payload with known since in any position
    unsigned int 
    freeInsert(cond::Time_t sinceTime ,
               const std::string& payloadToken
	       );

    virtual void deleteEntries( bool withPayload=false );

    virtual void import( const std::string& sourceIOVtoken );

    std::string token() const {
      return m_token;
    }

    cond::Time_t firstSince() const;
    cond::Time_t lastTill() const;
    
    cond::TimeType timetype() const;

  private:
    void init();
    bool validTime(cond::Time_t time, cond::TimeType timetype) const;
    bool validTime(cond::Time_t time) const;

    cond::PoolTransaction* m_pooldb;
    std::string m_token;
    bool m_isActive;
    cond::TypedRef<cond::IOVSequence> m_iov;
  };
}//ns cond
#endif
