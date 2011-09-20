#ifndef CondCore_IOVService_IOVEditor_h
#define CondCore_IOVService_IOVEditor_h
#include <string>
#include <vector>
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondFormats/Common/interface/IOVSequence.h"
#include<iosfwd>

//
// Package:     CondCore/IOVService
// Class  :     IOVEditor
//
/**\class IOVEditor IOVEditor.h CondCore/IOVService/interface/IOVEditor.h
   Description: iov sequence manipulator
*/
//
// Author:      Zhen Xie
//

namespace cond{

  class IOVElement;
  class IOVSequence;

  class IOVEditor{
  public:

    // default constructor
    explicit IOVEditor(cond::DbSession& dbSess);

    // constructor from existing iov
    IOVEditor( cond::DbSession& dbSess, const std::string& token);
 
    /// Destructor
    ~IOVEditor();

    // create empty default sequence
    void create( cond::TimeType timetype );

    // create empty sequence with fixed time boundary
    void create(cond::TimeType timetype, cond::Time_t lastTill );

    // return the current sequence
    IOVSequence & iov();

    /// Assign a payload with till time. Returns the payload index in the iov sequence
    unsigned int insert( cond::Time_t tillTime, const std::string& payloadToken );

    /// Append a payload with known since time. The previous last payload's till time will be adjusted to the new payload since time. 
    /// Returns the payload index in the iov sequence
    unsigned int append(  cond::Time_t sinceTime, const std::string& payloadToken );

    /// insert a payload with known since in any position
    unsigned int freeInsert( cond::Time_t sinceTime, const std::string& payloadToken );

    /// Bulk append of iov chunck
    void bulkAppend( std::vector< std::pair<cond::Time_t, std::string > >& values );

    void bulkAppend(std::vector< cond::IOVElement >& values);
    
    //stamp iov
    void stamp(std::string const & icomment, bool append=false);

    /// edit metadata
    void editMetadata( std::string const & metadata, bool append=false);

    /// set the scope
    void setScope( cond::IOVSequence::ScopeType scope );

    /// Update the closure of the iov sequence
    void updateClosure( cond::Time_t newtillTime );

    // remove last entry
    unsigned int truncate(bool withPayload=false);

    // delete all entries
    void deleteEntries( bool withPayload=false);

    // 
    void import( const std::string& sourceIOVtoken );

    /// Returns the token of the iov sequence associated with this editor
    std::string const & token() const { return m_token;}

    Time_t firstSince() const;
  
    Time_t lastTill() const;
  
    TimeType timetype() const;
 

  private:

    void loadData( const std::string& token );
    void flushInserts();
    void flushUpdates();

    void init();
    bool validTime(cond::Time_t time, cond::TimeType timetype) const;
    bool validTime(cond::Time_t time) const;

    void debugInfo(std::ostream & co) const;
    void reportError(std::string message) const;
    void reportError(std::string message, cond::Time_t time) const;

  private:

    cond::DbSession m_dbSess;
    std::string m_token;
    bool m_isActive;
    boost::shared_ptr<cond::IOVSequence> m_iov;  
  };
}//ns cond
#endif
