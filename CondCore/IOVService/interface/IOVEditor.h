#ifndef CondCore_IOVService_IOVEditor_h
#define CondCore_IOVService_IOVEditor_h

#include "CondCore/IOVService/interface/IOVProxy.h"

#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondFormats/Common/interface/IOVSequence.h"

//
// Package:     CondCore/IOVService
// Class  :     IOVEditor
//
/**\class IOVEditor IOVEditor.h CondCore/IOVService/interface/IOVEditor.h
   Description: iov sequence manipulator
*/
//
// Author:      Zhen Xie
// Fixes and other changes: Giacomo Govi
//

namespace cond{

  class IOVElement;
  class IOVSequence;

  class ExportRegistry {
  public:
    explicit ExportRegistry( DbConnection& conn );
    ExportRegistry();

    void open( const std::string& connectionString, bool readOnly=false );

    void addMapping( const std::string& oId, const std::string& newOid );

    std::string getMapping( const std::string& oId );

    void flush();

    void close();
  private:
    cond::DbConnection m_conn;
    cond::DbSession m_session;
    std::map<std::string,std::string> m_buffer;
  };

  class IOVImportIterator {
  public:
    explicit IOVImportIterator( boost::shared_ptr<cond::IOVProxyData>& destIov );

    virtual ~IOVImportIterator();

    void setUp( cond::IOVProxy& sourceIov, cond::Time_t since, cond::Time_t till, bool outOfOrder, size_t bulkSize = 1 );

    void setUp( cond::DbSession& sourceSess, const std::string& sourceIovToken, cond::Time_t since, cond::Time_t till,
		bool outOfOrder, size_t bulkSize = 1 );

    void setUp( cond::IOVProxy& sourceIov, size_t bulkSize = 1 );

    void setUp( cond::DbSession& sourceSess, const std::string& sourceIovToken, size_t bulkSize = 1 );

    void setUp( cond::IOVProxy& sourceIov, cond::ExportRegistry& registry, size_t bulkSize = 1 );

    bool hasMoreElements();

    size_t importMoreElements();

    size_t importAll();

  private:
    std::string importPayload( const std::string& payloadToken );

  private:
    cond::IOVProxy m_sourceIov;
    boost::shared_ptr<cond::IOVProxyData> m_destIov;
    cond::Time_t m_lastSince;
    size_t m_bulkSize;
    IOVSequence::const_iterator m_cursor;
    IOVSequence::const_iterator m_till;
    ExportRegistry* m_registry;
  };

  class IOVEditor{
  public:

    // default constructor
    explicit IOVEditor(cond::DbSession& dbSess);

    // constructor from existing iov
    IOVEditor( cond::DbSession& dbSess, const std::string& token);
 
    /// Destructor
    ~IOVEditor();

    void reload();

    void load( const std::string& token );

    bool createIOVContainerIfNecessary();

    // create empty default sequence
    std::string create( cond::TimeType timetype );

    // create empty sequence with fixed time boundary
    std::string create(cond::TimeType timetype, cond::Time_t lastTill, const std::string& metadata );

    // ####### TO BE REOMOVED ONLY USED IN TESTS
    std::string create(cond::TimeType timetype, cond::Time_t lastTill );

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
    size_t import( cond::DbSession& sourceSess, const std::string& sourceIovToken );

    // 
    boost::shared_ptr<IOVImportIterator> importIterator();

    TimeType timetype() const;

    std::string const & token() const; 

    cond::IOVProxy proxy();

  private:

    bool validTime(cond::Time_t time, cond::TimeType timetype) const;
    bool validTime(cond::Time_t time) const;

    void debugInfo(std::ostream & co) const;
    void reportError(std::string message) const;
    void reportError(std::string message, cond::Time_t time) const;

  private:

    bool m_isLoaded;
    boost::shared_ptr<cond::IOVProxyData> m_iov;
  };
}//ns cond
#endif
