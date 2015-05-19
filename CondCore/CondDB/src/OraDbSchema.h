#ifndef CondCore_CondDB_OraDbSchema_h
#define CondCore_CondDB_OraDbSchema_h

#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/Types.h"
#include "CondCore/CondDB/interface/Binary.h"
#include "IDbSchema.h"
//
#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
//
#include <boost/date_time/posix_time/posix_time.hpp>

namespace cond {

  namespace persistency {

    class IOVCache {
    public:
      IOVCache( cond::DbSession& s );
      cond::DbSession& session();
      cond::IOVProxy iovSequence();
      cond::IOVEditor editor();
      bool existsTag( const std::string& tag );
      void addTag( const std::string& tag, const std::string token );
      std::string getToken( const std::string& tag );
      bool load( const std::string& tag );
      void setValidationMode(){
	m_validationMode = true;
      }
      bool validationMode(){
	return m_validationMode;
      }
    private:
      cond::IOVEditor m_iovAccess ;
      std::string m_tag;
      std::string m_token;
      bool m_validationMode = false;
    };
    
    class OraTagTable : public ITagTable {
    public:
      explicit OraTagTable( IOVCache& cache );
      virtual ~OraTagTable(){}
      bool exists(){
	return true;
      }
      void create(){
      }

      bool select( const std::string& name );
      bool select( const std::string& name, cond::TimeType& timeType, std::string& objectType, cond::SynchronizationType& synchronizationType,
		   cond::Time_t& endOfValidity, std::string& description, cond::Time_t& lastValidatedTime );
      bool getMetadata( const std::string& name, std::string& description, 
			boost::posix_time::ptime& insertionTime, boost::posix_time::ptime& modificationTime );
      void insert( const std::string& name, cond::TimeType timeType, const std::string& objectType, 
		   cond::SynchronizationType synchronizationType, cond::Time_t endOfValidity, const std::string& description, 
		   cond::Time_t lastValidatedTime, const boost::posix_time::ptime& insertionTime );
      void update( const std::string& name, cond::Time_t& endOfValidity, const std::string& description, 
		   cond::Time_t lastValidatedTime, const boost::posix_time::ptime& updateTime );
      void updateValidity( const std::string& name, cond::Time_t lastValidatedTime, 
			   const boost::posix_time::ptime& updateTime );
      void setValidationMode();
    private:
      IOVCache& m_cache;
    };

    class OraPayloadTable : public IPayloadTable {
    public:
      explicit OraPayloadTable( DbSession& session );
      virtual ~OraPayloadTable(){}
      bool exists(){
	return true;
      }
      void create(){
      }
      bool select( const cond::Hash& payloadHash, std::string& objectType, cond::Binary& payloadData, cond::Binary& streamerInfoData );
      bool getType( const cond::Hash& payloadHash, std::string& objectType );
      cond::Hash insertIfNew( const std::string& objectType, const cond::Binary& payloadData, 
			      const cond::Binary& streamerInfoData, const boost::posix_time::ptime& insertionTime );
    private:
      cond::DbSession m_session;
    };

    class OraIOVTable : public IIOVTable {
    public:
      explicit OraIOVTable( IOVCache& cache );
      virtual ~OraIOVTable(){}
      bool exists(){
	return true;
      }
      void create(){
      }
      size_t selectGroups( const std::string& tag, std::vector<cond::Time_t>& groups );
      size_t selectSnapshotGroups( const std::string& tag, const boost::posix_time::ptime& snapshotTime, 
				   std::vector<cond::Time_t>& groups );
      size_t selectLatestByGroup( const std::string& tag, cond::Time_t lowerGroup, cond::Time_t upperGroup , 
				  std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs);
      size_t selectSnapshotByGroup( const std::string& tag, cond::Time_t lowerGroup, cond::Time_t upperGroup, 
				    const boost::posix_time::ptime& snapshotTime, 
				    std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs);
      size_t selectLatest( const std::string& tag, std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs);
      size_t selectSnapshot( const std::string& tag, const boost::posix_time::ptime& snapshotTime, std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs);
      bool getLastIov( const std::string& tag, cond::Time_t& since, cond::Hash& hash );
      bool getSnapshotLastIov( const std::string& tag, const boost::posix_time::ptime& snapshotTime, cond::Time_t& since, cond::Hash& hash );
      bool getSize( const std::string& tag, size_t& size );
      bool getSnapshotSize( const std::string& tag, const boost::posix_time::ptime& snapshotTime, size_t& size );
      void insertOne( const std::string& tag, cond::Time_t since, cond::Hash payloadHash, 
		      const boost::posix_time::ptime& insertTime );
      void insertMany( const std::string& tag, 
		       const std::vector<std::tuple<cond::Time_t,cond::Hash,boost::posix_time::ptime> >& iovs );
      void erase( const std::string& tag );
    private:
      IOVCache& m_cache;
    };
        
    class OraIOVSchema : public IIOVSchema {
    public: 
      explicit OraIOVSchema( DbSession& session );
      virtual ~OraIOVSchema(){}
      bool exists();
      bool create();
      ITagTable& tagTable();
      IIOVTable& iovTable();
      IPayloadTable& payloadTable();
      ITagMigrationTable& tagMigrationTable();
      IPayloadMigrationTable& payloadMigrationTable();
      std::string parsePoolToken( const std::string& poolToken );
    private:
      IOVCache m_cache;
      OraTagTable m_tagTable;
      OraIOVTable m_iovTable;
      OraPayloadTable m_payloadTable;
    };

    class OraGTTable : public IGTTable {
    public:
      explicit OraGTTable( DbSession& session ); 
      virtual ~OraGTTable(){}
      void create(){}
      bool exists(){
	return true;
      }
      bool select( const std::string& name );
      bool select( const std::string& name, cond::Time_t& validity, boost::posix_time::ptime& snapshotTime );
      bool select( const std::string& name, cond::Time_t& validity, std::string& description, 
		   std::string& release, boost::posix_time::ptime& snapshotTime );
      void insert( const std::string& name, cond::Time_t validity, const std::string& description, const std::string& release, 
		   const boost::posix_time::ptime& snapshotTime, const boost::posix_time::ptime& insertionTime );
      void update( const std::string& name, cond::Time_t validity, const std::string& description, const std::string& release, 
		   const boost::posix_time::ptime& snapshotTime, const boost::posix_time::ptime& insertionTime );
    private:
      cond::DbSession m_session;
    };
     
    class OraGTMapTable : public IGTMapTable {
    public:
      explicit OraGTMapTable( DbSession& session );
      virtual ~OraGTMapTable(){}
      void create(){}
      bool exists(){
	return true;
      }
      bool select( const std::string& gtName, std::vector<std::tuple<std::string,std::string,std::string> >& tags );
      bool select( const std::string& gtName, const std::string& preFix, const std::string& postFix,
		   std::vector<std::tuple<std::string,std::string,std::string> >& tags );
      void insert( const std::string& gtName, const std::vector<std::tuple<std::string,std::string,std::string> >& tags );
    private:
      cond::DbSession m_session;
    };
    
    class OraGTSchema : public IGTSchema {
    public: 
      OraGTSchema( DbSession& session );
      virtual ~OraGTSchema(){}
      void create();
      bool exists();
      IGTTable& gtTable();
      IGTMapTable& gtMapTable();
    private:
      cond::DbSession m_session;
      OraGTTable m_gtTable;
      OraGTMapTable m_gtMapTable;
    };
    
  }
}
#endif
