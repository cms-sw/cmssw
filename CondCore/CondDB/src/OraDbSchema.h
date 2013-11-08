#ifndef CondCore_CondDB_OraDbSchema_h
#define CondCore_CondDB_OraDbSchema_h

#include "IDbSchema.h"
//
#include <boost/date_time/posix_time/posix_time.hpp>

namespace cond {

  namespace persistency {
    
    class OraTagTable : public ITagTable {
    public:
      explicit OraTagTable( IOVProxy& proxy );
      virtual ~OraTagTable(){}
      bool exists(){
	return true;
      }
      void create(){
      }

      bool select( const std::string& name );
      bool select( const std::string& name, cond::TimeType& timeType, std::string& objectType, 
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
    private:
      MetaData m_metadata;
      IOVProxy m_iov;
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
      bool select( const cond::Hash& payloadHash );
      bool select( const cond::Hash& payloadHash, std::string& objectType, cond::Binary& payloadData );
      bool insert( const cond::Hash& payloadHash, const std::string& objectType, 
		   const cond::Binary& payloadData, const boost::posix_time::ptime& insertionTime );
      cond::Hash insertIfNew( const std::string& objectType, const cond::Binary& payloadData, 
	       		      const boost::posix_time::ptime& insertionTime );
    private:
      DbSession m_session;
    };

    class OraIOVTable : public IIOVTable {
    public:
      explicit OraPayloadTable( IOVProxy& proxy );
      virtual ~IIOVTable(){}
      bool exists(){
	return true;
      }
      void create(){
      }
      size_t selectGroups( const std::string& tag, std::vector<cond::Time_t>& groups );
      size_t selectSnapshotGroups( const std::string& tag, const boost::posix_time::ptime& snapshotTime, 
				   std::vector<cond::Time_t>& groups );
      size_t selectLastByGroup( const std::string& tag, cond::Time_t lowerGroup, cond::Time_t upperGroup , 
				std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs);
      size_t selectSnapshotByGroup( const std::string& tag, cond::Time_t lowerGroup, cond::Time_t upperGroup, 
				    const boost::posix_time::ptime& snapshotTime, 
				    std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs);
      size_t selectLast( const std::string& tag, std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs);
      void insertOne( const std::string& tag, cond::Time_t since, cond::Hash payloadHash, 
		      const boost::posix_time::ptime& insertTime );
      void insertMany( const std::string& tag, 
		       const std::vector<std::tuple<cond::Time_t,cond::Hash,boost::posix_time::ptime> >& iovs );
    private:
      IOVProxy m_iov;
    };
    
    class OraTagMigrationTable : public ITagMigrationTable {
    public:
      OraTagMigrationTable();
      virtual ~OraTagMigrationTable(){}
      bool exists(){
	return true;
      }
      void create(){
      }
      bool select( const std::string&, const std::string&, std::string& ){
	return true;
      }
      void insert( const std::string&, const std::string&, const std::string&, 
		   const boost::posix_time::ptime& ){
      }
    };
    
    class OraIOVSchema {
    public: 
      explicit OraIOVSchema( DbSession& session );
      virtual ~OraIOVSchema(){}
      bool exists();
      bool create();
      ITagTable& tagTable();
      IIOVTable& iovTable();
      IPayloadTable& payloadTable();
      ITagMigrationTable& tagMigrationTable();
    private:
      DbSession m_session;
    };

    class OraGTTable {
    public:
      explicit OraGtTable( DbSession& session ); 
      virtual ~OraGTTable(){}
      bool exists(){
	return true;
      }
      bool select( const std::string& name );
      bool select( const std::string& name, cond::Time_t& validity, boost::posix_time::ptime& snapshotTime );
      bool select( const std::string& name, cond::Time_t& validity, std::string& description, 
		   std::string& release, boost::posix_time::ptime& snapshotTime );
      void insert( const std::string& name, cond::Time_t validity, const std::string& description, const std::string& release, 
		   const boost::posix_time::ptime& snapshotTime, const boost::posix_time::ptime& insertionTime );
      virtual void update( const std::string& name, cond::Time_t validity, const std::string& description, const std::string& release, 
			   const boost::posix_time::ptime& snapshotTime, const boost::posix_time::ptime& insertionTime );
    private:
      DbSession m_session;
    };
     
    class OraGTMapTable {
    public:
      explicit OraGTMapTable( DbSession& session );
      virtual ~OraGTMapTable(){}
      bool exists(){
	return true;
      }
      bool select( const std::string& gtName, std::vector<std::tuple<std::string,std::string,std::string> >& tags );
      void insert( const std::string& gtName, const std::vector<std::tuple<std::string,std::string,std::string> >& tags );
    private:
      DbSession m_session;
    };
    
    class OraGTSchema {
    public: 
      OraGTSchema( DbSession& session );
      virtual ~OraGTSchema(){}
      bool exists();
      IGTTable& gtTable();
      IGTMapTable& gtMapTable();
    private:
      DbSession m_session;
    };
    
  }
}
#endif
