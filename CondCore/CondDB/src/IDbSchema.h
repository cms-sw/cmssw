#ifndef CondCore_CondDB_IDbSchema_h
#define CondCore_CondDB_IDbSchema_h

//
#include <boost/date_time/posix_time/posix_time.hpp>

namespace cond {

  namespace persistency {

    class ITagTable {
    public:
      virtual ~ITagTable(){}
      virtual bool exists() = 0;
      virtual void create() = 0;
      virtual bool select( const std::string& name ) = 0;
      virtual bool select( const std::string& name, cond::TimeType& timeType, std::string& objectType, 
			   cond::Time_t& endOfValidity, std::string& description, cond::Time_t& lastValidatedTime ) = 0;
      virtual bool getMetadata( const std::string& name, std::string& description, 
				boost::posix_time::ptime& insertionTime, boost::posix_time::ptime& modificationTime ) = 0;
      virtual void insert( const std::string& name, cond::TimeType timeType, const std::string& objectType, 
			   cond::SynchronizationType synchronizationType, cond::Time_t endOfValidity, const std::string& description, 
			   cond::Time_t lastValidatedTime, const boost::posix_time::ptime& insertionTime ) = 0;
      virtual void update( const std::string& name, cond::Time_t& endOfValidity, const std::string& description, 
			   cond::Time_t lastValidatedTime, const boost::posix_time::ptime& updateTime ) = 0;
      virtual void updateValidity( const std::string& name, cond::Time_t lastValidatedTime, 
				   const boost::posix_time::ptime& updateTime ) = 0;
    };

    class IPayloadTable {
    public:
      virtual ~IPayloadTable(){}
      virtual bool exists() = 0;
      virtual void create() = 0;
      virtual bool select( const cond::Hash& payloadHash, std::string& objectType, cond::Binary& payloadData ) = 0;
      virtual bool getType( const cond::Hash& payloadHash, std::string& objectType ) = 0;
      //virtual bool insert( const cond::Hash& payloadHash, const std::string& objectType, 
      //			   const cond::Binary& payloadData, const boost::posix_time::ptime& insertionTime ) = 0;
      virtual cond::Hash insertIfNew( const std::string& objectType, const cond::Binary& payloadData, 
				      const boost::posix_time::ptime& insertionTime ) = 0;
    };

    class IIOVTable {
    public:
      virtual ~IIOVTable(){}
      virtual bool exists() = 0;
      virtual void create() = 0;
      virtual size_t selectGroups( const std::string& tag, std::vector<cond::Time_t>& groups ) = 0;
      virtual size_t selectSnapshotGroups( const std::string& tag, const boost::posix_time::ptime& snapshotTime, 
					   std::vector<cond::Time_t>& groups ) = 0;
      virtual size_t selectLatestByGroup( const std::string& tag, cond::Time_t lowerGroup, cond::Time_t upperGroup , 
					  std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs) = 0;
      virtual size_t selectSnapshotByGroup( const std::string& tag, cond::Time_t lowerGroup, cond::Time_t upperGroup, 
					    const boost::posix_time::ptime& snapshotTime, 
					    std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs) = 0;
      virtual size_t selectLatest( const std::string& tag, std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs) = 0;
      virtual bool getLastIov( const std::string& tag, cond::Time_t& since, cond::Hash& hash ) = 0;
      virtual bool getSize( const std::string& tag, size_t& size ) = 0;
      virtual bool getSnapshotSize( const std::string& tag, const boost::posix_time::ptime& snapshotTime, size_t& size ) = 0;
      virtual void insertOne( const std::string& tag, cond::Time_t since, cond::Hash payloadHash, 
			      const boost::posix_time::ptime& insertTime ) = 0;
      virtual void insertMany( const std::string& tag, 
			       const std::vector<std::tuple<cond::Time_t,cond::Hash,boost::posix_time::ptime> >& iovs ) = 0;
    };
    
    class ITagMigrationTable {
    public:
      virtual ~ITagMigrationTable(){}
      virtual bool exists() = 0;
      virtual void create() = 0;
      virtual bool select( const std::string& sourceAccount, const std::string& sourceTag, std::string& tagName ) = 0;
      virtual void insert( const std::string& sourceAccount, const std::string& sourceTag, const std::string& tagName, 
			   const boost::posix_time::ptime& insertionTime ) = 0;
    };
    
    class IIOVSchema {
    public: 
      virtual ~IIOVSchema(){}
      virtual bool exists() = 0;
      virtual bool create() = 0;
      virtual ITagTable& tagTable() = 0;
      virtual IIOVTable& iovTable() = 0;
      virtual IPayloadTable& payloadTable() = 0;
      virtual ITagMigrationTable& tagMigrationTable() = 0;
    };

    class IGTTable {
    public:
      virtual ~IGTTable(){}
      virtual bool exists() = 0;
      virtual bool select( const std::string& name ) = 0;
      virtual bool select( const std::string& name, cond::Time_t& validity, boost::posix_time::ptime& snapshotTime ) = 0;
      virtual bool select( const std::string& name, cond::Time_t& validity, std::string& description, 
			   std::string& release, boost::posix_time::ptime& snapshotTime ) = 0;
      virtual void insert( const std::string& name, cond::Time_t validity, const std::string& description, const std::string& release, 
			   const boost::posix_time::ptime& snapshotTime, const boost::posix_time::ptime& insertionTime ) = 0;
      virtual void update( const std::string& name, cond::Time_t validity, const std::string& description, const std::string& release, 
			   const boost::posix_time::ptime& snapshotTime, const boost::posix_time::ptime& insertionTime ) = 0;
    };
     
    class IGTMapTable {
    public:
      virtual ~IGTMapTable(){}
      virtual bool exists() = 0;
      virtual bool select( const std::string& gtName, std::vector<std::tuple<std::string,std::string,std::string> >& tags ) = 0;
      virtual bool select( const std::string& gtName, const std::string& preFix, const std::string& postFix, 
			   std::vector<std::tuple<std::string,std::string,std::string> >& tags ) = 0;
      virtual void insert( const std::string& gtName, const std::vector<std::tuple<std::string,std::string,std::string> >& tags ) = 0;
    };
    
    class IGTSchema {
    public: 
      virtual ~IGTSchema(){}
      virtual bool exists() = 0;
      virtual IGTTable& gtTable() = 0;
      virtual IGTMapTable& gtMapTable() = 0;
    };
    
  }
}
#endif
