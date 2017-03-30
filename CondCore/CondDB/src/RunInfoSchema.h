#ifndef CondCore_CondDB_RunInfoSchema_h
#define CondCore_CondDB_RunInfoSchema_h

#include "DbCore.h"
#include "IDbSchema.h"
//
#include <boost/date_time/posix_time/posix_time.hpp>

namespace cond {

  namespace persistency {

    conddb_table( RUN_INFO ) {
      
      conddb_column( RUN_NUMBER, cond::Time_t );
      conddb_column( START_TIME, boost::posix_time::ptime );
      conddb_column( END_TIME, boost::posix_time::ptime );

      struct MAX_RUN_NUMBER {					 
	typedef cond::Time_t type;				   
	static constexpr size_t size = 0;
	static std::string tableName(){ return RUN_NUMBER::tableName(); }	
	static std::string fullyQualifiedName(){ 
	  return "MAX("+RUN_NUMBER::fullyQualifiedName()+")";	  
	} 
      };

      struct MIN_RUN_NUMBER {					 
	typedef cond::Time_t type;				   
	static constexpr size_t size = 0;
	static std::string tableName(){ return RUN_NUMBER::tableName(); }	
	static std::string fullyQualifiedName(){ 
	  return "MIN("+RUN_NUMBER::fullyQualifiedName()+")";	  
	} 
      };

      struct MIN_START_TIME {					 
	typedef boost::posix_time::ptime type;				   
	static constexpr size_t size = 0;
	static std::string tableName(){ return START_TIME::tableName(); }	
	static std::string fullyQualifiedName(){ 
	  return "MIN("+START_TIME::fullyQualifiedName()+")";	  
	} 
      };
      
      class Table : public IRunInfoTable {
      public:
	explicit Table( coral::ISchema& schema );
	virtual ~Table(){}
	bool exists();
	void create();
	bool select( cond::Time_t runNumber, boost::posix_time::ptime& start, boost::posix_time::ptime& end );
        cond::Time_t getLastInserted();
	bool getInclusiveRunRange( cond::Time_t lower, cond::Time_t upper,
				   std::vector<std::tuple<cond::Time_t,boost::posix_time::ptime,boost::posix_time::ptime> >& runData );
	bool getInclusiveTimeRange( const boost::posix_time::ptime& lower ,const boost::posix_time::ptime& upper, 
				    std::vector<std::tuple<cond::Time_t,boost::posix_time::ptime,boost::posix_time::ptime> >& runData );
	void insertOne( cond::Time_t runNumber, const boost::posix_time::ptime& start, const boost::posix_time::ptime& end);
	void insert( const std::vector<std::tuple<cond::Time_t,boost::posix_time::ptime,boost::posix_time::ptime> >& runs );
	void updateEnd( cond::Time_t runNumber, const boost::posix_time::ptime& end );

      private:
	coral::ISchema& m_schema;
      };
    }

    
    class RunInfoSchema : public IRunInfoSchema {
    public: 
      explicit RunInfoSchema( coral::ISchema& schema );
      virtual ~RunInfoSchema(){}
      bool exists();
      bool create();
      IRunInfoTable& runInfoTable();
    private:
      RUN_INFO::Table m_runInfoTable;
    };

  }
}
#endif
