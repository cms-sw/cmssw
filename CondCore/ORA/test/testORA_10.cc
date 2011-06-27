#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/ScopedTransaction.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/IReferenceHandler.h"
#include "CondCore/ORA/test/TestBase.h"
#include <iostream>

#include "classes.h"

using namespace testORA;

namespace ora {
  class Test10: public TestBase {
  public:
    Test10(): TestBase( "testORA_10" ){
    }

    virtual ~Test10(){
    }

    void execute( const std::string& connStr ){
      ora::Database db;
      //db.configuration().setMessageVerbosity( coral::Debug );
      db.connect( connStr );
      ora::ScopedTransaction trans( db.transaction() );
      trans.start( false );
      if(!db.exists()){
	db.create();
      }
      std::set< std::string > conts = db.containers();
      if( conts.find( "Cont0" )!= conts.end() ) db.dropContainer( "Cont0" );
      db.createContainer<SE>("Cont0");
      std::vector<boost::shared_ptr<SE> > buff;
      std::vector<ora::OId> oids;
      for( unsigned int i = 0; i<5; i++){
	boost::shared_ptr<SE> data( new SE( i ) );
	if(data->m_vec.size()) for( unsigned int j=0;j<data->m_vec.size();j++){
	  std::cout << " For i="<<i<<" Vec elem size="<<data->m_vec[j].m_v.size()<<std::endl;
	}
	
	oids.push_back( db.insert( "Cont0", *data ) );
	buff.push_back( data );
      }
      db.flush();
      buff.clear();
      trans.commit();
      db.disconnect();
      ::sleep(1);
      // reading back...
      db.connect( connStr );
      trans.start( true );
      ora::Container cont0 = db.containerHandle( "Cont0" );
      ora::ContainerIterator iter = cont0.iterator();
      while( iter.next() ){
	boost::shared_ptr<SE> obj = iter.get<SE>();
	unsigned int seed = obj->m_intData;
	SE r(seed);
	if( r != *obj ){
	  std::stringstream mess;
	  mess << "Data for class SE different from expected for seed = "<<seed;
	  ora::throwException( mess.str(),"testORA_10");
	} else{
	  std::cout << "** Read out data for class SE with seed="<<seed<<" is ok."<<std::endl;
	}
      }
      trans.commit();
      db.disconnect();
      db.connect( connStr );
      trans.start( false );
      for( std::vector<ora::OId>::const_iterator iOid=oids.begin();
	   iOid != oids.end(); ++iOid ){
	boost::shared_ptr<SE> data = db.fetch<SE>(*iOid);
	db.update( *iOid, *data );
	buff.push_back( data );
	data->m_vec.push_back( SM(99) );
	data->m_vec.push_back( SM(100) );
      }
      db.flush();
      buff.clear();
      trans.commit();
      db.disconnect();
      db.connect( connStr );
      trans.start( true );
      cont0 = db.containerHandle( "Cont0" );
      iter = cont0.iterator();
      while( iter.next() ){
	boost::shared_ptr<SE> obj = iter.get<SE>();
	unsigned int seed = obj->m_intData;
	SE r(seed);
	r.m_vec.push_back( SM(99) );
	r.m_vec.push_back( SM(100) );
	
	if( r != *obj ){
	  std::stringstream mess;
	  mess << "Data for class SE after update (1) different from expected for seed = "<<seed;
	  ora::throwException( mess.str(),"testORA_10");
	} else{
	  std::cout << "** Read out data for class SE after update (1) with seed="<<seed<<" is ok."<<std::endl;
	}
      }
      trans.commit();
      db.disconnect();
      db.connect( connStr );
      trans.start( false );
      for( std::vector<ora::OId>::const_iterator iOid=oids.begin();
	   iOid != oids.end(); ++iOid ){
	boost::shared_ptr<SE> data = db.fetch<SE>(*iOid);
	db.update( *iOid, *data );
	buff.push_back( data );
	if(!data->m_vec.empty()) data->m_vec.pop_back();
	if(!data->m_vec.empty()) data->m_vec.pop_back();
      }
      db.flush();
      buff.clear();
      trans.commit();
      db.disconnect();
      db.connect( connStr );
      trans.start( true );
      cont0 = db.containerHandle( "Cont0" );
      iter = cont0.iterator();
      while( iter.next() ){
	boost::shared_ptr<SE> obj = iter.get<SE>();
	unsigned int seed = obj->m_intData;
	SE r(seed);
	if( r != *obj ){
	  std::stringstream mess;
	  mess << "Data for class SE after update (2) different from expected for seed = "<<seed;
	  ora::throwException( mess.str(),"testORA_10");
	} else{
	  std::cout << "** Read out data for class SD after update (2) with seed="<<seed<<" is ok."<<std::endl;
	}
      }
      db.transaction().commit();
      db.transaction().start( false );
      db.drop();
      db.transaction().commit();
      db.disconnect();
    }
  };
}

int main( int argc, char** argv ){
  ora::Test10 test;
  test.run();
}

