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
  class Test12: public TestBase {
  public:
    Test12(): TestBase( "testORA_12" ){
    }

    virtual ~Test12(){
    }

    int execute( const std::string& connStr ){
      ora::Database db;
      //db.configuration().setMessageVerbosity( coral::Debug );
      db.connect( connStr );
      ora::ScopedTransaction trans( db.transaction() );
      //creating database
      trans.start( false );
      if(!db.exists()){
	db.create();
      }
      std::set< std::string > conts = db.containers();
      if( conts.find( "Cont0" )!= conts.end() ) db.dropContainer( "Cont0" );
      //creating container
      ora::Container cont0 = db.createContainer<SG>("Cont0");
      cont0.extendSchema<D0>();
      //inserting
      std::vector<boost::shared_ptr<SG> > buff0;
      for( unsigned int i = 0; i<10; i++){
	boost::shared_ptr<SG> data( new SG( i ) );
	db.insert( "Cont0", *data );
	buff0.push_back( data );
	data->m_ref = new D0(i);
	data->m_ref2 = new D0(i+10);
      }
      db.flush();
      buff0.clear();
      trans.commit();
      db.disconnect();
      sleep();
      // reading back...
      db.connect( connStr );
      trans.start( true );
      cont0 = db.containerHandle( "Cont0" );
      ora::ContainerIterator iter = cont0.iterator();
      while( iter.next() ){
	boost::shared_ptr<SG> obj = iter.get<SG>();
	unsigned int seed = obj->m_intData;
	SG r(seed);
	r.m_ref = new D0(seed);
	r.m_ref2 = new D0(seed+10);
	if( r != *obj ){
	  std::stringstream mess;
	  mess << "Data for class SG (1) different from expected for seed = "<<seed;
	  ora::throwException( mess.str(),"testORA_12");
	} else{
	  std::cout << "** Read out data for class SG (1) with seed="<<seed<<" is ok."<<std::endl;
	}
      }
      trans.commit();
      db.disconnect();
      db.disconnect();
      //inserting an instance of a class inherithing from the same base class
      std::cout << "************** writing more data..."<<std::endl;
      db.configuration().properties().setFlag( ora::Configuration::automaticContainerCreation() );
      db.connect( connStr );
      trans.start( false );
      std::vector<ora::OId> oids;
      for( unsigned int i = 0; i<10; i++){
	boost::shared_ptr<SG> data( new SG( i ) );
	oids.push_back( db.insert( "Cont0", *data ) );
	buff0.push_back( data );
	data->m_ref = new D1(i);
	data->m_ref2 = new D2(i);
      }
      db.flush();
      buff0.clear();
      trans.commit();
      db.disconnect();
      sleep();
      //reading back again
      db.connect( connStr );
      trans.start( true );
      for( std::vector<ora::OId>::iterator iO = oids.begin();
	   iO != oids.end(); ++iO ){
	std::string soid = iO->toString();
	ora::OId oid;
	oid.fromString( soid );
	boost::shared_ptr<SG> obj = db.fetch<SG>( oid );
	unsigned int seed = obj->m_intData;
	SG r(seed);
	r.m_ref = new D1(seed);
	r.m_ref2 = new D2(seed);
	if( r != *obj ){
	  std::stringstream mess;
	  mess << "Data for class SG (2) different from expected for seed = "<<seed;
	  ora::throwException( mess.str(),"testORA_12");
	} else{
	  std::cout << "** Read out data for class SG (2) with seed="<<seed<<" is ok."<<std::endl;
	}
      }
      
      trans.commit();
      //clean up
      db.transaction().start( false );
      db.drop();
      db.transaction().commit();
      db.disconnect();
      return 0;
    }
  };
}

int main( int argc, char** argv ){
  ora::Test12 test;
  return test.run();
}

