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
  class Test9: public TestBase {
  public:
    Test9(): TestBase( "testORA_9" ){
    }

    virtual ~Test9(){
    }

    void execute( const std::string& connStr ){
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
      db.createContainer<SD>("Cont0");
      std::vector<boost::shared_ptr<SD> > buff;
      //inserting
      for( unsigned int i = 0; i<10; i++){
	boost::shared_ptr<SD> data( new SD( i ) );
	db.insert( "Cont0", *data );
	buff.push_back( data );
	data->m_ptr = new SimpleClass(i);
	for( unsigned int j=0;j<i;j++ ){
	  data->m_ptrVec.push_back( ora::Ptr<SimpleMember>( new SimpleMember(j) ));
	}
      }
      db.flush();
      buff.clear();
      trans.commit();
      db.disconnect();
      sleep();
      // reading back...
      db.connect( connStr );
      trans.start( true );
      ora::Container cont0 = db.containerHandle( "Cont0" );
      ora::ContainerIterator iter = cont0.iterator();
      while( iter.next() ){
	boost::shared_ptr<SD> obj = iter.get<SD>();
	unsigned int seed = obj->m_intData;
	SD r(seed);
	r.m_ptr = new SimpleClass(seed);
	for( unsigned int j=0;j<seed;j++ ){
	  r.m_ptrVec.push_back( ora::Ptr<SimpleMember>( new SimpleMember(j) ));
	}
	if( r != *obj ){
	  std::stringstream mess;
	  mess << "(1) Data for class SD different from expected for seed = "<<seed;
	  ora::throwException( mess.str(),"testORA_9");
	} else{
	  std::cout << "** (1) Read out data for class SD with seed="<<seed<<" is ok."<<std::endl;
	}
      }
      iter.reset();
      while( iter.next() ){
	boost::shared_ptr<SD> obj = iter.get<SD>();
	unsigned int seed = obj->m_intData;
	SD r(seed);
	r.m_ptr = new SimpleClass(seed);
	for( unsigned int j=0;j<seed;j++ ){
	  r.m_ptrVec.push_back( ora::Ptr<SimpleMember>( new SimpleMember(j) ));
	}
	if( r != *obj ){
	  std::stringstream mess;
	  mess << "(2) Data for class SD different from expected for seed = "<<seed;
	  ora::throwException( mess.str(),"testORA_9");
	} else{
	  std::cout << "** (2) Read out data for class SD with seed="<<seed<<" is ok."<<std::endl;
	}
      }
      trans.commit();
      //clean up
      trans.start( false );
      db.drop();
      trans.commit();
      db.disconnect();
    }
  };
}

int main( int argc, char** argv ){
  ora::Test9 test;
  test.run();
}

