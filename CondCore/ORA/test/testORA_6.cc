#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/ScopedTransaction.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/test/TestBase.h"
#include <cstdlib>
#include <iostream>
#include "classes.h"

using namespace testORA;

namespace ora {
  class Test6: public TestBase {
  public:
    Test6(): TestBase( "testORA_6" ){
    }

    virtual ~Test6(){
    }

    void execute( const std::string& connStr ){
      ora::Database db;
      //db.configuration().setMessageVerbosity( coral::Debug );
      db.connect( connStr );
      ora::ScopedTransaction trans0( db.transaction() );
      //creating database
      trans0.start( false );
      if(!db.exists()){
	db.create();
      }
      std::set< std::string > conts = db.containers();
      if( conts.find( "Cont0" )!= conts.end() ) db.dropContainer( "Cont0" );
      //create container
      db.createContainer<SA>("Cont0");
      trans0.commit();
      db.disconnect();
      //inserting
      db.connect( connStr );
      trans0.start( false );
      ora::Container contH0 = db.containerHandle( "Cont0" );
      std::vector<boost::shared_ptr<SA> > buff;
      for( unsigned int i=0;i<5;i++){
	boost::shared_ptr<SA> obj( new SA(i) );
	contH0.insert( *obj );
	buff.push_back( obj );
      }
      contH0.flush();
      buff.clear();
      trans0.commit();
      db.disconnect();
      sleep();
      // reading back...
      db.connect( connStr );
      trans0.start( true );
      contH0 = db.containerHandle( "Cont0" );
      ora::ContainerIterator iter = contH0.iterator();
      while( iter.next() ){
	boost::shared_ptr<SA> obj = iter.get<SA>();
	int seed = obj->m_intData;
	SA r(seed);
	if( r != *obj ){
	  std::stringstream mess;
	  mess << "Data different from expected for seed = "<<seed;
	  ora::throwException( mess.str(),"testORA_6");
	} else{
	  std::cout << "** Read out data with seed="<<seed<<" is ok."<<std::endl;
	}
      }
      trans0.commit();
      //clean up
      trans0.start( false );
      db.drop();
      trans0.commit();
      db.disconnect();
    }
  };
}

int main( int argc, char** argv ){
  ora::Test6 test;
  test.run();
}

