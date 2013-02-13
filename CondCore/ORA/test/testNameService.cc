#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/ScopedTransaction.h"
#include "CondCore/ORA/test/TestBase.h"
#include <iostream>
#include "classes.h"

using namespace testORA;

namespace ora {
  class TestNameService: public TestBase {
  public:
    TestNameService(): TestBase( "testNameService" ){
    }

    virtual ~TestNameService(){
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
      //creating container
      ora::Container contH0 = db.createContainer<SimpleClass>("Cont0");
      //inserting
      SimpleClass s0(4);
      int oid0 = contH0.insert( s0 );
      SimpleClass s1(999);
      int oid1 = contH0.insert( s1 );
      contH0.flush();
      std::cout << "******* setting name..."<<std::endl; 
      if( db.eraseObjectName( "Peeppino" ) ){
	std::cout << "** Name was found and erased."<<std::endl;
      }
      db.setObjectName( "Peeppino",ora::OId(contH0.id(),oid1)); 
      trans0.commit();
      db.disconnect();
      // reading back...
      sleep();
      db.connect( connStr );
      ora::ScopedTransaction trans1( db.transaction() );
      trans1.start( true );
      contH0 = db.containerHandle( "Cont0" );
      boost::shared_ptr<SimpleClass> sr0 = contH0.fetch<SimpleClass>( oid0);
      if( *sr0 != s0 ){
	ora::throwException( "Data read on oid0 different from expected.","testORA_1");
      } else {
	std::cout << "Data read on oid="<<oid0<<" is correct."<<std::endl;
      }
      std::cout << "*********Getting from name..."<<std::endl;
      boost::shared_ptr<SimpleClass> sr1 = db.fetchByName<SimpleClass>( "Peeppino" ); 
      if( *sr1 != s1 ){
	ora::throwException( "Data read on oid1 different from expected.","testORA_1");
      } else {
	std::cout << "Data read on oid="<<oid1<<" is correct."<<std::endl;
      }
      trans1.commit();
      //clean up
      trans1.start( false );
      db.drop();
      trans1.commit();
      db.disconnect();
    }
  };
}

int main(int argc, char** argv ){
  ora::TestNameService test;
  test.run();
}

