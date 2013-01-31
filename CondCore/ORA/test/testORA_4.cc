#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/ScopedTransaction.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/test/TestBase.h"
#include <iostream>
#include "classes.h"

using namespace testORA;

namespace ora {
  class Test4: public TestBase {
  public:
    Test4(): TestBase( "testORA_4" ){
    }

    virtual ~Test4(){
    }

    void execute( const std::string& connStr ){
      ora::Database db;
      //db.configuration().setMessageVerbosity( coral::Debug );
      db.connect( connStr );
      ora::ScopedTransaction trans0( db.transaction() );
      trans0.start( false );
      //creating database
      if(!db.exists()){
	db.create();
      }
      std::set< std::string > conts = db.containers();
      if( conts.find( "Cont0" )!= conts.end() ) db.dropContainer( "Cont0" );
      //creating container
      db.createContainer<MultiArrayClass>("Cont0");
      ora::Container contH0 = db.containerHandle( "Cont0" );
      //inserting
      MultiArrayClass a0(10);
      int oid0 = contH0.insert( a0 );
      MultiArrayClass a1(20);
      int oid1 = contH0.insert( a1 );
      contH0.flush();
      //
      trans0.commit();
      db.disconnect();
      sleep();
      // reading back...
      db.connect( connStr );
      trans0.start( true );
      contH0 = db.containerHandle( "Cont0" );
      boost::shared_ptr<MultiArrayClass> ar0 = contH0.fetch<MultiArrayClass >( oid0 );
      if( *ar0 != a0 ){
	ora::throwException( "Data read on oid0 different from expected.","testORA_4");
      } else {
	std::cout << "Data read on oid="<<oid0<<" is correct."<<std::endl;
      }
      ora::OId foid1( contH0.id(), oid1 );
      ora::Object r1 = db.fetchItem( foid1 );
      MultiArrayClass* ar1 = r1.cast<MultiArrayClass>();
      if( *ar1 != a1 ){
	ora::throwException( "Data read on oid1 different from expected.","testORA_4");
      } else {
	std::cout << "Data read on oid="<<oid1<<" is correct."<<std::endl;
      }
      r1.destruct();
      trans0.commit();
      //clean up
      trans0.start( false );
      db.drop();
      trans0.commit();
      db.disconnect();
     }
  };
}

int main(  int argc, char** argv ){
  ora::Test4 test;
  test.run();
}

