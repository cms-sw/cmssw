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
  class Test5: public TestBase {
  public:
    Test5(): TestBase( "testORA_5" ){
    }

    virtual ~Test5(){
    }

    void execute( const std::string& connStr ){
      ora::Database db;
      //db.configuration().setMessageVerbosity( coral::Debug );
      db.connect( connStr );
      ora::ScopedTransaction trans0( db.transaction() );
      trans0.start( false );
      if(!db.exists()){
	db.create();
      }
      std::set< std::string > conts = db.containers();
      if( conts.find( "Cont0" )!= conts.end() ) db.dropContainer( "Cont0" );
      //
      db.createContainer<MultiArrayClass2>("Cont0");
      ora::Container contH0 = db.containerHandle( "Cont0" );
      if( contH0.size() != 0 ){
	ora::throwException( "(0) Container Cont0 size is not 0 as expected.","testORA_5");
      } else {
	std::cout << "Container Cont0 size is 0 as expected."<<std::endl;
      }
      int oid0, oid1;
      MultiArrayClass2 a0(2);
      MultiArrayClass2 a1(3);
      {
	oid0 = contH0.insert( a0 );
	oid1 = contH0.insert( a1 );
	contH0.flush();
      }
      ora::OId id0( contH0.id(), oid0 );
      ora::OId id1( contH0.id(), oid1 );
      trans0.commit();
      db.disconnect();
      ::sleep(1);
      // reading back...
      db.connect( connStr );
      trans0.start( true );
      contH0 = db.containerHandle( "Cont0" );
      if( contH0.size() != 2 ){
	std::cout << "Container Cont0 size is="<<contH0.size()<<std::endl;
	ora::throwException( "(1) Container Cont0 size is not 2 as expected.","testORA_5");
      } else {
	std::cout << "Container Cont0 size is 2 as expected."<<std::endl;
      }
      boost::shared_ptr<MultiArrayClass2> ar0 = contH0.fetch<MultiArrayClass2 >( oid0 );
      if( *ar0 != a0 ){
	ora::throwException( "(2) Data read on oid0 different from expected.","testORA_5");
      } else {
	std::cout << "Data read on oid="<<oid0<<" is correct."<<std::endl;
      }
      boost::shared_ptr<MultiArrayClass2> ar1 = contH0.fetch<MultiArrayClass2 >( oid1 );
      if( *ar1 != a1 ){
	ora::throwException( "(3) Data read on oid1 different from expected.","testORA_5");
      } else {
	std::cout << "Data read on oid="<<oid1<<" is correct."<<std::endl;
      }
      trans0.commit();
      db.disconnect();
      // update...
      db.connect( connStr );
      trans0.start( false );
      MultiArrayClass2 au0(3);
      db.update( id0, au0 );
      std::auto_ptr<MultiArrayClass2> au1( new MultiArrayClass2(4) );
      db.update( id1, *au1 );
      db.flush();
      ora::OId id2( id1.containerId(), 400 );
      db.update( id2, au0 );
      trans0.commit();
      db.disconnect();
      // reading back...
      db.connect( connStr );
      trans0.start( true );
      ar0 = db.fetch<MultiArrayClass2 >( id0 );
      if( *ar0 != au0 ){
	ora::throwException( "(4) Data read on oid0 after update different from expected.","testORA_5");
      } else {
	std::cout << "Data read on oid="<<oid0<<" after update is correct."<<std::endl;
      }
      ar1 = db.fetch<MultiArrayClass2 >( id1 );
      if( *ar1 != *au1 ){
	ora::throwException( "(5) Data read on oid1 after update different from expected.","testORA_5");
      } else {
	std::cout << "Data read on oid="<<oid1<<" after update is correct."<<std::endl;
      }
      trans0.commit();
      db.disconnect();
      // delete...
      db.connect( connStr );
      trans0.start( false );
      contH0 = db.containerHandle( "Cont0" );
      if( contH0.size() != 2 ){
	ora::throwException( "(6) Container Cont0 size before delete is not 2 as expected.","testORA_5");
      } else {
	std::cout << "Container Cont0 size is 2 before delete as expected."<<std::endl;
      }
      db.erase( id0 );
      db.erase( id2 );
      if( contH0.size() != 2 ){
	ora::throwException( "(7) Container Cont0 size is not 2 before delete flush as expected.","testORA_5");
      } else {
	std::cout << "Container Cont0 size is 2 before delete flush as expected."<<std::endl;
      }
      db.flush();
      std::cout << "%%%% size after flush="<<contH0.size()<<std::endl;
      if( contH0.size() != 1 ){
	std::stringstream mess;
	mess << "(8) Container Cont0 after delete flush size="<<contH0.size();
	mess << " (expected=1)"<<std::endl;
	ora::throwException( mess.str(),"testORA_5");
      } else {
	std::cout << "Container Cont0 size is 2 after delete flush as expected."<<std::endl;
      }
      trans0.commit();
      db.disconnect();
      //
      db.connect( connStr );
      trans0.start( true );
      contH0 = db.containerHandle( "Cont0" );
      if( contH0.size() != 1 ){
	ora::throwException( "(9) Container Cont0 size is not 1 after delete as expected.","testORA_5");
      } else {
	std::cout << "Container Cont0 size is 1 after delete as expected."<<std::endl;
      }
      trans0.commit();
      trans0.start( false );
      db.drop();
      trans0.commit();
      db.disconnect();
    }
  };
}

int main(  int argc, char** argv ){
  ora::Test5 test;
  test.run();
}

