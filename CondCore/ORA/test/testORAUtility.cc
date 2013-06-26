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
  class TestORAUtilities: public TestBase {
  public:
    TestORAUtilities(): TestBase( "testORAUtilities" ){
    }

    virtual ~TestORAUtilities(){
    }

    void execute( const std::string& connStr0 ){
      ora::Database db;
      std::string connStr1( "sqlite_file:test1.db" );
      std::string connStr2( "sqlite_file:test2.db" );
      //db.configuration().setMessageVerbosity( coral::Debug );
      db.connect( connStr0 );
      ora::ScopedTransaction trans( db.transaction() );
      //creating source database
      trans.start( false );
      if(db.exists()){
	db.drop();
      }
      db.create();
      std::set< std::string > conts = db.containers();
      if( conts.find( "Cont0" )!= conts.end() ) db.dropContainer( "Cont0" );
      //creating container
      db.createContainer<SimpleClass>("Cont0");
      ora::Container contH0 = db.containerHandle( "Cont0" );
      //inserting into source db
      SimpleClass s0(4);
      int oid0 = contH0.insert( s0 );
      SimpleClass s1(999);
      int oid01 = contH0.insert( s1 );
      contH0.flush();
      trans.commit();
      db.disconnect();
      std::cout <<"## Source DB created."<<std::endl;
      sleep();
      db.connect( connStr1 );
      //creating dest db
      trans.start( false );
      if(db.exists()){
	db.drop();
      }
      db.create();
      trans.commit();
      //importing schema from source
      trans.start( false );
      ora::DatabaseUtility util = db.utility();
      util.importContainerSchema( connStr0, "Cont0" );
      std::cout <<"## Imported Schema for container \"Cont0\""<<std::endl;
      //inserting
      contH0 = db.containerHandle( "Cont0" );
      SimpleClass s01(5);
      oid0 = contH0.insert( s01 );
      SimpleClass s11(998);
      oid01 = contH0.insert( s11 );
      contH0.flush();
      trans.commit();
      db.disconnect();
      std::cout <<"## Objects inserted in new Container"<<std::endl;
      try {
	util.listMappingVersions( "Cont0" );
      } catch ( ora::Exception& e ){
	std::cout << "## Expected exception: "<<e.what()<<std::endl;
      }
      //reading back from dest db
      db.connect( connStr1 );
      trans.start( true );
      util = db.utility();
      std::set<std::string> vers = util.listMappingVersions( "Cont0" );
      for(std::set<std::string>::const_iterator iV = vers.begin();
	  iV != vers.end(); iV++ ){
	std::cout << "** vers=\""<<*iV<<"\""<<std::endl;
      }
      contH0 = db.containerHandle( "Cont0" );
      boost::shared_ptr<SimpleClass> sr0 = contH0.fetch<SimpleClass>( oid0);
      if( *sr0  != s01){
	std::stringstream mess;
	mess << "Data for oid="<<oid0<<" in db conn1 different from expected.";
	ora::throwException( mess.str(),"testORAUtilities");
      } else {
	std::cout << "** Read out data for oid="<<oid0<<" in db conn1 is ok."<<std::endl;
      }
      boost::shared_ptr<SimpleClass> sr1 = contH0.fetch<SimpleClass>( oid01);
      if( *sr1 != s11 ){
	std::stringstream mess;
	mess << "Data for oid="<<oid01<<" in db conn1 different from expected.";
	ora::throwException( mess.str(),"testORAUtilities");
      } else {
	std::cout << "** Read out data for oid="<<oid01<<" in db conn1 is ok."<<std::endl;
      }
      trans.commit();
      db.disconnect();
      //creating another database automatically
      db.configuration().properties().setFlag( ora::Configuration::automaticDatabaseCreation() );
      db.connect( connStr2 );
      db.transaction().start( false );
      if(db.exists()){
	db.drop();
      }
      db.create();
      conts = db.containers();
      if( conts.find( "Cont0" )!= conts.end() ) db.dropContainer( "Cont0" );
      util = db.utility();
      util.importContainer( connStr0, "Cont0" );
      std::cout <<"## Container \"Cont0\" imported."<<std::endl;
      trans.commit();
      db.disconnect();
      // reading back...
      db.connect( connStr2 );
      trans.start( true );
      contH0 = db.containerHandle( "Cont0" );
      ora::ContainerIterator iter = contH0.iterator();
      while( iter.next() ){
	boost::shared_ptr<SimpleClass> obj = iter.get<SimpleClass>();
	unsigned int seed = obj->id;
	SimpleClass r(seed);
	if( *obj != r ){
	  std::stringstream mess;
	  mess << "Data for seed="<<seed<<" in db conn2 different from expected.";
	  ora::throwException( mess.str(),"testORAUtilities");
	} else {
	  std::cout << "** Read out data for seed="<<seed<<" in db conn2 is ok."<<std::endl;        
	}
      }
      trans.commit();
      db.disconnect();
      //clean up source db
      db.connect( connStr0 );
      trans.start( false );
      db.drop();
      trans.commit();
      db.disconnect();
    }
  };
}

int main( int argc, char** argv ){
  ora::TestORAUtilities test;
  test.run();
}

