#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/ScopedTransaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/test/TestBase.h"
#include <iostream>

namespace ora {
  class TestContainerLock: public TestBase {
  public:
    TestContainerLock(): TestBase( "testContainerLock" ){
    }

    virtual ~TestContainerLock(){
    }

    int execute( const std::string& connStr ){
      ora::Database db0;
      //db0.configuration().setMessageVerbosity( coral::Debug );
      db0.connect( connStr );
      ora::ScopedTransaction trans0( db0.transaction() );
      //creating database
      trans0.start( false );
      if(!db0.exists()){
	db0.create();
      }
      std::set< std::string > conts = db0.containers();
      //creating containers
      std::cout << "#### creating containers..."<<std::endl;
      if( conts.find( "Cont0" )== conts.end() ) db0.createContainer<int>("Cont0");
      trans0.commit();
      //inserting
      trans0.start( false );
      ora::Container contH0 = db0.containerHandle( "Cont0" );
      if( contH0.isLocked() ){
	ora::throwException( "Container should not be locked.","testContainerLock");
      }
      std::cout << "#### locking..."<<std::endl;
      contH0.lock();
      if( contH0.isLocked() ){
	std::cout <<"### container has been locked..."<<std::endl;
      }
      std::cout << "#### writing..."<<std::endl;
      int myInt0(999);
      int myInt1(1234567890);
      contH0.insert( myInt0 );
      contH0.insert( myInt1 );
      contH0.flush();
      //db0.dropContainer( "Cont0" );
      trans0.commit();
      db0.disconnect();
      sleep();
      //reading back
      db0.connect( connStr );
      ora::ScopedTransaction trans1( db0.transaction() );
      trans1.start( false );
      ora::Container cnt = db0.containerHandle( "Cont0" );
      std::cout <<"### Container \""<<cnt.name()<<"\" has got "<<cnt.size()<<" objects."<<std::endl; 
      //clean up
      db0.drop();
      trans1.commit();
      db0.disconnect();
      return 0;
    }
  };
}

int main(int argc, char** argv){
  ora::TestContainerLock test;
  return test.run();
}

