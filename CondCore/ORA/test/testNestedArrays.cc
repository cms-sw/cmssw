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
  class Test1: public TestBase {
  public:
    Test1(): TestBase( "testORA_1" ){
    }

    virtual ~Test1(){
    }

    int execute( const std::string& connStr ){
      ora::Database db;
      // writing...
      db.connect( connStr );
      ora::ScopedTransaction trans0( db.transaction() );
      trans0.start( false );
      if(!db.exists()){
	db.create();
      }
      std::set< std::string > conts = db.containers();
      if( conts.find( "Cont0" )!= conts.end() ) db.dropContainer( "Cont0" );
      if( conts.find( "Cont1" )!= conts.end() ) db.dropContainer( "Cont1" );
      if( conts.find( "Cont2" )!= conts.end() ) db.dropContainer( "Cont2" );
      // ***
      ora::Container contH0 = db.createContainer<MixingModuleConfigV>("Cont0");
      std::map<int,boost::shared_ptr<MixingModuleConfigV> > buff0;
      for( int i=0; i<300;i++){
	boost::shared_ptr<MixingModuleConfigV> s(new MixingModuleConfigV(i));
        buff0.insert( std::make_pair( contH0.insert( *s ), s ) );
      }
      contH0.flush();
      ora::Container contH1 = db.createContainer<MixingModuleConfigA>("Cont1");
      std::map<int,boost::shared_ptr<MixingModuleConfigA> > buff1;
      for( int i=0; i<300;i++){
	boost::shared_ptr<MixingModuleConfigA> s(new MixingModuleConfigA(i));
        buff1.insert( std::make_pair( contH1.insert( *s ), s ) );
      }
      contH1.flush();
      ora::Container contH2 = db.createContainer<MixingModuleConfigIA>("Cont2");
      std::map<int,boost::shared_ptr<MixingModuleConfigIA> > buff2;
      for( int i=0; i<300;i++){
	boost::shared_ptr<MixingModuleConfigIA> s(new MixingModuleConfigIA(i));
        buff2.insert( std::make_pair( contH2.insert( *s ), s ) );
      }
      contH2.flush();
      trans0.commit();
      db.disconnect();
      // reading back...
      sleep();
      db.connect( connStr );
      ora::ScopedTransaction trans( db.transaction() );
      trans.start( true );
      contH0 = db.containerHandle( "Cont0" );
      for( std::map<int,boost::shared_ptr<MixingModuleConfigV> >::const_iterator iO = buff0.begin(); iO != buff0.end(); ++iO ){
	boost::shared_ptr<MixingModuleConfigV> sr = contH0.fetch<MixingModuleConfigV>( iO->first );
        if( *sr != *iO->second ){
	  std::cout <<"Vector Data read on oid="<<iO->first<<" different from expected."<<std::endl;
	  return 1;
        } 
      }
      std::cout << "Vector Data read are correct."<<std::endl;
      contH1 = db.containerHandle( "Cont1" );
      for( std::map<int,boost::shared_ptr<MixingModuleConfigA> >::const_iterator iO = buff1.begin(); iO != buff1.end(); ++iO ){
	boost::shared_ptr<MixingModuleConfigA> sr = contH1.fetch<MixingModuleConfigA>( iO->first );
        if( *sr != *iO->second ){
	  std::cout <<"Array Data read on oid="<<iO->first<<" different from expected."<<std::endl;
	  return 1;
        } 
      }
      std::cout << "Array Data read are correct."<<std::endl;
      contH2 = db.containerHandle( "Cont2" );
      for( std::map<int,boost::shared_ptr<MixingModuleConfigIA> >::const_iterator iO = buff2.begin(); iO != buff2.end(); ++iO ){
	boost::shared_ptr<MixingModuleConfigIA> sr = contH2.fetch<MixingModuleConfigIA>( iO->first );
        if( *sr != *iO->second ){
	  std::cout <<"Inline Array Data read on oid="<<iO->first<<" different from expected."<<std::endl;
	  return 1;
        } 
      }
      std::cout << "Inline Array Data read are correct."<<std::endl;
      trans.commit();
      trans.start( false );
      db.drop();
      trans.commit();
      db.disconnect();
      return 0;
    }
  };
}

int main( int argc, char** argv ){
  ora::Test1 test;
  return test.run( );
}

