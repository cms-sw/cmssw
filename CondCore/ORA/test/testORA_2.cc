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
  class Test2: public TestBase {
  public:
    Test2(): TestBase( "testORA_2" ){
    }

    virtual ~Test2(){
    }

    void execute( const std::string& connStr ){
      ora::Database db;
      db.connect( connStr );
      ora::ScopedTransaction trans0( db.transaction() );
      //creating database
      trans0.start( false );
      if(!db.exists()){
	db.create();
      }
      std::set< std::string > conts = db.containers();
      if( conts.find( "Cont0" )!= conts.end() ) db.dropContainer( "Cont0" );
      if( conts.find( "Cont1" )!= conts.end() ) db.dropContainer( "Cont1" );
      if( conts.find( "std::vector<std::vector<int> >" )!= conts.end() ) db.dropContainer( "std::vector<std::vector<int> >" );
      //creating containers
      ora::Container contH0 = db.createContainer<std::vector<int> >("Cont0");    
      ora::Container contH1 = db.createContainer<std::vector<SimpleMember> >("Cont1");
      int contId = db.createContainer<std::vector<std::vector<int> > >().id();
      //inserting
      ora::Container contH2 = db.containerHandle( contId );
      std::vector<int> v0;
      int oid0 = contH0.insert( v0 );
      std::vector<int> v1;
      for(int i=0;i<2;i++) v1.push_back( i );
      int oid1 = contH0.insert( v1 );
      std::vector<int> v2;
      for(int i=0;i<100000;i++) v2.push_back( i );
      int oid2 = contH0.insert( v2 );
      contH0.flush();
      //
      std::vector<SimpleMember> vs;
      for(long i=0;i<5;i++) vs.push_back( SimpleMember( i ));
      int oids = contH1.insert( vs );
      contH1.flush();
      //
      std::vector<std::vector<int> > v3;
      for(int i=0;i<2000;i++) {
	std::vector<int> in;
	for(int j=0;j<50;j++) in.push_back(j);
	v3.push_back( in );
      }
      int oid3 = contH2.insert( v3 );
      std::vector<std::vector<int> > v4;
      for(int i=0;i<10;i++) {
	std::vector<int> in;
	for(int j=0;j<10000;j++) in.push_back(j);
	v4.push_back( in );
      }
      int oid4 = contH2.insert( v4 );
      contH2.flush();
      //
      trans0.commit();
      db.disconnect();
      sleep();
      // reading back...
      db.connect( connStr );
      trans0.start( true );
      contH0 = db.containerHandle( "Cont0" );
      boost::shared_ptr<std::vector<int> > vr0 = contH0.fetch<std::vector<int> >( oid0 );
      if( *vr0 != v0 ){
	ora::throwException( "Data read on oid0 different from expected.","testORA_2");
      } else {
	std::cout << "Data read on oid="<<oid0<<" is correct."<<std::endl;
      }
      boost::shared_ptr<std::vector<int> > vr1 = contH0.fetch<std::vector<int> >( oid1 );
      if( *vr1 != v1 ){
	ora::throwException( "Data read on oid1 different from expected.","testORA_2");
      } else {
	std::cout << "Data read on oid="<<oid1<<" is correct."<<std::endl;
      }
      boost::shared_ptr<std::vector<int> > vr2 = contH0.fetch<std::vector<int> >( oid2 );
      if( *vr2 != v2 ){
	ora::throwException( "Data read on oid2 different from expected.","testORA_2");
      } else {
	std::cout << "Data read on oid="<<oid2<<" is correct."<<std::endl;
      }
      //
      contH1 = db.containerHandle( "Cont1" );
      boost::shared_ptr<std::vector<SimpleMember> > vrs = contH1.fetch<std::vector<SimpleMember> >( oids );
      if( *vrs != vs ){
	ora::throwException( "Data read on oids different from expected.","testORA_2");
      } else {
	std::cout << "Data read on oid="<<oids<<" is correct."<<std::endl;
      }
      //
      contH2 = db.containerHandle( contId );
      boost::shared_ptr<std::vector<std::vector<int> > > vr3 = contH2.fetch<std::vector<std::vector<int> > >( oid3 );
      if( *vr3 != v3 ){
	ora::throwException( "Data read on oid3 different from expected.","testORA_2");
      } else {
	std::cout << "Data read on oid="<<oid3<<" is correct."<<std::endl;
      }
      boost::shared_ptr<std::vector<std::vector<int> > > vr4 = contH2.fetch<std::vector<std::vector<int> > >( oid4 );
      if( *vr4 != v4 ){
	ora::throwException( "Data read on oid4 different from expected.","testORA_2");
      } else {
	std::cout << "Data read on oid="<<oid4<<" is correct."<<std::endl;
      }
      //
      trans0.commit();
      db.disconnect();
      // update
      db.connect( connStr );
      trans0.start( false );
      //
      contH0 = db.containerHandle( "Cont0" );
      std::vector<int> vn;
      for(int i=10;i>0;i--) vn.push_back( i );
      contH0.update( oid1, vn );
      contH0.flush();
      //
      contH2 = db.containerHandle( contId );
      std::vector<std::vector<int> > v3n;
      for(int i=0;i<20;i++) {
	std::vector<int> in;
	for(int j=5000;j>0;j--) in.push_back(j);
	v3n.push_back( in );
      }
      contH2.update( oid3 , v3n );
      contH2.flush();
      //
      trans0.commit();
      db.disconnect();
      // reading back...
      db.connect( connStr );
      trans0.start( true );
      contH0 = db.containerHandle( "Cont0" );
      vr1 = contH0.fetch<std::vector<int> >( oid1 );
      if( *vr1 != vn ){
	ora::throwException( "Data read on oid1 after update different from expected.","testORA_2");
      } else {
	std::cout << "Data read on oid="<<oid1<<" after update is correct."<<std::endl;
      }
      contH2 = db.containerHandle( contId );
      vr3 = contH2.fetch<std::vector<std::vector<int> > >( oid3 );
      if( *vr3 != v3n ){
	ora::throwException( "Data read on oid3 after update different from expected.","testORA_2");
      } else {
	std::cout << "Data read on oid="<<oid3<<" after update is correct."<<std::endl;
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
  ora::Test2 test;
  test.run();
}

