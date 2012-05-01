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
  class TestNamedReference: public TestBase {
  public:
    TestNamedReference(): TestBase( "testNamedReference" ){
    }

    virtual ~TestNamedReference(){
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
      if( conts.find( "Cont1" )!= conts.end() ) db.dropContainer( "Cont1" );
      // ***
      ora::Container contH0 = db.createContainer<SimpleClass>("Cont0");
      ora::Container contH1 = db.createContainer<SH>("Cont1");
      boost::shared_ptr<SimpleClass> s0(new SimpleClass(999));
      int oid0 = contH0.insert( *s0 );
      contH0.flush();
      std::cout << "******* setting name..."<<std::endl; 
      if( db.eraseObjectName( "Peeppino" ) ){
	std::cout << "** Name was found and erased."<<std::endl;
      }
      db.setObjectName( "Peeppino",ora::OId(contH0.id(),oid0)); 
      SH s1(111);
      int oid1 = contH1.insert( s1 );
      SH s2(222);
      int oid2 = contH1.insert( s2 );
      s2.m_ref = ora::NamedRef<SimpleClass>( "Peeppino",s0 );
      contH1.flush();
      trans0.commit();
      db.disconnect();
      // reading back...
      ::sleep(1);
      db.connect( connStr );
      ora::ScopedTransaction trans1( db.transaction() );
      trans1.start( true );
      contH0 = db.containerHandle( "Cont0" );
      contH1 = db.containerHandle( "Cont1" );
      std::vector<std::string> contNames; 
      if(!contH0.getNames( contNames )){
	std::cout << "** No names found for container "<<contH0.name()<<std::endl;
      } else {
	std::cout << "** Found "<<contNames.size()<<" names for container "<<contH0.name()<<std::endl;
	for( std::vector<std::string>::const_iterator iN = contNames.begin();
	     iN != contNames.end(); ++iN ){
	  std::cout << "** Name: "<<*iN<<std::endl;
	}
      }
      contNames.clear();
      if(!contH1.getNames( contNames )){
	std::cout << "** No names found for container "<<contH1.name()<<std::endl;
      } else {
	std::cout << "** Found "<<contNames.size()<<" names for container "<<contH1.name()<<std::endl;
	for( std::vector<std::string>::const_iterator iN = contNames.begin();
	     iN != contNames.end(); ++iN ){
	  std::cout << "** Name: "<<*iN<<std::endl;
	}
      }
      std::vector<std::string> o0Names;
      if(!db.getNamesForObject( ora::OId( contH0.id(),oid0 ), o0Names)){
	std::cout << "** No names found for object cid="<<contH0.id()<<" oid="<<oid0<<std::endl;
      } else {
	std::cout << "** Found "<<o0Names.size()<<" names for object cid="<<contH0.id()<<" oid="<<oid0<<std::endl;
	for( std::vector<std::string>::const_iterator iN = o0Names.begin();
	     iN != o0Names.end(); ++iN ){
	  std::cout << "** Name: "<<*iN<<std::endl;
	}
      }
      std::cout << "** First object..."<<std::endl;
      boost::shared_ptr<SH> sr1 = contH1.fetch<SH>( oid1);
      try{
	if( *sr1 != s1 ){
	  ora::throwException( "Data read on oid1 different from expected.","testNamedReference");
	} else {
	  std::cout << "Data read on oid="<<oid1<<" is correct."<<std::endl;
	}
      } catch (ora::Exception& e ){
	std::cout <<"****1 Exception: "<<e.what()<<std::endl;
      }
      std::vector<std::string> o1Names;
      if(!db.getNamesForObject( ora::OId( contH1.id(),oid1 ), o1Names)){
	std::cout << "** No names found for object cid="<<contH1.id()<<" oid="<<oid1<<std::endl;
      } else {
	std::cout << "** Found "<<o1Names.size()<<" names for object cid="<<contH1.id()<<" oid="<<oid2<<std::endl;
	for( std::vector<std::string>::const_iterator iN = o1Names.begin();
	     iN != o1Names.end(); ++iN ){
	  std::cout << "** Name: "<<*iN<<std::endl;
	}
      }
      std::cout << "** Second object..."<<std::endl;
      boost::shared_ptr<SH> sr2 = contH1.fetch<SH>( oid2);
      try{
	if( *sr2 != s2 ){
	  ora::throwException( "Data read on oid1 different from expected.","testNamedReference");
	} else {
	  std::cout << "Data read on oid="<<oid2<<" is correct."<<std::endl;
	}
      } catch (ora::Exception& e ){
	std::cout <<"****2 Exception: "<<e.what()<<std::endl;
      }
      std::vector<std::string> o2Names;
      if(!db.getNamesForObject( ora::OId( contH1.id(),oid2 ), o2Names)){
	std::cout << "** No names found for object cid="<<contH1.id()<<" oid="<<oid2<<std::endl;
      } else {
	std::cout << "** Found "<<o2Names.size()<<" names for object cid="<<contH1.id()<<" oid="<<oid2<<std::endl;
	for( std::vector<std::string>::const_iterator iN = o2Names.begin();
	     iN != o2Names.end(); ++iN ){
	  std::cout << "** Name: "<<*iN<<std::endl;
	}
      }
      {
	boost::shared_ptr<SH> sr2_1 = contH1.fetch<SH>( oid2);
	std::cout << "** ptr2_0="<<sr2->m_ref.get()<<std::endl;
	std::cout << "** ptr2_1="<<sr2_1->m_ref.get()<<std::endl;
      }
      sr2->m_ref.reset();
      {  
	boost::shared_ptr<SH> sr2_2 = contH1.fetch<SH>( oid2);
	boost::shared_ptr<SH> sr2_3 = contH1.fetch<SH>( oid2);
	std::cout << "** ptr2_2="<<sr2_2->m_ref.get()<<std::endl;
	std::cout << "** ptr2_3="<<sr2_3->m_ref.get()<<std::endl;
      }
      
      trans1.commit();
      trans1.start( false );
      db.drop();
      trans1.commit();
      db.disconnect();
    }
  };
}

int main(int argc, char** argv ){
  ora::TestNamedReference test;
  test.run();
}

