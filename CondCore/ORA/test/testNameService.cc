#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/ScopedTransaction.h"
#include "CondCore/ORA/test/Serializer.h"
#include <cstdlib>
#include <iostream>
#include "classes.h"

int main(){
  using namespace testORA;
  ora::Database db;
  try {

    // writing...
    std::string authpath("/afs/cern.ch/cms/DB/conddb");
    std::string pathenv(std::string("CORAL_AUTH_PATH=")+authpath);
    ::putenv(const_cast<char*>(pathenv.c_str()));
    //std::string connStr( "oracle://cms_orcoff_prep/CMS_COND_UNIT_TESTS" );
    std::string connStr( "sqlite_file:test.db" );
    //ora::Serializer serializer( "ORA_TEST" );
    //serializer.lock( connStr );
    db.configuration().setMessageVerbosity( coral::Debug );
    db.connect( connStr );
    ora::ScopedTransaction trans0( db.transaction() );
    trans0.start( false );
    if(!db.exists()){
      db.create();
    }
    std::set< std::string > conts = db.containers();
    if( conts.find( "Cont0" )!= conts.end() ) db.dropContainer( "Cont0" );
    // ***
    ora::Container contH0 = db.createContainer<SimpleClass>("Cont0");
    SimpleClass s0(4);
    int oid0 = contH0.insert( s0 );
    SimpleClass s1(999);
    int oid1 = contH0.insert( s1 );
    contH0.flush();
    std::cout << "******* setting name..."<<std::endl; 
    db.setObjectName( "Peeppino",ora::OId(contH0.id(),oid1)); 
    trans0.commit();
    db.disconnect();
    // reading back...
    ::sleep(1);
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
    //boost::shared_ptr<SimpleClass> sr1 = contH0.fetch<SimpleClass>( oid1);
    boost::shared_ptr<SimpleClass> sr1 = db.fetchByName<SimpleClass>( "Peeppino" ); 
    if( *sr1 != s1 ){
      ora::throwException( "Data read on oid1 different from expected.","testORA_1");
    } else {
      std::cout << "Data read on oid="<<oid1<<" is correct."<<std::endl;
    }
    trans1.commit();
    trans1.start( false );
    //db.drop();
    trans1.commit();
    db.disconnect();
  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  }
}

