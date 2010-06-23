#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/ScopedTransaction.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include <iostream>
#include "classes.h"

int main(){
  try {

    // writing...  
    ora::Database db;
    db.configuration().setMessageVerbosity( coral::Debug );
    //std::string connStr( "sqlite_file:test.db" );
    std::string connStr( "oracle://devdb10/giacomo" );
    db.connect( connStr );
    ora::ScopedTransaction trans0( db.transaction() );
    trans0.start( false );
    //
    if(!db.exists()){
      db.create();
    }
    std::set< std::string > conts = db.containers();
    if( conts.find( "Cont0_ABCDEFGHILMNOPQRSTUVZ" )!= conts.end() ) db.dropContainer( "Cont0_ABCDEFGHILMNOPQRSTUVZ" );
    //
    ora::Container contH0 = db.createContainer<ArrayClass>("Cont0_ABCDEFGHILMNOPQRSTUVZ");
    ArrayClass a0(10);
    int oid0 = contH0.insert( a0 );
    ArrayClass a1(20);
    int oid1 = contH0.insert( a1 );
    contH0.flush();
    //
    trans0.commit();
    db.disconnect();
    ::sleep(1);
    // reading back...
    db.connect( connStr );
    trans0.start( true );
    contH0 = db.containerHandle( "Cont0_ABCDEFGHILMNOPQRSTUVZ" );
    ora::Object r0 = contH0.fetchItem( oid0 );
    ArrayClass* ar0 = r0.cast<ArrayClass>();
    if( *ar0 != a0 ){
      ora::throwException( "Data read on oid0 different from expected.","testORA_3");
    } else {
      std::cout << "Data read on oid="<<oid0<<" is correct."<<std::endl;
    }
    r0.destruct();
    ora::OId foid1( contH0.id(), oid1 );
    ora::Object r1 = db.fetchItem( foid1 );
    ArrayClass* ar1 = r1.cast<ArrayClass >();
    if( *ar1 != a1 ){
      ora::throwException( "Data read on oid1 different from expected.","testORA_3");
    } else {
      std::cout << "Data read on oid="<<oid1<<" is correct."<<std::endl;
    }
    r1.destruct();
    trans0.commit();
    db.disconnect();
  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  }
}

