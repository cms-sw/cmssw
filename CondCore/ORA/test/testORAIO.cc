#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/ScopedTransaction.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/test/Serializer.h"
#include <cstdlib>
#include <iostream>
#include "classes.h"

int main(){
  using namespace testORA;
  try {

    // writing...  
    std::string authpath("/afs/cern.ch/cms/DB/conddb");
    std::string pathenv(std::string("CORAL_AUTH_PATH=")+authpath);
    ::putenv(const_cast<char*>(pathenv.c_str()));
    ora::Database db;
    db.configuration().setMessageVerbosity( coral::Debug );
    std::string connStr( "oracle://cms_orcoff_prep/CMS_COND_UNIT_TESTS" );
    std::vector<int> oids;
    ora::Serializer serializer( "ORA_TEST" );
    serializer.lock( connStr );
    db.connect( connStr );
    ora::ScopedTransaction trans0( db.transaction() );
    trans0.start( false );
    //
    if(!db.exists()){
      db.create();
    }
    std::set< std::string > conts = db.containers();
    if( conts.find( "Cont0" )!= conts.end() ) db.dropContainer( "Cont0" );
    if( conts.find( "IOV" )!= conts.end() ) db.dropContainer( "IOV" );
    //
    ora::Container contIOV = db.createContainer<IOV>();
    ora::Container contH0 = db.createContainer<ArrayClass>("Cont0");
    for( int j = 0; j!= 2; j++ ){
      IOV iov;
      oids.push_back( contIOV.insert( iov ) );
      for( int i=0; i!=2; i++){
        ArrayClass a(i,10000);
        iov.oids.push_back( ora::OId( contH0.id(),contH0.insert(a) ) );
        contH0.flush();
      }
      contIOV.flush();
    }
    //
    trans0.commit();
    db.disconnect();
    ::sleep(1);
    // opening dest db
    std::cout << "** creating dest db"<<std::endl;
    ora::Database db2;
    db2.configuration().setMessageVerbosity( coral::Debug );
    std::string connStr2( "sqlite_file:test.db" );
    db2.connect( connStr2 );
    ora::ScopedTransaction trans2( db2.transaction() );
    trans2.start( false );
    if(!db2.exists()){
      db2.create();
    }
    conts = db2.containers();
    if( conts.find( "Cont0" )!= conts.end() ) db2.dropContainer( "Cont0" );
    if( conts.find( "IOV" )!= conts.end() ) db2.dropContainer( "IOV" );
    ora::Container contH2 = db2.createContainer<ArrayClass>("Cont0");
    ora::Container contIOV2 = db2.createContainer<IOV>();
    // reading back...
    std::cout << "** exporting in dest db"<<std::endl;
    db.connect( connStr );
    trans0.start( true );
    contIOV = db.containerHandle( "IOV" );
    for( std::vector<int>::const_iterator iO = oids.begin(); iO!= oids.end(); ++iO ){
      IOV iov;
      int iovOid = contIOV2.insert( iov );
      contIOV2.flush();
      boost::shared_ptr<IOV> riov = contIOV.fetch<IOV>( *iO );
      for( std::vector<ora::OId>::const_iterator iP = riov->oids.begin();
           iP != riov->oids.end(); ++iP ){
        boost::shared_ptr<ArrayClass> r = db.fetch<ArrayClass>( *iP );
        iov.oids.push_back( ora::OId( contH2.id(), contH2.insert( *r )) );
        contH2.flush();
        contIOV2.update( iovOid, iov );
        contIOV2.flush();
      }
    }
    trans0.commit();
    trans2.commit();
    db2.disconnect();
    std::cout << "** disconnecting dest db"<<std::endl;
    trans0.start( false );
    std::cout << "** dropping source db"<<std::endl;
    db.drop();
    trans0.commit();
    std::cout << "** disconnecting source db"<<std::endl;
    db.disconnect();
  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  }
}

