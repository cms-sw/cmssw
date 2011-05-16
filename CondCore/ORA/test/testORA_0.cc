#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/ScopedTransaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/test/Serializer.h"
#include <cstdlib>
#include <iostream>

int main( int argc, char** argv){
  // writing...
  std::string authpath("/afs/cern.ch/cms/DB/conddb");
  std::string pathenv(std::string("CORAL_AUTH_PATH=")+authpath);
  ::putenv(const_cast<char*>(pathenv.c_str()));
  try {

    std::string connStr( "oracle://cms_orcoff_prep/CMS_COND_UNIT_TESTS" );
    //std::string connStr( "sqlite_file:test.db" );
    ora::Serializer serializer( "ORA_TEST" );
    serializer.lock( connStr, std::string(argv[0]) );
    ora::Database db;
    //db.configuration().setMessageVerbosity( coral::Debug );
    db.connect( connStr );
    ora::ScopedTransaction trans0( db.transaction() );
    trans0.start( false );
    bool exists = db.exists();
    if(exists){
      db.drop();
    }
    db.create();
    std::set< std::string > conts = db.containers();
    std::cout << "#### creating containers..."<<std::endl;
    if( conts.find( "Cont0" )!= conts.end() ) db.dropContainer( "Cont0" );
    if( conts.find( "std::string" )!= conts.end() ) db.dropContainer( "std::string" );
    db.createContainer<int>("Cont0");
    int contId = db.createContainer<std::string>().id();
    //**
    std::cout << "#### writing..."<<std::endl;
    ora::Container contH0 = db.containerHandle( "Cont0" );
    int myInt0(999);
    int myInt1(1234567890);
    int oid00 = contH0.insert( myInt0 );
    int oid01 = contH0.insert( myInt1 );
    contH0.flush();
    //**
    ora::Container contH1 = db.containerHandle( contId );
    std::string myStr0("ABCDEFGHILMNOPQRSTUVZ1234567890");
    std::string myStr1("BlaBlaBlaBla");
    int oid10 = contH1.insert( myStr0 );
    int oid11 = contH1.insert( myStr1 );
    contH1.flush();
    //**
    trans0.commit();
    db.disconnect();
    ::sleep(1);
    db.connect( connStr );
    ora::ScopedTransaction trans1( db.transaction() );
    trans1.start( true );
    bool exists2 = db.exists();
    if(!exists2){
      ora::throwException( "ORA database does not exist in "+connStr,"testORA_0");
    }
    conts = db.containers();
    if( conts.find( "Cont0" )== conts.end() ) {
      ora::throwException( "Container Cont0 has not been found.","testORA_0");
    }
    std::cout << "#### reading..."<<std::endl;
    contH0 = db.containerHandle( "Cont0" );
    boost::shared_ptr<int> rInt0 = contH0.fetch<int>( oid00);
    if( *rInt0 != myInt0 ){
      ora::throwException( "Data read on oid00 different from expected.","testORA_0");      
    } else {
      std::cout << "Data read on oid="<<oid00<<" is correct."<<std::endl;
    }
    boost::shared_ptr<int> rInt1 = contH0.fetch<int>( oid01);
    if( *rInt1 != myInt1 ){
      ora::throwException( "Data read on oid01 different from expected.","testORA_0");      
    } else {
      std::cout << "Data read on oid="<<oid00<<" is correct."<<std::endl;
    } 
    //**
    if( conts.find( "std::string" )== conts.end() ) {
      ora::throwException( "Container std::string has not been found.","testORA_0");
    }
    contH1  = db.containerHandle( contId );
    boost::shared_ptr<std::string> rStr0 = contH1.fetch<std::string>( oid10 );
    if( *rStr0 != myStr0 ){
      ora::throwException( "Data read on oid10 different from expected.","testORA_0");      
    } else {
      std::cout << "Data read on oid="<<oid10<<" is correct."<<std::endl;
    }
    boost::shared_ptr<std::string> rStr1 = contH1.fetch<std::string>( oid11 );
    if( *rStr1 != myStr1 ){
      ora::throwException( "Data read on oid11 different from expected.","testORA_0");      
    } else {
      std::cout << "Data read on oid="<<oid11<<" is correct."<<std::endl;
    }
    trans1.commit();
    db.disconnect();
    //***
    std::cout << "#### updating..."<<std::endl;
    db.connect( connStr );  
    ora::ScopedTransaction trans2( db.transaction() );
    trans2.start( false );
    contH0 = db.containerHandle( "Cont0" );
    int nInt0(888);
    contH0.update( oid00, nInt0 );
    int nInt1(987654321);
    contH0.update( oid01, nInt1 );
    contH0.flush();
    rInt0 = contH0.fetch<int>( oid00);
    if( *rInt0 != nInt0 ){
      ora::throwException( "Data read after update (bc) on oid00 different from expected.","testORA_0");      
    } else {
      std::cout << "Data read after update (bc) on oid="<<oid00<<" is correct."<<std::endl;
    }
    rInt1 = contH0.fetch<int>( oid01);
    if( *rInt1 != nInt1 ){
      ora::throwException( "Data read after update (bc) on oid01 different from expected.","testORA_0");      
    } else {
      std::cout << "Data read after update on (bc) oid="<<oid01<<" is correct."<<std::endl;
    }
    contH1 = db.containerHandle( contId );
    std::string nStr0("0987654321abcdefghilmnopqrstuvz");
    contH1.update( oid10, nStr0 );
    std::string nStr1("PincoPallino");
    contH1.update( oid11, nStr1 );
    contH1.flush();
    rStr0 = contH1.fetch<std::string>( oid10 );
    if( *rStr0 != nStr0 ){
      ora::throwException( "Data read after update (bc) on oid10 different from expected.","testORA_0");      
    } else {
      std::cout << "Data read after update (bc) on oid="<<oid10<<" is correct."<<std::endl;
    }
    rStr1 = contH1.fetch<std::string>( oid11 );
    if( *rStr1 != nStr1 ){
      ora::throwException( "Data read after update (bc) on oid11 different from expected.","testORA_0");      
    } else {
      std::cout << "Data read after update (bc) on oid="<<oid11<<" is correct."<<std::endl;
    }
    trans2.commit();
    db.disconnect();
    //**
    std::cout << "#### reading after update..."<<std::endl;
    db.connect( connStr );
    ora::ScopedTransaction trans3( db.transaction() );
    trans3.start( true );
    contH0 = db.containerHandle( "Cont0" );
    rInt0 = contH0.fetch<int>( oid00);
    if( *rInt0 != nInt0 ){
      ora::throwException( "Data read after update on oid00 different from expected.","testORA_0");      
    } else {
      std::cout << "Data read after update on oid="<<oid00<<" is correct."<<std::endl;
    }
    rInt1 = contH0.fetch<int>( oid01);
    if( *rInt1 != nInt1 ){
      ora::throwException( "Data read after update on oid01 different from expected.","testORA_0");      
    } else {
      std::cout << "Data read after update on oid="<<oid01<<" is correct."<<std::endl;
    }
    contH1 = db.containerHandle( contId );
    rStr0 = contH1.fetch<std::string>( oid10 );
    if( *rStr0 != nStr0 ){
      ora::throwException( "Data read after update on oid10 different from expected.","testORA_0");      
    } else {
      std::cout << "Data read after update on oid="<<oid10<<" is correct."<<std::endl;
    }
    rStr1 = contH1.fetch<std::string>( oid11 );
    if( *rStr1 != nStr1 ){
      ora::throwException( "Data read after update on oid11 different from expected.","testORA_0");      
    } else {
      std::cout << "Data read after update on oid="<<oid11<<" is correct."<<std::endl;
    }
    //*
    ora::ContainerIterator iter0 = contH0.iterator();
    while( iter0.next() ){
      boost::shared_ptr<int> o = iter0.get<int>();
      std::cout << " **** Cont="<<contH0.name()<<" val="<<*o<<std::endl;
    }
    ora::ContainerIterator iter1 = contH1.iterator();
    while( iter1.next() ){
      boost::shared_ptr<std::string> s = iter1.get<std::string>();
      std::cout << " **** Cont="<<contH1.name()<<" val="<<*s<<std::endl;
    }
    trans3.commit();
    try {
      rStr1 = contH1.fetch<std::string>( oid11 );
    } catch (ora::Exception& e){
      std::cout << "*** expected exception:"<<e.what()<<std::endl;
    }
    db.disconnect();
    try {
      boost::shared_ptr<std::string> null = iter1.get<std::string>();
    } catch (ora::Exception& e){
      std::cout << "*** expected exception:"<<e.what()<<std::endl;
    }
    std::cout << "#### deleting..."<<std::endl;
    db.connect( connStr );
    ora::ScopedTransaction trans4( db.transaction() );
    trans4.start( false );
    ora::OId foid00( db.containerHandle( "Cont0" ).id(), oid00);
    db.erase( foid00 );
    db.containerHandle( contId ).erase( oid11 );
    db.flush();
    trans4.commit();
    db.disconnect();
    std::cout << "#### reading after delete..."<<std::endl;
    db.connect( connStr );  
    ora::ScopedTransaction trans5( db.transaction() );
    trans5.start( false );
    contH0 = db.containerHandle( "Cont0" );
    rInt0 = contH0.fetch<int>( oid00);
    if( rInt0 ){
      ora::throwException( "Found entry for deleted oid="+oid00,"testORA_0");
    } else {
      std::cout << "## Entry for oid="<<oid00<<" deleted as expected."<<std::endl;
    }
    rInt1 = contH0.fetch<int>( oid01);
    if( rInt1 ){
      std::cout << "## Entry for oid="<<oid01<<" found as expected."<<std::endl;
    } else {
      ora::throwException( "Entry for oid01 not found.","testORA_0");
    }
    contH1 = db.containerHandle( contId );
    rStr0 = contH1.fetch<std::string>( oid10);
    if( rStr0 ){
      std::cout << "## Entry for oid="<<oid10<<" found as expected."<<std::endl;
    } else {
      ora::throwException( "Entry for oid10 not found.","testORA_0");
    }
    rStr1 = contH1.fetch<std::string>( oid11);
    if( rStr1 ){
      ora::throwException( "Found entry for deleted oid="+oid11,"testORA_0");
    } else {
      std::cout << "## Entry for oid="<<oid11<<" deleted as expected."<<std::endl;
    }
    iter0 = contH0.iterator();
    while( iter0.next() ){
      boost::shared_ptr<int> o = iter0.get<int>();
      std::cout << " **** Cont="<<contH0.name()<<" val="<<*o<<std::endl;
    }
    iter1 = contH1.iterator();
    while( iter1.next() ){
      boost::shared_ptr<std::string> s = iter1.get<std::string>();
      std::cout << " **** Cont="<<contH1.name()<<" val="<<*s<<std::endl;
    }
    std::cout <<"#### Database schema version="<<db.schemaVersion()<<std::endl;
    db.drop();
    trans5.commit();
    db.disconnect();
    serializer.release();


  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  }
}

