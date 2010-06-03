#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include <iostream>
#include "classes.h"

int main(){
  ora::Database db;
  try {

  // writing...  
  //std::string connStr( "oracle://devdb10/giacomo" );
  std::string connStr0( "sqlite_file:test0.db" );
  std::string connStr1( "sqlite_file:test1.db" );
  std::string connStr2( "sqlite_file:test2.db" );
  db.configuration().setMessageVerbosity( coral::Debug );
  db.connect( connStr0 );
  db.transaction().start( false );
  bool exists = db.exists();
  if(exists){
    std::cout << "############# ORA database does exist in "<< connStr0<<"."<<std::endl;
    db.dropContainer( "Cont0" );
  }
  else {
    std::cout << "############# ORA database does not exist in "<< connStr0<<", creating it..."<<std::endl;
    db.create();
  }
  std::set< std::string > conts = db.containers();
  std::cout << conts.size() <<" ############# container(s) found."<<std::endl;
  for(std::set<std::string>::const_iterator iC = conts.begin();
      iC != conts.end(); iC++ ){
    std::cout << "############# CONT=\""<<*iC<<"\""<<std::endl;
  }
  db.createContainer<SimpleClass>("Cont0");
  ora::Container contH0 = db.containerHandle( "Cont0" );
  SimpleClass s0(4);
  int oid0 = contH0.insert( s0 );
  SimpleClass s1(999);
  int oid01 = contH0.insert( s1 );
  contH0.flush();
  db.transaction().commit();
  db.disconnect();
  ::sleep(1);
  db.connect( connStr1 );
  db.transaction().start( false );
  if(db.exists()){
    std::cout << "############# ORA database does exist in "<< connStr1<<"."<<std::endl;
    db.dropContainer( "Cont0" );
  }
  else {
    std::cout << "############# ORA database does not exist in "<< connStr1<<", creating it..."<<std::endl;
    db.create();
  }
  ora::DatabaseUtility util = db.utility();
  std::cout << "*** importing cont..."<<std::endl;
  util.importContainerSchema( connStr0, "Cont0" );
  contH0 = db.containerHandle( "Cont0" );
  SimpleClass s01(4);
  oid0 = contH0.insert( s01 );
  SimpleClass s11(999);
  oid01 = contH0.insert( s11 );
  contH0.flush();  
  db.transaction().commit();
  db.disconnect();
  try {
    util.listMappingVersions( "Cont0" );
  } catch ( ora::Exception& e ){
    std::cout << "## exception: "<<e.what()<<std::endl;
  }
  // reading back...
  db.connect( connStr1 );  
  db.transaction().start( true );
  exists = db.exists();
  if(exists){
    std::cout << "############# ORA database does exist in "<< connStr1<<"."<<std::endl;
  } else {
    std::cout << "############# ERROR!! ORA database does not exist in "<< connStr1<<std::endl;
  }
  conts = db.containers();
  std::cout << conts.size() <<" ############# container(s) found."<<std::endl;
  for(std::set<std::string>::const_iterator iC = conts.begin();
      iC != conts.end(); iC++ ){
    std::cout << "############# CONT=\""<<*iC<<"\""<<std::endl;
  }
  util = db.utility();
  std::set<std::string> vers = util.listMappingVersions( "Cont0" );
  for(std::set<std::string>::const_iterator iV = vers.begin();
      iV != vers.end(); iV++ ){
    std::cout << "======= VERS=\""<<*iV<<"\""<<std::endl;
  }  
  ora::Container contHR0 = db.containerHandle( "Cont0" );
  std::cout << "############# ContID="<<contHR0.id()<<std::endl;
  boost::shared_ptr<SimpleClass> sr0 = contHR0.fetch<SimpleClass>( oid0);
  if( sr0 ){
    std::cout << "############# (0) Read out data0=";
    sr0->print();
  } else {
    std::cout << "############# (0) No data for oid="<<oid0<<std::endl;
  } 
  boost::shared_ptr<SimpleClass> sr1 = contHR0.fetch<SimpleClass>( oid01);
  if( sr0 ){
    std::cout << "############# (0) Read out data1=";
    sr1->print();
  } else {
    std::cout << "############# (0) No data for oid="<<oid01<<std::endl;
  } 
  db.transaction().commit();
  db.disconnect();
  db.configuration().properties().setFlag( ora::Configuration::automaticDatabaseCreation() );
  db.connect( connStr2 );
  db.transaction().start( false );
  std::cout << "############# (0) getting utility..."<<std::endl;
  util = db.utility();
  std::cout << "############# (0) importing..."<<std::endl;
  util.importContainer( connStr1, "Cont0" );
  db.transaction().commit();
  db.disconnect(); 
  // reading back...
  db.connect( connStr2 );  
  db.transaction().start( true );
  exists = db.exists();
  if(exists){
    std::cout << "############# ORA database does exist in "<< connStr2<<"."<<std::endl;
  } else {
    std::cout << "############# ERROR!! ORA database does not exist in "<< connStr2<<std::endl;
  }
  conts = db.containers();
  std::cout << conts.size() <<" ############# container(s) found."<<std::endl;
  for(std::set<std::string>::const_iterator iC = conts.begin();
      iC != conts.end(); iC++ ){
    std::cout << "############# CONT=\""<<*iC<<"\""<<std::endl;
  }
  contHR0 = db.containerHandle( "Cont0" );
  std::cout << "############# ContID="<<contHR0.id()<<std::endl;
  sr0 = contHR0.fetch<SimpleClass>( oid0);
  if( sr0 ){
    std::cout << "############# (1) Read out data0=";
    sr0->print();
  } else {
    std::cout << "############# (1) No data for oid="<<oid0<<std::endl;
  } 
  sr1 = contHR0.fetch<SimpleClass>( oid01);
  if( sr0 ){
    std::cout << "############# (1) Read out data1=";
    sr1->print();
  } else {
    std::cout << "############# (1) No data for oid="<<oid01<<std::endl;
  } 
  db.transaction().commit();
  db.disconnect();
  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  }
}

