#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include <iostream>
//#include <typeinfo>
//#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "classes.h"

int main(){
  try {

    //const boost::filesystem::path dict_path("testORADict");
    //edmplugin::SharedLibrary shared( dict_path );
    // writing...  
    ora::Database db;
    db.configuration().setMessageVerbosity( coral::Debug );
    //std::string connStr( "sqlite_file:test.db" );
  std::string connStr( "oracle://devdb10/giacomo" );
  db.connect( connStr );
  db.transaction().start( false );
  bool exists = db.exists();
  if(exists){
    std::cout << "############# ORA database does exist in "<< connStr<<"."<<std::endl;
    db.dropContainer("Cont0_ABCDEFGHILMNOPQRSTUVZ");
    db.dropContainer("ABDC");
  } else {
    std::cout << "############# ORA database does not exist in "<< connStr<<", creating it..."<<std::endl;
    db.create();
  }
  std::set< std::string > conts = db.containers();
  std::cout << conts.size() <<" ############# container(s) found."<<std::endl;
  for(std::set<std::string>::const_iterator iC = conts.begin();
      iC != conts.end(); iC++ ){
    std::cout << "############# CONT=\""<<*iC<<"\""<<std::endl;
  }
  ora::Container contH0 = db.createContainer<ArrayClass>("ABDC");
  //db.createContainer<ArrayClass>("Cont0_ABCDEFGHILMNOPQRSTUVZ");
  //ora::Container contH0 = db.containerHandle( "Cont0_ABCDEFGHILMNOPQRSTUVZ" );
  ArrayClass a0(10);
  std::cout << "######### A0 map size="<<a0.m_map.size()<<std::endl;
  a0.print();
  int oid0 = contH0.insert( a0 );
  ArrayClass a1(20);
  std::cout << "######### A1 map size="<<a0.m_map.size()<<std::endl;
  int oid1 = contH0.insert( a1 );
  contH0.flush();
  db.transaction().commit();
  db.disconnect();
  ::sleep(1);
  // reading back...
  db.connect( connStr );  
  db.transaction().start( true );
  exists = db.exists();
  if(exists){
    std::cout << "############# ORA database does exist in "<< connStr<<"."<<std::endl;
  } else {
    std::cout << "############# ERROR!! ORA database does not exist in "<< connStr<<std::endl;
  }
  conts = db.containers();
  std::cout << conts.size() <<" ############# container(s) found."<<std::endl;
  for(std::set<std::string>::const_iterator iC = conts.begin();
      iC != conts.end(); iC++ ){
    std::cout << "############# CONT=\""<<*iC<<"\""<<std::endl;
  }
  //ora::Container contHR0 = db.containerHandle( "Cont0_ABCDEFGHILMNOPQRSTUVZ" );
  ora::Container contHR0 = db.containerHandle( "ABDC" );
  std::cout << "############# ContID="<<contHR0.id()<<std::endl;
  ora::Object r0 = contHR0.fetchItem( oid0 );
  ArrayClass* ar0 = r0.cast<ArrayClass>();
  if( ar0 ){
    std::cout << "Read out vector size="<<ar0->m_arrayData.size()<<std::endl;
    ar0->print();
    r0.destruct();
  } else {
    std::cout << "############# No data for oid="<<oid0<<std::endl;
  }
  ora::OId foid1( contHR0.id(), oid1 );
  ora::Object r1 = db.fetchItem( foid1 );
  ArrayClass* ar1 = r1.cast<ArrayClass >();
  if( ar1 ){
    std::cout << "Read out vector size="<<ar1->m_arrayData.size()<<std::endl;
    ar1->print();
    r1.destruct();
  } else {
    std::cout << "############# No data for oid="<<oid1<<std::endl;
  } 
  db.transaction().commit();
  db.disconnect();
  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  }
}

