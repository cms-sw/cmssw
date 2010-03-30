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
    std::string connStr( "oracle://devdb10/giacomo" );
    //std::string connStr( "sqlite_file:test.db" );
  db.connect( connStr );
  db.transaction().start( false );
  bool exists = db.exists();
  if(exists){
    std::cout << "############# ORA database does exist in "<< connStr<<"."<<std::endl;
    db.dropContainer( "Cont0" );
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
  db.createContainer<MultiArrayClass>("Cont0");
  ora::Container contH0 = db.containerHandle( "Cont0" );
  MultiArrayClass a0(10);
  a0.print();
  int oid0 = contH0.insert( a0 );
  MultiArrayClass a1(20);
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
  ora::Container contHR0 = db.containerHandle( "Cont0" );
  std::cout << "############# ContID="<<contHR0.id()<<std::endl;
  boost::shared_ptr<MultiArrayClass> ar0 = contHR0.fetch<MultiArrayClass >( oid0 );
  if( ar0 ){
    std::cout << "Read out vector size="<<ar0->m_a.size()<<std::endl;
    ar0->print();
  } else {
    std::cout << "############# No data for oid="<<oid0<<std::endl;
  }
  ora::OId foid1( contHR0.id(), oid1 );
  ora::Object r1 = db.fetchItem( foid1 );
  MultiArrayClass* ar1 = r1.cast<MultiArrayClass>();
  if( ar1 ){
    std::cout << "Read out vector size="<<ar1->m_a.size()<<std::endl;
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

