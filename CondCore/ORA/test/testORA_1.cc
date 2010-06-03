#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include <iostream>
//#include <typeinfo>
//#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "classes.h"

int main(){
  ora::Database db;
  try {

    //const boost::filesystem::path dict_path("testORADict");
    //edmplugin::SharedLibrary shared( dict_path );
    // writing...  
  std::string connStr( "oracle://devdb10/giacomo" );
  //std::string connStr( "sqlite_file:test.db" );
  db.configuration().setMessageVerbosity( coral::Debug );
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
  std::cout <<" ############# Creating container Cont0."<<std::endl;
  db.createContainer<SimpleClass>("Cont0");
  std::cout <<" ############# Created container Cont0."<<std::endl;
  ora::Container contH0 = db.containerHandle( "Cont0" );
  SimpleClass s0(4);
  int oid0 = contH0.insert( s0 );
  SimpleClass s1(999);
  int oid01 = contH0.insert( s1 );
  contH0.flush();
  db.transaction().commit();
  db.disconnect();
  // reading back...
  ::sleep(1);
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
  boost::shared_ptr<SimpleClass> sr0 = contHR0.fetch<SimpleClass>( oid0);
  if( sr0 ){
    std::cout << "############# Read out data0=";
    sr0->print();
  } else {
    std::cout << "############# No data for oid="<<oid0<<std::endl;
  } 
  boost::shared_ptr<SimpleClass> sr1 = contHR0.fetch<SimpleClass>( oid01);
  if( sr0 ){
    std::cout << "############# Read out data1=";
    sr1->print();
  } else {
    std::cout << "############# No data for oid="<<oid01<<std::endl;
  } 
  db.transaction().commit();
  db.disconnect();
  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  db.transaction().commit();
  }
}

