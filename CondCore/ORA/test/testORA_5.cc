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
  //std::string connStr( "myalias" );
  std::string connStr( "oracle://devdb10/giacomo" );
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
  db.createContainer<MultiArrayClass2>("Cont0");
  ora::Container contH0 = db.containerHandle( "Cont0" );
  std::cout << "** W start objs in container="<<contH0.size()<<std::endl;
  int oid0, oid1;
  {
    MultiArrayClass2 a0(2);
    //a0.print();
    oid0 = contH0.insert( a0 );
    MultiArrayClass2 a1(3);
    oid1 = contH0.insert( a1 );
    std::cout << "** W start objs in container bef flush="<<contH0.size()<<std::endl;
    contH0.flush();
    std::cout << "** W start objs in container aft flush="<<contH0.size()<<std::endl;
  }
  ora::OId id0( contH0.id(), oid0 );
  ora::OId id1( contH0.id(), oid1 );
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
  std::cout << "** R start objs in container="<<contHR0.size()<<std::endl;
  boost::shared_ptr<MultiArrayClass2> ar0 = contHR0.fetch<MultiArrayClass2 >( oid0 );
  if( ar0 ){
    std::cout << "Read out vector size="<<ar0->m_a.size()<<std::endl;
    ar0->print();
  } else {
    std::cout << "############# No data for oid="<<oid0<<std::endl;
  }
  boost::shared_ptr<MultiArrayClass2> ar1 = contHR0.fetch<MultiArrayClass2 >( oid1 );
  if( ar1 ){
    std::cout << "Read out vector size="<<ar1->m_a.size()<<std::endl;
    ar1->print();
  } else {
    std::cout << "############# No data for oid="<<oid1<<std::endl;
  } 
  db.transaction().commit();
  db.disconnect();
// update...
  std::cout << "++++++++++++ update +++++++++++"<<std::endl;
  db.connect( connStr );
  db.transaction().start( false );
  MultiArrayClass2 au0(3);
  std::cout << "update oid="<<id0.itemId()<<std::endl;
  db.update( id0, au0 );
  MultiArrayClass2* au1 = new MultiArrayClass2(4);
  std::cout << "update oid="<<id1.itemId()<<std::endl;
  db.update( id1, *au1 );
  db.flush();
  delete au1;
  ora::OId id2( id1.containerId(), 400 );
  std::cout << "update oid="<<id2.itemId()<<std::endl;
  db.update( id2, au0 );
  db.transaction().commit();
  db.disconnect();
  // reading back...
  std::cout << "++++++++++++ Reading II +++++++++++"<<std::endl;
  db.connect( connStr );  
  db.transaction().start( true );
  ar0 = db.fetch<MultiArrayClass2 >( id0 );
  if( ar0 ){
    std::cout << "Read after update vector size="<<ar0->m_a.size()<<std::endl;
    ar0->print();
  } else {
    std::cout << "############# No data for oid="<<id0.itemId()<<std::endl;
  }
  ar1 = db.fetch<MultiArrayClass2 >( id1 );
  if( ar1 ){
    std::cout << "Read after update vector size="<<ar1->m_a.size()<<std::endl;
    ar1->print();
  } else {
    std::cout << "############# No data for oid="<<id1.itemId()<<std::endl;
  } 
  db.transaction().commit();
  db.disconnect();
  // delete...
  std::cout << "++++++++++++ delete +++++++++++"<<std::endl;
  db.connect( connStr );
  db.transaction().start( false );
  contHR0 = db.containerHandle( "Cont0" );
  std::cout << "** start objs in container="<<contHR0.size()<<std::endl;
  std::cout << "delete oid="<<id0.itemId()<<std::endl;
  db.erase( id0 );
  std::cout << "delete oid="<<id2.itemId()<<std::endl;
  db.erase( id2 );
  std::cout << "** objs in container before flush="<<contHR0.size()<<std::endl;
  db.flush();
  std::cout << "** objs in container after flush="<<contHR0.size()<<std::endl;
  db.transaction().commit();
  db.disconnect();
  db.connect( connStr );  
  db.transaction().start( true );
  contHR0 = db.containerHandle( "Cont0" );
  std::cout << "** start objs in container="<<contHR0.size()<<std::endl;
  db.transaction().commit();
  db.disconnect();
  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  }
}

