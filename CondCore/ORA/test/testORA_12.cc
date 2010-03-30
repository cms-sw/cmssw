#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/IReferenceHandler.h"
#include <iostream>

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
  ora::Container cont0 = db.createContainer<SG>("Cont0");
  std::cout << "#######  EXTENDING SCHEMA ** "<<std::endl;
  cont0.extendSchema<D0>();
  std::cout << "#######  DONE EXTENDING ** "<<std::endl;
  std::vector<boost::shared_ptr<SG> > buff0;
  for( unsigned int i = 0; i<10; i++){
    boost::shared_ptr<SG> data( new SG( i ) );
    db.insert( "Cont0", *data );
    buff0.push_back( data );
    data->m_ref = new D0(i);
    data->m_ref2 = new D0(i+10);
  }
  db.flush();
  buff0.clear();
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
  cont0 = db.containerHandle( "Cont0" );
  std::cout << "############# ContID="<<cont0.id()<<std::endl;
  std::cout << "** R start objs in container="<<cont0.size()<<std::endl;
  ora::ContainerIterator iter = cont0.iterator();
  while( iter.next() ){
    boost::shared_ptr<SG> obj = iter.get<SG>();    
    unsigned int seed = obj->m_intData;
    
    SG r(seed);
    r.m_ref = new D0(seed);
    r.m_ref2 = new D0(seed+10);

    if( r != *obj ){
      std::cout <<"** test ERROR: data for class SG different from expected for seed = "<<seed<<std::endl;
    } else{
      std::cout << "** Read out data for class SG with seed="<<seed<<" is ok."<<std::endl;
    }
  }
  db.transaction().commit();
  db.disconnect();
  db.disconnect();
  std::cout << "************** writing more data..."<<std::endl;
  db.configuration().properties().setFlag( ora::Configuration::automaticContainerCreation() );
  db.connect( connStr );
  db.transaction().start( false );
  std::vector<ora::OId> oids;
  for( unsigned int i = 0; i<10; i++){
    boost::shared_ptr<SG> data( new SG( i ) );
    oids.push_back( db.insert( "Cont0", *data ) );
    buff0.push_back( data );
    data->m_ref = new D1(i);
    data->m_ref2 = new D2(i);
  }
  db.flush();
  buff0.clear();
  db.transaction().commit();
  db.disconnect();
  ::sleep(1);
  db.connect( connStr );  
  db.transaction().start( true );
  for( std::vector<ora::OId>::iterator iO = oids.begin();
       iO != oids.end(); ++iO ){
    std::string soid = iO->toString();
    std::cout <<" ** cid="<<iO->containerId()<<" iid="<<iO->itemId()<<" soid="<< soid<<std::endl;
    ora::OId oid;
    oid.fromString( soid );
    std::cout <<" oid="<< oid.toString()<<std::endl;
    boost::shared_ptr<SG> obj = db.fetch<SG>( oid );
    unsigned int seed = obj->m_intData;
    SG r(seed);
    r.m_ref = new D1(seed);
    r.m_ref2 = new D2(seed);
    if( r != *obj )
    {
      std::cout <<"** test ERROR: data for class SG different from expected for seed = "<<seed<<std::endl;
    } else{
      std::cout << "** Read out data for class SG with seed="<<seed<<" is ok."<<std::endl;
    }      
  }
  
  db.transaction().commit();
  db.disconnect();
  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  }
}

