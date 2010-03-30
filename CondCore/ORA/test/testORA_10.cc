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
    std::string connStr( "sqlite_file:test.db" );
  //std::string connStr( "myalias" );
  //std::string connStr( "oracle://devdb10/giacomo" );
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
  db.createContainer<SE>("Cont0");
  std::vector<boost::shared_ptr<SE> > buff;
  std::vector<ora::OId> oids;
  for( unsigned int i = 0; i<10; i++){
    boost::shared_ptr<SE> data( new SE( i ) );
    oids.push_back( db.insert( "Cont0", *data ) );
    buff.push_back( data );
  }
  db.flush();
  buff.clear();
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
  ora::Container cont0 = db.containerHandle( "Cont0" );
  std::cout << "############# ContID="<<cont0.id()<<std::endl;
  std::cout << "** R start objs in container="<<cont0.size()<<std::endl;
  ora::ContainerIterator iter = cont0.iterator();
  while( iter.next() ){
    boost::shared_ptr<SE> obj = iter.get<SE>();    
    unsigned int seed = obj->m_intData;
    
    SE r(seed);

    if( r != *obj ){
      std::cout <<"** test ERROR: data for class SD different from expected for seed = "<<seed<<std::endl;
    } else{
      std::cout << "** Read out data for class SD with seed="<<seed<<" is ok."<<std::endl;
    }
    if( !obj->m_vec.empty() )
      std::cout << "** last vec value="<<obj->m_vec.back().m_id<<std::endl;
  }
  db.transaction().commit();
  db.disconnect();
  db.connect( connStr );
  db.transaction().start( false );
  buff.clear();
  for( std::vector<ora::OId>::const_iterator iOid=oids.begin();
       iOid != oids.end(); ++iOid ){
    boost::shared_ptr<SE> data = db.fetch<SE>(*iOid);
    db.update( *iOid, *data );
    buff.push_back( data );
    data->m_vec.push_back( SM(99) );
    data->m_vec.push_back( SM(100) );
  }
  db.flush();
  db.transaction().commit();
  db.disconnect();
  db.connect( connStr );  
  db.transaction().start( true );
  cont0 = db.containerHandle( "Cont0" );
  iter = cont0.iterator();
  while( iter.next() ){
    boost::shared_ptr<SE> obj = iter.get<SE>();    
    unsigned int seed = obj->m_intData;
    
    SE r(seed);
    r.m_vec.push_back( SM(99) );
    r.m_vec.push_back( SM(100) );

    if( r != *obj ){
      std::cout <<"** test ERROR: data for class SD different from expected for seed = "<<seed<<std::endl;
    } else{
      std::cout << "** Read out data for class SD with seed="<<seed<<" is ok."<<std::endl;
    }
    if( !obj->m_vec.empty() )
      std::cout << "** last vec value="<<obj->m_vec.back().m_id<<std::endl;
  }
  db.transaction().commit();
  db.disconnect();
  db.connect( connStr );  
  db.transaction().start( false );
  buff.clear();
  for( std::vector<ora::OId>::const_iterator iOid=oids.begin();
       iOid != oids.end(); ++iOid ){
    boost::shared_ptr<SE> data = db.fetch<SE>(*iOid);
    db.update( *iOid, *data );
    buff.push_back( data );
    if(!data->m_vec.empty()) data->m_vec.pop_back();
    if(!data->m_vec.empty()) data->m_vec.pop_back();
  }
  db.flush();
  db.transaction().commit();
  db.disconnect();
  db.connect( connStr );  
  db.transaction().start( true );
  cont0 = db.containerHandle( "Cont0" );
  iter = cont0.iterator();
  while( iter.next() ){
    boost::shared_ptr<SE> obj = iter.get<SE>();    
    unsigned int seed = obj->m_intData;
    
    SE r(seed);

    if( r != *obj ){
      std::cout <<"** test ERROR: data for class SD different from expected for seed = "<<seed<<std::endl;
    } else{
      std::cout << "** Read out data for class SD with seed="<<seed<<" is ok."<<std::endl;
    }
    if( !obj->m_vec.empty() )
      std::cout << "** last vec value="<<obj->m_vec.back().m_id<<std::endl;
  }
  db.transaction().commit();
  db.disconnect();

  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  }
}

