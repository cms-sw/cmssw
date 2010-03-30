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
  db.createContainer<SD>("Cont0");
  std::vector<boost::shared_ptr<SD> > buff;
  for( unsigned int i = 0; i<10; i++){
    boost::shared_ptr<SD> data( new SD( i ) );
    db.insert( "Cont0", *data );
    buff.push_back( data );
    data->m_ptr = new SimpleCl(i);
    for( unsigned int j=0;j<i;j++ ){
      data->m_ptrVec.push_back( ora::Ptr<SimpleMember>( new SimpleMember(j) ));
    }
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
    boost::shared_ptr<SD> obj = iter.get<SD>();    
    unsigned int seed = obj->m_intData;
    
    SD r(seed);
    r.m_ptr = new SimpleCl(seed);
    for( unsigned int j=0;j<seed;j++ ){
      r.m_ptrVec.push_back( ora::Ptr<SimpleMember>( new SimpleMember(j) ));
    }

    if( r != *obj ){
      std::cout <<"** test ERROR: data for class SD different from expected for seed = "<<seed<<std::endl;
    } else{
      std::cout << "** Read out data for class SD with seed="<<seed<<" is ok."<<std::endl;
    }
  }
  /**
  iter.reset();
  while( iter.next() ){
    boost::shared_ptr<SD> obj = iter.get<SD>();    
    obj->m_intData;
    std::cout <<"** main ind="<<obj->m_intData<<std::endl;
  }
  **/
  db.transaction().commit();
  db.disconnect();
  db.disconnect();
  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  }
}

