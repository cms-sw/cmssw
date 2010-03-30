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
  db.createContainer<SA>("Cont0");
  ora::Container contH0 = db.containerHandle( "Cont0" );
  std::cout << "** W start objs in container="<<contH0.size()<<std::endl;
  std::vector<boost::shared_ptr<SA> > buff;
  for( unsigned int i=0;i<10;i++){
    boost::shared_ptr<SA> obj( new SA(i) );
    contH0.insert( *obj );
    buff.push_back( obj );
  }
  contH0.flush();
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
  contH0 = db.containerHandle( "Cont0" );
  std::cout << "############# ContID="<<contH0.id()<<std::endl;
  std::cout << "** R start objs in container="<<contH0.size()<<std::endl;
  ora::ContainerIterator iter = contH0.iterator();
  while( iter.next() ){
    boost::shared_ptr<SA> obj = iter.get<SA>();
    int seed = obj->m_intData;
    SA r(seed);
    if( r != *obj ){
      std::cout <<"** test ERROR: data different from expected for seed = "<<seed<<std::endl;
    } else{
      std::cout << "** Read out data with seed="<<seed<<" is ok."<<std::endl;
    }
  }
  db.transaction().commit();
  db.disconnect();
  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  }
}

