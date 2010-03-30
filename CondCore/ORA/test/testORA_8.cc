#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/IReferenceHandler.h"
#include <iostream>

#include "classes.h"

class ReferenceHandler : public ora::IReferenceHandler {
  public:
    explicit ReferenceHandler( ora::Database& db ):
      m_db( db ){
    }
    
    /// destructor
    virtual ~ReferenceHandler() {}

    bool onSave( ora::Reference& ref ){
      return true;
    }
    

    bool onLoad( ora::Reference& r ){
      RefBase& ref = dynamic_cast<RefBase&>( r );
      ref.setDb( m_db );
      std::cout << "## on load called oid0="<<r.oid().containerId()<<std::endl;
      return true;
    }
  private:
    ora::Database& m_db;
};

int main(){
  try {

    //const boost::filesystem::path dict_path("testORADict");
    //edmplugin::SharedLibrary shared( dict_path );
    // writing...  
    ora::Database db;
    ReferenceHandler* refHandler = new ReferenceHandler( db );
    db.configuration().setReferenceHandler( refHandler );
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
    db.dropContainer( "Cont1" );
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
  db.createContainer<SC>("Cont0");
  db.createContainer<SimpleCl>("Cont1");
  std::vector<boost::shared_ptr<SimpleCl> > buff0;
  std::vector<boost::shared_ptr<SC> > buff1;
  std::vector<ora::OId> oids;
  std::vector<Ref<SimpleCl> > refs;
  for( unsigned int i = 0; i<10; i++){
    boost::shared_ptr<SimpleCl> data0( new SimpleCl(i) );
    buff0.push_back( data0 );
    oids.push_back( db.insert("Cont1", *data0 ) );
    Ref<SimpleCl> r;
    r.set( oids.back() );
    refs.push_back( r );
  }
  for( unsigned int i = 0; i<10; i++){
    boost::shared_ptr<SC> data1( new SC( i ) );
    db.insert( "Cont0", *data1 );
    buff1.push_back( data1 );
    data1->m_ref.set( oids[i] );
    data1->m_refVec = refs;
  }
  db.flush();
  buff1.clear();
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
  ora::Container cont0 = db.containerHandle( "Cont0" );
  std::cout << "############# ContID="<<cont0.id()<<std::endl;
  std::cout << "** R start objs in container="<<cont0.size()<<std::endl;
  ora::ContainerIterator iter = cont0.iterator();
  while( iter.next() ){
    boost::shared_ptr<SC> obj = iter.get<SC>();
    obj->m_ref.load();
    for( size_t i=0;i<obj->m_refVec.size();i++){
      std::cout << "#### loading refVec i="<<i<<std::endl;
      obj->m_refVec[i].load();
    }
    
    unsigned int seed = obj->m_intData;
    SC r(seed);
    boost::shared_ptr<SimpleCl> sc( new SimpleCl(seed) );
    r.m_ref.m_data = sc;
    std::vector<Ref<SimpleCl> > refs;
    for( size_t i=0;i<10; i++ ){
      boost::shared_ptr<SimpleCl> sci( new SimpleCl(i) );
      refs.push_back( Ref<SimpleCl>() );
      refs.back().m_data = sci;
    }
    r.m_refVec = refs;

    if( r != *obj ){
      std::cout <<"** test ERROR: data for class SiStripNoises different from expected for seed = "<<seed<<std::endl;
    } else{
      std::cout << "** Read out data for class SiStripNoises with seed="<<seed<<" is ok."<<std::endl;
    }
  }
  db.transaction().commit();
  db.disconnect();
  db.disconnect();
  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  }
}

