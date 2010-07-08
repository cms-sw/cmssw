#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/ScopedTransaction.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/IReferenceHandler.h"
#include <cstdlib>
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
      return true;
    }
  private:
    ora::Database& m_db;
};

int main(){
  try {

    // writing...  
    std::string authpath("/afs/cern.ch/cms/DB/conddb");
    std::string pathenv(std::string("CORAL_AUTH_PATH=")+authpath);
    ::putenv(const_cast<char*>(pathenv.c_str()));
    ora::Database db;
    ReferenceHandler* refHandler = new ReferenceHandler( db );
    db.configuration().setReferenceHandler( refHandler );
    db.configuration().setMessageVerbosity( coral::Debug );
    std::string connStr( "oracle://cms_orcoff_prep/CMS_COND_WEB" );
    db.connect( connStr );
    ora::ScopedTransaction trans( db.transaction() );
    trans.start( false );
    if(!db.exists()){
      db.create();
    }
    std::set< std::string > conts = db.containers();
    if( conts.find( "Cont0" )!= conts.end() ) db.dropContainer( "Cont0" );
    if( conts.find( "Cont1" )!= conts.end() ) db.dropContainer( "Cont1" );
    //
    db.createContainer<SC>("Cont0");
    db.createContainer<SimpleClass>("Cont1");
    std::vector<boost::shared_ptr<SimpleClass> > buff0;
    std::vector<boost::shared_ptr<SC> > buff1;
    std::vector<ora::OId> oids;
    std::vector<Ref<SimpleClass> > refs;
    for( unsigned int i = 0; i<10; i++){
      boost::shared_ptr<SimpleClass> data0( new SimpleClass(i) );
      buff0.push_back( data0 );
      oids.push_back( db.insert("Cont1", *data0 ) );
      Ref<SimpleClass> r;
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
    trans.commit();
    db.disconnect();
    ::sleep(1);
    // reading back...
    db.connect( connStr );
    trans.start( true );
    ora::Container cont0 = db.containerHandle( "Cont0" );
    ora::ContainerIterator iter = cont0.iterator();
    while( iter.next() ){
      boost::shared_ptr<SC> obj = iter.get<SC>();
      obj->m_ref.load();
      for( size_t i=0;i<obj->m_refVec.size();i++){
        obj->m_refVec[i].load();
      }

      unsigned int seed = obj->m_intData;
      SC r(seed);
      boost::shared_ptr<SimpleClass> sc( new SimpleClass(seed) );
      r.m_ref.m_data = sc;
      std::vector<Ref<SimpleClass> > refs;
      for( size_t i=0;i<10; i++ ){
        boost::shared_ptr<SimpleClass> sci( new SimpleClass(i) );
        refs.push_back( Ref<SimpleClass>() );
        refs.back().m_data = sci;
      }
      r.m_refVec = refs;

      if( r != *obj ){
        std::stringstream mess;
        mess << "Data for class SC different from expected for seed = "<<seed;
        ora::throwException( mess.str(),"testORA_8");
      } else{
        std::cout << "** Read out data for class SC with seed="<<seed<<" is ok."<<std::endl;
      }
    }
    trans.commit();
    trans.commit();
    trans.start( false );
    db.drop();
    trans.commit();
    db.disconnect();
    db.disconnect();
  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  }
}

