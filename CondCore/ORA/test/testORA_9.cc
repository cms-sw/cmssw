#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/ScopedTransaction.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/IReferenceHandler.h"
#include "CondCore/ORA/test/Serializer.h"
#include <cstdlib>
#include <iostream>

#include "classes.h"

int main(){
  using namespace testORA;
  try {

    // writing...  
    std::string authpath("/afs/cern.ch/cms/DB/conddb");
    std::string pathenv(std::string("CORAL_AUTH_PATH=")+authpath);
    ::putenv(const_cast<char*>(pathenv.c_str()));
    ora::Database db;
    db.configuration().setMessageVerbosity( coral::Debug );
    std::string connStr( "oracle://cms_orcoff_prep/CMS_COND_UNIT_TESTS" );
    ora::Serializer serializer( "ORA_TEST" );
    serializer.lock( connStr );
    db.connect( connStr );
    ora::ScopedTransaction trans( db.transaction() );
    trans.start( false );
    if(!db.exists()){
      db.create();
    }
    std::set< std::string > conts = db.containers();
    if( conts.find( "Cont0" )!= conts.end() ) db.dropContainer( "Cont0" );
    //
    db.createContainer<SD>("Cont0");
    std::vector<boost::shared_ptr<SD> > buff;
    for( unsigned int i = 0; i<10; i++){
      boost::shared_ptr<SD> data( new SD( i ) );
      db.insert( "Cont0", *data );
      buff.push_back( data );
      data->m_ptr = new SimpleClass(i);
      for( unsigned int j=0;j<i;j++ ){
        data->m_ptrVec.push_back( ora::Ptr<SimpleMember>( new SimpleMember(j) ));
      }
    }
    db.flush();
    buff.clear();
    trans.commit();
    db.disconnect();
    ::sleep(1);
    // reading back...
    db.connect( connStr );
    trans.start( true );
    ora::Container cont0 = db.containerHandle( "Cont0" );
    ora::ContainerIterator iter = cont0.iterator();
    while( iter.next() ){
      boost::shared_ptr<SD> obj = iter.get<SD>();
      unsigned int seed = obj->m_intData;
      SD r(seed);
      r.m_ptr = new SimpleClass(seed);
      for( unsigned int j=0;j<seed;j++ ){
        r.m_ptrVec.push_back( ora::Ptr<SimpleMember>( new SimpleMember(j) ));
      }
      if( r != *obj ){
        std::stringstream mess;
        mess << "(1) Data for class SD different from expected for seed = "<<seed;
        ora::throwException( mess.str(),"testORA_9");
      } else{
        std::cout << "** (1) Read out data for class SD with seed="<<seed<<" is ok."<<std::endl;
      }
    }
    iter.reset();
    while( iter.next() ){
      boost::shared_ptr<SD> obj = iter.get<SD>();
      unsigned int seed = obj->m_intData;
      SD r(seed);
      r.m_ptr = new SimpleClass(seed);
      for( unsigned int j=0;j<seed;j++ ){
        r.m_ptrVec.push_back( ora::Ptr<SimpleMember>( new SimpleMember(j) ));
      }
      if( r != *obj ){
        std::stringstream mess;
        mess << "(2) Data for class SD different from expected for seed = "<<seed;
        ora::throwException( mess.str(),"testORA_9");
      } else{
        std::cout << "** (2) Read out data for class SD with seed="<<seed<<" is ok."<<std::endl;
      }
    }
    trans.commit();
    trans.start( false );
    db.drop();
    trans.commit();
    db.disconnect();
  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  }
}

