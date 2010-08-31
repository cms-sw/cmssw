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
    db.createContainer<SF>("Cont0");
    std::vector<boost::shared_ptr<SF> > buff;
    std::vector<ora::OId> oids;
    for( unsigned int i = 0; i<10; i++){
      boost::shared_ptr<SF> data( new SF( i ) );
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
    trans.start( true );
    ora::Container cont0 = db.containerHandle( "Cont0" );
    ora::ContainerIterator iter = cont0.iterator();
    while( iter.next() ){
      boost::shared_ptr<SF> obj = iter.get<SF>();
      unsigned int seed = obj->m_intData;
      SF r(seed);
      if( r != *obj ){
        std::stringstream mess;
        mess << "Data for class SF different from expected for seed = "<<seed;
        ora::throwException( mess.str(),"testORA_11");
      } else{
        std::cout << "** Read out data for class SF with seed="<<seed<<" is ok."<<std::endl;
      }
    }
    trans.commit();
    db.disconnect();
    std::cout << "#### UPDATING .."<<std::endl;
    db.connect( connStr );
    trans.start( false );
    for( std::vector<ora::OId>::const_iterator iOid=oids.begin();
         iOid != oids.end(); ++iOid ){
      boost::shared_ptr<SF> data = db.fetch<SF>(*iOid);
      db.update( *iOid, *data );
      buff.push_back( data );
      data->m_vec.push_back( SM(99) );
      data->m_vec.push_back( SM(100) );
    }
    db.flush();
    buff.clear();
    trans.commit();
    db.disconnect();
    std::cout << "#### READING AFTER UPDATE... .."<<std::endl;
    db.connect( connStr );
    trans.start( true );
    cont0 = db.containerHandle( "Cont0" );
    iter = cont0.iterator();
    while( iter.next() ){
      boost::shared_ptr<SF> obj = iter.get<SF>();
      unsigned int seed = obj->m_intData;
      SF r(seed);
      r.m_vec.push_back( SM(99) );
      r.m_vec.push_back( SM(100) );
      if( r != *obj ){
        std::stringstream mess;
        mess << "Data for class SF after update (1) different from expected for seed = "<<seed;
        ora::throwException( mess.str(),"testORA_11");
      } else{
        std::cout << "** Read out data for class SF after update (1) with seed="<<seed<<" is ok."<<std::endl;
      }
    }
    trans.commit();
    db.disconnect();
    db.connect( connStr );
    trans.start( false );
    for( std::vector<ora::OId>::const_iterator iOid=oids.begin();
         iOid != oids.end(); ++iOid ){
      boost::shared_ptr<SF> data = db.fetch<SF>(*iOid);
      db.update( *iOid, *data );
      buff.push_back( data );
      if(!data->m_vec.empty()) data->m_vec.pop_back();
      if(!data->m_vec.empty()) data->m_vec.pop_back();
    }
    db.flush();
    buff.clear();
    trans.commit();
    db.disconnect();
    db.connect( connStr );
    db.transaction().start( true );
    cont0 = db.containerHandle( "Cont0" );
    iter = cont0.iterator();
    std::vector<ora::QueryableVector<SM> > qvecs;
    while( iter.next() ){
      boost::shared_ptr<SF> obj = iter.get<SF>();
      unsigned int seed = obj->m_intData;
      qvecs.push_back( obj->m_vec );
      SF r(seed);
      if( r != *obj ){
        std::stringstream mess;
        mess << "Data for class SF after update (2) different from expected for seed = "<<seed;
        ora::throwException( mess.str(),"testORA_11");
      } else{
        std::cout << "** Read out data for class SF after update (2) with seed="<<seed<<" is ok."<<std::endl;
      }
    }
    std::cout << "********** Queries.------"<<std::endl;
    for( std::vector<ora::QueryableVector<SM> >::iterator iV = qvecs.begin();
         iV != qvecs.end(); iV++ ){
      std::cout << "*** Query cycle...------"<<std::endl;
      ora::Query<SM> q1 = iV->query();
      q1.addSelection("m_id",ora::GT,2);
      ora::Range<SM> rg1 = q1.execute();
      std::cout << "**range size="<<rg1.size()<<std::endl;
      ora::Query<SM> q2 = iV->query();
      q2.addSelection("m_id",ora::GT,2);
      std::cout << "**count size="<<q2.count()<<std::endl;
      ora::Range<SM> sel0 = iV->select(3,5);
      std::cout << "**range sel0 size="<<sel0.size()<<std::endl;
      ora::Range<SM> sel1 = iV->select(4);
      std::cout << "**range sel1 size="<<sel1.size()<<std::endl;
      ora::Range<SM> sel2 = iV->select(4,-1);
      std::cout << "**range sel2 size="<<sel2.size()<<std::endl;
      ora::Range<SM> sel3 = iV->select(2,2);
      std::cout << "**range sel3 size="<<sel3.size()<<std::endl;
      ora::Range<SM> sel4 = iV->select(0);
      std::cout << "**range sel4 size="<<sel4.size()<<std::endl;
      ora::Query<SM> q3 = iV->query();
      q3.addSelection(ora::Selection::indexVariable(),ora::GE,3);
      q3.addSelection(ora::Selection::indexVariable(),ora::LE,5);
      ora::Range<SM> rg2 = q3.execute();
      std::cout << "**range rg2 size="<<rg2.size()<<std::endl;
      ora::Selection sel;
      sel.addDataItem<int>("m_id",ora::GT,2);
      sel.addDataItem<int>("m_id",ora::LT,8);
      sel.addDataItem<int>("m_id",ora::NE,4);
      sel.addIndexItem(3);
      ora::Range<SM> rg3 = iV->select(sel);
      std::cout << "**range rg3 size="<<rg3.size()<<std::endl;
      unsigned int i=0;
      for(ora::Range<SM>::const_iterator iR=rg3.begin();
          iR!=rg3.end();++iR){
        std::cout << "##Rg3 ("<<i<<") SM["<<iR.index()<<"]="<<iR->m_id<<std::endl;
        i++;
      }
    }
    trans.commit();
    db.transaction().start( false );
    db.drop();
    db.transaction().commit();
    db.disconnect();

    const ora::QueryableVector<std::string> pv(1);
    pv.size();
    const ora::QueryableVector<std::string> pv0(1,std::string(""));
    ora::QueryableVector<std::string> pv1(pv0);
    ora::QueryableVector<std::string> pv2;
    for(ora::QueryableVector<std::string>::iterator i=pv2.begin();
        i!=pv2.end();i++){}
    for(ora::QueryableVector<std::string>::const_iterator i=pv0.begin();
        i!=pv0.end();i++){}
    for(ora::QueryableVector<std::string>::reverse_iterator i=pv2.rbegin();
        i!=pv2.rend();i++){}
    for(ora::QueryableVector<std::string>::const_reverse_iterator i=pv0.rbegin();
        i!=pv0.rend();i++){}
    //
    pv0.max_size();
    pv2.resize(2);
    pv2.resize(3,std::string(""));
    pv0.capacity();
    pv0.empty();
    pv2.reserve(5);
    std::string s0 = pv2[0];
    std::string s1 = pv2.at(1);
    std::string s2 = pv2.front ( );
    std::string s3 = pv2.back ( );
    s0.append(s1).append(s2).append(s3);
    std::string s4 = pv0[0];
    std::string s5 = pv0.at(0);
    std::string s6 = pv0.front ( );
    std::string s7 = pv0.back ( );
    s4.append(s5).append(s6).append(s7);
    pv2.assign (3, std::string("1") );
    pv2.push_back (std::string("2") );
    pv2.pop_back ();
    pv2.clear ( );
    if(pv0==pv1) {  }
    if(pv0!=pv1) { }

  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  }
}

