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
    //std::string connStr( "myalias" );
    std::string connStr( "sqlite_file:test.db" );
  db.connect( connStr );
  db.transaction().start( false );
  bool exists = db.exists();
  if(exists){
    std::cout << "############# ORA database does exist in "<< connStr<<"."<<std::endl;
    db.dropContainer( "Cont0" );
    db.dropContainer( "Cont1" );
    db.dropContainer( "std::vector<std::vector<int> >" );
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
  ora::Container contH0 = db.createContainer<std::vector<int> >("Cont0");
  ora::Container contH1 = db.createContainer<std::vector<SimpleMember> >("Cont1");
  int contId = db.createContainer<std::vector<std::vector<int> > >().id();
  std::vector<int> v0;
  int oid0 = contH0.insert( v0 );
  std::vector<int> v1;
  for(int i=0;i<2;i++) v1.push_back( i );
  int oid1 = contH0.insert( v1 );
  std::vector<int> v2;
  for(int i=0;i<10;i++) v2.push_back( i );
  int oid2 = contH0.insert( v2 );
  contH0.flush();
  std::vector<SimpleMember> vs;
  for(long i=0;i<5;i++) vs.push_back( SimpleMember( i ));
  int oids = contH1.insert( vs );
  contH1.flush();
  ora::Container contH2 = db.containerHandle( contId );
  std::vector<std::vector<int> > v3;
  for(int i=0;i<3;i++) {
    std::vector<int> in;
    for(int j=0;j<i;j++) in.push_back(j);
    std::cout << "---> W v ["<<i<<"] size="<<in.size()<<std::endl;
    v3.push_back( in );
  }
  int oid3 = contH2.insert( v3 );
  std::vector<std::vector<int> > v4;
  for(int i=0;i<10;i++) {
    std::vector<int> in;
    for(int j=0;j<i;j++) in.push_back(j);
    std::cout << "---> W v ["<<i<<"] size="<<in.size()<<std::endl;
    v4.push_back( in );
  }
  int oid4 = contH2.insert( v4 );
  contH2.flush();
  db.transaction().commit();
  db.disconnect();
  ::sleep(1);
  // reading back...
  db.connect( connStr );  
  db.transaction().start( true );
  ora::Container contHR0 = db.containerHandle( "Cont0" );
  std::cout << "############# Cont 0 ID="<<contHR0.id()<<" oid="<<oid0<<std::endl;
  boost::shared_ptr<std::vector<int> > vr0 = contHR0.fetch<std::vector<int> >( oid0 );
  if( vr0 ){
    std::cout << "Read out vector size="<<vr0->size()<<std::endl;
    for(size_t j=0;j<vr0->size();j++) std::cout << "v["<<j<<"]="<<vr0->operator[](j)<<std::endl;
  } else {
    std::cout << "############# No data for oid="<<oid0<<std::endl;
  }
  std::cout << "############# Cont 0 ID="<<contHR0.id()<<" oid="<<oid1<<std::endl;
  boost::shared_ptr<std::vector<int> > vr1 = contHR0.fetch<std::vector<int> >( oid1 );
  if( vr1 ){
    std::cout << "Read out vector size="<<vr1->size()<<std::endl;
    for(size_t j=0;j<vr1->size();j++) std::cout << "v["<<j<<"]="<<vr1->operator[](j)<<std::endl;
  } else {
    std::cout << "############# No data for oid="<<oid1<<std::endl;
  }
  std::cout << "############# Cont 0 ID="<<contHR0.id()<<" oid="<<oid2<<std::endl;
  boost::shared_ptr<std::vector<int> > vr2 = contHR0.fetch<std::vector<int> >( oid2 );
  if( vr2 ){
    std::cout << "Read out vector size="<<vr2->size()<<std::endl;
    for(size_t j=0;j<vr2->size();j++) std::cout << "v["<<j<<"]="<<vr2->operator[](j)<<std::endl;
  } else {
    std::cout << "############# No data for oid="<<oid2<<std::endl;
  }
  ora::Container contHR1 = db.containerHandle( "Cont1" );
  std::cout << "############# Cont 1 ID="<<contHR1.id()<<" oid="<<oids<<std::endl;
  boost::shared_ptr<std::vector<SimpleMember> > vrs = contHR1.fetch<std::vector<SimpleMember> >( oids );
  if( vrs ){
    std::cout << "Read out vector size="<<vrs->size()<<std::endl;
    for(size_t j=0;j<vrs->size();j++) {
      std::string flag("FALSE");
      if( vrs->operator[](j).m_flag) flag="TRUE";
      std::cout << "v["<<j<<"].flag="<<flag<<std::endl;
      std::cout << "v["<<j<<"].id="<<vrs->operator[](j).m_id<<std::endl;
    }
  } else {
     std::cout << "############# No data for oid="<<oids<<std::endl;
  }

  ora::Container contHR2 = db.containerHandle( contId );
  std::cout << "############# Cont 2 ID="<<contHR2.id()<<" size="<<contHR2.size()<<" oid3="<<oid3<<std::endl;
  boost::shared_ptr<std::vector<std::vector<int> > > vr3 = contHR2.fetch<std::vector<std::vector<int> > >( oid3 );
  if( vr3 ){
    std::cout << "Read out vector size="<<vr3->size()<<std::endl;
    for(size_t j=0;j<vr3->size();j++) {
      std::vector<int>& el = vr3->operator[](j);
      std::cout << "---> R v ["<<j<<"] size="<<el.size()<<std::endl;
      for( size_t i=0;i<el.size();i++) {
        std::cout << "R v3["<<j<<"]["<<i<<"]="<<el[i]<<std::endl;
      }
    }
    } else {
    std::cout << "############# No data for oid="<<oid3<<std::endl;
  }
  std::cout << "############# Cont 2 ID="<<contHR2.id()<<" size="<<contHR2.size()<<" oid3="<<oid4<<std::endl;
  boost::shared_ptr<std::vector<std::vector<int> > > vr4 = contHR2.fetch<std::vector<std::vector<int> > >( oid4 );
  if( vr4 ){
    std::cout << "Read out vector size="<<vr4->size()<<std::endl;
    for(size_t j=0;j<vr4->size();j++) {
      std::vector<int>& el = vr4->operator[](j);
      std::cout << "---> R v ["<<j<<"] size="<<el.size()<<std::endl;
      for( size_t i=0;i<el.size();i++) {
        std::cout << "R v4["<<j<<"]["<<i<<"]="<<el[i]<<std::endl;
      }
    }
    } else {
    std::cout << "############# No data for oid="<<oid4<<std::endl;
  } 
  db.transaction().commit();
  db.disconnect();
  // update
  db.connect( connStr );
  db.transaction().start( false );
  ora::Container contHU0 = db.containerHandle( "Cont0" );
  std::vector<int> vn;
  for(int i=10;i>0;i--) vn.push_back( i );
  contHU0.update( oid1, vn );
  contHU0.flush();
  ora::Container contHU2 = db.containerHandle( contId );
  std::vector<std::vector<int> > v3n;
  for(int i=0;i<10;i++) {
    std::vector<int> in;
    for(int j=i;j>0;j--) in.push_back(j);
    v3n.push_back( in );
  }
  contHU2.update( oid3 , v3n );
  contHU2.flush();

  db.transaction().commit();
  db.disconnect();
    // reading back...
  db.connect( connStr );  
  db.transaction().start( true );
  contHR0 = db.containerHandle( "Cont0" );
  std::cout << "############# Cont 0 ID="<<contHR0.id()<<" oid="<<oid1<<std::endl;
  vr1 = contHR0.fetch<std::vector<int> >( oid1 );
  if( vr1 ){
    std::cout << "Read out after update vector size="<<vr1->size()<<std::endl;
    for(size_t j=0;j<vr1->size();j++) std::cout << "v["<<j<<"]="<<vr1->operator[](j)<<std::endl;
  } else {
    std::cout << "############# No data after update for oid="<<oid1<<std::endl;
  }
  contHR2 = db.containerHandle( contId );
  std::cout << "############# Cont 2 ID="<<contHR2.id()<<" size="<<contHR2.size()<<" oid3="<<oid3<<std::endl;
  vr3 = contHR2.fetch<std::vector<std::vector<int> > >( oid3 );
  if( vr4 ){
    std::cout << "Read out after update vector size="<<vr3->size()<<std::endl;
    for(size_t j=0;j<vr3->size();j++) {
      std::vector<int>& el = vr3->operator[](j);
      std::cout << "---> R v ["<<j<<"] size="<<el.size()<<std::endl;
      for( size_t i=0;i<el.size();i++) {
        std::cout << "R v3["<<j<<"]["<<i<<"]="<<el[i]<<std::endl;
      }
    }
    } else {
    std::cout << "############# No data after update for oid="<<oid3<<std::endl;
  } 
  db.transaction().commit();
  db.disconnect();
  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  }
}

