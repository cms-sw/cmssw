#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include <iostream>
//#include <typeinfo>

int main(){
    // writing...  
  ora::Database db;
  db.configuration().setMessageVerbosity( coral::Debug );
  try {

    //std::string connStr( "oracle://devdb10/giacomo" );
    std::string connStr( "sqlite_file:test.db" );
  db.connect( connStr );
  db.transaction().start( false );
  bool exists = db.exists();
  if(exists){
    db.dropContainer( "Cont0" );
    db.dropContainer( "std::string" );
    std::cout << "############# ORA database does exist in "<< connStr<<"."<<std::endl;
  } else {
    std::cout << "############# ORA database does not exist in "<< connStr<<", creating it..."<<std::endl;
    db.create();
  }
  std::cout <<" ############# opening db...."<<std::endl;
  std::set< std::string > conts = db.containers();
  std::cout << conts.size() <<" ############# container(s) found."<<std::endl;
  for(std::set<std::string>::const_iterator iC = conts.begin();
      iC != conts.end(); iC++ ){
    std::cout << "############# CONT=\""<<*iC<<"\""<<std::endl;
  }
  db.createContainer<int>("Cont0");
  ora::Container contH0 = db.containerHandle( "Cont0" );
  int myData0(999);
  int oid0 = contH0.insert( myData0 );
  int myData01(1234567890);
  int oid01 = contH0.insert( myData01 );
  contH0.flush();
  int contId = db.createContainer<std::string>().id();
  ora::Container contH1 = db.containerHandle( contId );
  std::string myData1("ABCDEFGHILMNOPQRSTUVZ1234567890");
  int oid1 = contH1.insert( myData1 );
  std::string myData11("GiacomoGovi");
  int oid11 = contH1.insert( myData11 );
  contH1.flush();
  db.transaction().commit();
  db.disconnect();
  ::sleep(1);
  db.connect( connStr );  
  db.transaction().start( true );
  bool exists2 = db.exists();
  if(exists2){
    std::cout << "############# ORA database does exist in "<< connStr<<"."<<std::endl;
  } else {
    std::cout << "############# ERROR!! ORA database does not exist in "<< connStr<<std::endl;
  }
  std::set<std::string> conts2 = db.containers();
  std::cout << conts2.size() <<" ############# container(s) found."<<std::endl;
  for(std::set<std::string>::const_iterator iC = conts2.begin();
      iC != conts2.end(); iC++ ){
    std::cout << "############# CONT=\""<<*iC<<"\""<<std::endl;
  }
  ora::Container contHR0 = db.containerHandle( "Cont0" );
  std::cout << "############# ContID="<<contHR0.id()<<std::endl;
  boost::shared_ptr<int> readData0 = contHR0.fetch<int>( oid0);
  std::cout << "############# Read out data="<<*readData0<<std::endl;
  readData0 = contHR0.fetch<int>( oid01);
  std::cout << "############# Read out data="<<*readData0<<std::endl;
  ora::Container contHR1 = db.containerHandle( contId );
  std::cout << "############# ContID="<<contHR1.id()<<std::endl;
  boost::shared_ptr<std::string> readData1 = contHR1.fetch<std::string>( oid1 );
  std::cout << "############# Read out data="<<*readData1<<std::endl;
  readData1 = contHR1.fetch<std::string>( oid11 );
  std::cout << "############# Read out data="<<*readData1<<std::endl;
  db.transaction().commit();
  db.disconnect();
  db.connect( connStr );  
  db.transaction().start( false );
  ora::Container contHU0 = db.containerHandle( "Cont0" );
  int new0(888);
  contHU0.update( oid0, new0 );
  int new1(987654321);
  contHU0.update( oid01, new1 );
  contHU0.flush();
  boost::shared_ptr<int> rd0 = contHU0.fetch<int>( oid0);
  std::cout << "############# Read data before commit="<<*rd0<<std::endl;
  rd0 = contHU0.fetch<int>( oid01);
  std::cout << "############# Read data before commit="<<*rd0<<std::endl;
  ora::Container contHU1 = db.containerHandle( contId );
  std::string ns0("0987654321abcdefghilmnopqrstuvz");
  contHU1.update( oid1, ns0 );
  std::string ns1("PincoPallino");
  contHU1.update( oid11, ns1 );
  contHU1.flush();
  boost::shared_ptr<std::string> rd1 = contHU1.fetch<std::string>( oid1 );
  std::cout << "############# Read data before commit="<<*rd1<<std::endl;
  rd1 = contHU1.fetch<std::string>( oid11 );
  std::cout << "############# Read data before commit="<<*rd1<<std::endl;
  std::cout << "############# COMMIT 1 "<<std::endl;
  db.transaction().commit();
  std::cout << "############# DISCO 1 "<<std::endl;
  db.disconnect();
  db.connect( connStr );  
  db.transaction().start( true );
  std::cout << "############# opening 2."<<std::endl;
  ora::Container contHRR0 = db.containerHandle( "Cont0" );
  boost::shared_ptr<int> rd00 = contHRR0.fetch<int>( oid0);
  std::cout << "############# Read out data="<<*rd00<<std::endl;
  rd00 = contHRR0.fetch<int>( oid01);
  std::cout << "############# Read out data="<<*rd00<<std::endl;
  ora::Container contHRR1 = db.containerHandle( contId );
  boost::shared_ptr<std::string> rd11 = contHRR1.fetch<std::string>( oid1 );
  std::cout << "############# Read out data="<<*rd11<<std::endl;
  rd11 = contHRR1.fetch<std::string>( oid11 );
  std::cout << "############# Read out data="<<*rd11<<std::endl;
  ora::ContainerIterator iter0 = contHRR0.iterator();
  while( iter0.next() ){
    boost::shared_ptr<int> o = iter0.get<int>();
    std::cout << " **** Cont="<<contHRR0.name()<<" val="<<*o<<std::endl;
  }
  ora::ContainerIterator iter1 = contHRR1.iterator();
  while( iter1.next() ){
    boost::shared_ptr<std::string> s = iter1.get<std::string>();
    std::cout << " **** Cont="<<contHRR1.name()<<" val="<<*s<<std::endl;
  }
  db.transaction().commit();
  try {
    rd11 = contHRR1.fetch<std::string>( oid11 );
  } catch (ora::Exception& e){
    std::cout << "*** expected exception:"<<e.what()<<std::endl;
  }
  db.disconnect();
  try {
    boost::shared_ptr<std::string> null = iter1.get<std::string>();
  } catch (ora::Exception& e){
    std::cout << "*** expected exception:"<<e.what()<<std::endl;
  }
  std::cout << "#### deleting..."<<std::endl;
  db.connect( connStr );  
  db.transaction().start( false );
  ora::OId foid0( db.containerHandle( "Cont0" ).id(), oid0);
  db.erase( foid0 );
  db.containerHandle( contId ).erase( oid11 );
  db.flush();
  db.transaction().commit();
  db.disconnect();
  std::cout << "#### reading after delete..."<<std::endl;
  db.connect( connStr );  
  db.transaction().start( true );
  std::cout << "############# opening 3."<<std::endl;
  contHRR0 = db.containerHandle( "Cont0" );
  rd00 = contHRR0.fetch<int>( oid0);
  if( rd00 ){
    std::cout << "## ERROR: Read out data="<<*rd00<<std::endl;
  } else {
    std::cout << "## No data for oid="<<oid0<<std::endl;    
  }
  rd00 = contHRR0.fetch<int>( oid01);
  std::cout << "############# Read out data="<<*rd00<<std::endl;
  contHRR1 = db.containerHandle( contId );
  rd11 = contHRR1.fetch<std::string>( oid11 );
  if( rd11 ){
    std::cout << "## ERROR: Read out data="<<*rd11<<std::endl;
  } else {
    std::cout << "## No data for oid="<<oid11<<std::endl;    
  }
  rd11 = contHRR1.fetch<std::string>( oid1 );
  std::cout << "############# Read out data="<<*rd11<<std::endl;
  iter0 = contHRR0.iterator();
  while( iter0.next() ){
    boost::shared_ptr<int> o = iter0.get<int>();
    std::cout << " **** Cont="<<contHRR0.name()<<" val="<<*o<<std::endl;
  }
  iter1 = contHRR1.iterator();
  while( iter1.next() ){
    boost::shared_ptr<std::string> s = iter1.get<std::string>();
    std::cout << " **** Cont="<<contHRR1.name()<<" val="<<*s<<std::endl;
    }
  db.transaction().commit();
  db.disconnect();  
  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  }
}

