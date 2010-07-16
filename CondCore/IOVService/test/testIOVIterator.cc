#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"

#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondCore/IOVService/interface/KeyList.h"

#include <iostream>
#include <algorithm>
#include <boost/bind.hpp>

#include "CondCore/IOVService/interface/PayloadProxy.h"


namespace {

  std::string oid(std::string token) {
    int pos = token.rfind('[');
    if (pos<0) return "[]";
    return token.substr(pos);
  }

  void print(cond::IOVElementProxy const & e) {
    std::cout<<"oid "<< oid(e.wrapperToken())
	     <<", since "<< e.since()
	     <<", till "<< e.till()
	     << std::endl;
  }
  
  void printT(cond::PayloadProxy<cond::IOVElement> & data, cond::Time_t time) {
    cond::ValidityInterval iov = data.setIntervalFor(time);
    data.make();
    std::cout << "for " << time
	     <<": since "<< iov.first
	     <<", till "<< iov.second;
    if (data.isValid()) 
      std::cout    <<". Message "<< data().wrapperToken()
		   <<", since "<< data().sinceTime();
    else 
      std::cout << ". No data";
    std::cout << std::endl;
  }

  void printN(cond::PayloadProxy<cond::IOVElement> & data, size_t n) {
    cond::ValidityInterval iov = data.loadFor(n);
    std::cout << "for " << n
	      <<": since "<< iov.first
	      <<", till "<< iov.second;
    if (data.isValid()) 
      std::cout    <<". Message "<< data().wrapperToken()
		   <<", since "<< data().sinceTime();
    else 
      std::cout << ". No data";
    std::cout << std::endl;
  }

}

struct Add {

  Add( cond::DbSession& db,  cond::IOVEditor & e) :
    pooldb(db), editor(e){}


  cond::DbSession pooldb;
  cond::IOVEditor & editor;

    void operator()(int i, std::string mess) {
      boost::shared_ptr<cond::IOVElement> data(new cond::IOVElement(i,mess));
      std::string tok = pooldb.storeObject(data.get(),"SomeWhere");
      editor.append(i,tok);
    }

};

int main(){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  try{
    cond::DbConnection connection;
    connection.configuration().setPoolAutomaticCleanUp( false );
    connection.configure();
    cond::DbSession pooldb = connection.createSession();
    pooldb.open("sqlite_file:mytest.db");
    pooldb.transaction().start(false);
    cond::IOVService iovmanager( pooldb );
    cond::IOVEditor editor( pooldb );
    editor.create(cond::timestamp,60);
    Add add(pooldb, editor);
    add(1,"pay1");
    add(21,"pay2");
    add(41,"pay3");
    pooldb.transaction().commit();
    std::string iovtok=editor.token();

    pooldb.transaction().start(true);

    std::cout<<"is 30 valid? "<<iovmanager.isValid(iovtok,30)<<std::endl;
    std::pair<cond::Time_t, cond::Time_t> v =  iovmanager.validity(iovtok,30);
    std::cout<<"30 validity "<< v.first << " : " << v.second <<std::endl;
    std::cout<<"30 token "<< iovmanager.payloadToken(iovtok,30)<<std::endl;

    pooldb.transaction().commit();
    // use Proxy
    {
      std::cout<<"test proxy "<<std::endl;
      cond::IOVProxy iov(pooldb,iovtok, true, false);
      std::cout << "size " << iov.size()
		<<", Time Type " << iov.timetype() << std::endl;
      std::for_each(iov.begin(),iov.end(),boost::bind(&print,_1));
      std::cout << "range 5,45" << std::endl;
      iov.setRange(5,45);
      std::for_each(iov.begin(),iov.end(),boost::bind(&print,_1));
      std::cout << "range 35,45" << std::endl;
      iov.setRange(35,45);
      std::for_each(iov.begin(),iov.end(),boost::bind(&print,_1));
      std::cout << "range 45,70" << std::endl;
      iov.setRange(45,70);
      std::for_each(iov.begin(),iov.end(),boost::bind(&print,_1));
      std::cout << "range 45,47" << std::endl;
      iov.setRange(45,47);
      std::for_each(iov.begin(),iov.end(),boost::bind(&print,_1));
    }
    {
      // test "copy shallow"
      cond::IOVProxy iov( pooldb,iovtok, true, false);
      pooldb.close();
      std::cout << "size " << iov.size()
		<<", Time Type " << iov.timetype() << std::endl;
      std::cout << "head 2" << std::endl;
      iov.head(2);
      std::for_each(iov.begin(),iov.end(),boost::bind(&print,_1));
      std::cout << "find 3,23,43,63" << std::endl;
      print(*iov.find(3));
      print(*iov.find(23));
      print(*iov.find(43));
      print(*iov.find(63));
      iov.setRange(1,90);
      print(*iov.find(63));
      iov.resetRange();
      std::cout << "back" << std::endl;
      print(*(iov.end()-1));
      iov.tail(1);
      print(*iov.begin());

    }
    {
      pooldb.open("sqlite_file:mytest.db");
      // test PayloadProxy
      cond::PayloadProxy<cond::IOVElement> data(pooldb,iovtok,false);
      printT(data,3);
      printT(data,21);
      printT(data,33);
      printT(data,43);
      printT(data,21);
      printT(data,63);
      std::cout << "test refresh" << std::endl;
      // test refresh
      if (data.refresh()) std::cout << "error!, what refresh..." << std::endl;
      std::cout << " size " << data.iov().size() << std::endl;
      {
        cond::DbSession pooldb2 = connection.createSession();
        pooldb2.open("sqlite_file:mytest.db");
        pooldb2.transaction().start(false);
	cond::IOVEditor editor(pooldb2,iovtok);
        Add add(pooldb2, editor);
        add(54,"pay54");
        pooldb2.transaction().commit();
      }
      if (!data.refresh()) std::cout << "error!, NO refresh..." << std::endl;
      std::cout << " size " << data.iov().size() << std::endl;
      printT(data,3);
      printT(data,21);
      printT(data,33);
      printT(data,43);
      printT(data,54);
      printT(data,57);
      printT(data,60);
      printT(data,63);
      for (long i=0; i<data.iov().size()+2; i+=2) 
	printN(data,i);
      /*
      // test Keylist
      cond::KeyList kl;
      kl.init(data);
      std::vector<unsigned long long> v(3); v[0]=21; v[1]=3; v[2]=[54];
      kl.load(v);
      for (size_t i=0; i<v.size();++i) {
	
      }
      */
    }
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
