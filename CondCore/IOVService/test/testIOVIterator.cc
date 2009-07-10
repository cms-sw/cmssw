#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/IOVService/interface/IOVIterator.h"
#include "CondCore/IOVService/interface/IOVProxy.h"
#include <iostream>
#include <algorithm>
#include <boost/bind.hpp>

#include "CondCore/DBCommon/interface/TypedRef.h"
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
  
  void print(cond::PayloadProxy<cond::IOVElement> & data, cond::Time_t time) {
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

}

struct Add {

  Add( cond::PoolTransaction& db,  cond::IOVEditor & e) :
    pooldb(db), editor(e){}


  cond::PoolTransaction& pooldb;
  cond::IOVEditor & editor;

  void operator()(int i, std::string mess) {
    cond::TypedRef<cond::IOVElement> ref(pooldb,new cond::IOVElement(i,mess));
    ref.markWrite("SomeWhere");
    editor.append(i,ref.token());
  }

};

int main(){
  try{
    cond::DBSession* session=new cond::DBSession;
    session->open();
    cond::Connection myconnection("sqlite_file:mytest.db",0); 
    myconnection.connect(session);
    cond::PoolTransaction& pooldb=myconnection.poolTransaction();
    pooldb.start(false);
    cond::IOVService iovmanager(pooldb);
    cond::IOVEditor* editor=iovmanager.newIOVEditor();
    editor->create(cond::timestamp,60);
    Add add(pooldb,*editor);
    add(1,"pay1");
    add(21,"pay2");
    add(41,"pay3");
    pooldb.commit();
    std::string iovtok=editor->token();
    ///test iterator
    // forward
    cond::IOVIterator* it=iovmanager.newIOVIterator(iovtok);
    std::cout<<"test forward iterator "<<std::endl;
    pooldb.start(true);
    std::cout << "size " << it->size()
	      <<", Time Type " << it->timetype() << std::endl;
    while( it->next() ){
      std::cout<<"payloadToken "<< oid(it->payloadToken());
      std::cout<<", since "<<it->validity().first;
      std::cout<<", till "<<it->validity().second<<std::endl;
    }
    delete it;
    // backward
    it=iovmanager.newIOVIterator(iovtok,cond::IOVService::backwardIter);
    std::cout<<"test reverse iterator "<<std::endl;
    while( it->next() ){
      std::cout<<"payloadToken "<< oid(it->payloadToken());
      std::cout<<", since "<<it->validity().first;
      std::cout<<", till "<<it->validity().second<<std::endl;
    }
    delete it;

    std::cout<<"is 30 valid? "<<iovmanager.isValid(iovtok,30)<<std::endl;
    std::pair<cond::Time_t, cond::Time_t> v =  iovmanager.validity(iovtok,30);
    std::cout<<"30 validity "<< v.first << " : " << v.second <<std::endl;
    std::cout<<"30 token "<< iovmanager.payloadToken(iovtok,30)<<std::endl;

    pooldb.commit();
    delete editor;
    // use Proxy
    {
      std::cout<<"test proxy "<<std::endl;
      cond::IOVProxy iov(myconnection,iovtok, true, false);
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
      cond::IOVProxy iov(myconnection,iovtok, true, false);
      myconnection.disconnect();
      std::cout << "size " << iov.size()
		<<", Time Type " << iov.timetype() << std::endl;
      iov.head(2);
      std::for_each(iov.begin(),iov.end(),boost::bind(&print,_1));
      std::cout << "range 3,23,43,63" << std::endl;
      print(*iov.find(3));
      print(*iov.find(23));
      print(*iov.find(43));
      print(*iov.find(63));
      iov.setRange(1,90);
      print(*iov.find(63));
    }
    {
      // test PayloadProxy
      cond::PayloadProxy<cond::IOVElement> data(myconnection,iovtok,false);
      print(data,3);
      print(data,23);
      print(data,33);
      print(data,43);
      print(data,63);
      std::cout << "test refresh" << std::endl;
      // test refresh
      if (data.refresh()) std::cout << "error!, what refresh..." << std::endl;
      std::cout << " size " << data.iov().size() << std::endl;
      {
	myconnection.connect(session);
	cond::PoolTransaction& pooldb2=myconnection.poolTransaction();
	pooldb2.start(false);
	cond::IOVService iovmanager2(pooldb2);
	cond::IOVEditor* editor=iovmanager2.newIOVEditor(iovtok);
	Add add(pooldb2,*editor);
	add(54,"pay54");
	delete editor;
	pooldb2.commit();
      }
      if (!data.refresh()) std::cout << "error!, NO refresh..." << std::endl;
      print(data,3);
      print(data,23);
      print(data,33);
      print(data,43);
      print(data,57);
      print(data,63);
    }
    delete session;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
