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

namespace {
  void print(cond::IOVElement const & e) {
    std::cout<<"payloadToken "<< e.payloadToken()
	     <<", since "<< e.since()
	     <<", till "<< e.till()
	     << std::endl;
  }
}

int main(){
  try{
    cond::DBSession* session=new cond::DBSession;
    session->open();
    cond::Connection myconnection("sqlite_file:mytest.db",0); 
    myconnection.connect(session);
    cond::PoolTransaction& pooldb=myconnection.poolTransaction();
    pooldb.start(false);
    cond::IOVService iovmanager(pooldb,cond::timestamp);
    cond::IOVEditor* editor=iovmanager.newIOVEditor();
    editor->create(1,cond::timestamp);
    editor->insert(20,"pay1tok");
    editor->insert(40,"pay2tok");
    editor->insert(60,"pay3tok");
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
      std::cout<<"payloadToken "<<it->payloadToken();
      std::cout<<", since "<<it->validity().first;
      std::cout<<", till "<<it->validity().second<<std::endl;
    }
    delete it;
    // backward
    it=iovmanager.newIOVIterator(iovtok,cond::IOVService::backwardIter);
    std::cout<<"test reverse iterator "<<std::endl;
    while( it->next() ){
      std::cout<<"payloadToken "<<it->payloadToken();
      std::cout<<", since "<<it->validity().first;
      std::cout<<", till "<<it->validity().second<<std::endl;
    }
    delete it;

    std::cout<<"is 30 valid? "<<iovmanager.isValid(iovtok,30)<<std::endl;
    pooldb.commit();
    delete editor;
    // use Proxy
    {
      std::cout<<"test proxy "<<std::endl;
      cond::IOVProxy iov(pooldb,iovtok, true);
      std::cout << "size " << iov.size()
		<<", Time Type " << iov.timetype() << std::endl;
      std::for_each(iov.begin(),iov.end(),boost::bind(&print,_1));
    }
    myconnection.disconnect();
    delete session;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
}
