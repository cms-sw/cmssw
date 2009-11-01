// -*- C++ -*-
//
// Package:     HcalOnlineDb
// Class  :     HcalO2OManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Gena Kukartsev
//         Created:  Sun Aug 16 20:44:05 CEST 2009
// $Id: HcalO2OManager.cc,v 1.2 2009/08/17 02:12:52 kukartse Exp $
//


#include "CaloOnlineTools/HcalOnlineDb/interface/HcalO2OManager.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

#include "CondFormats/Common/interface/PayloadWrapper.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVIterator.h"
#include "CondCore/IOVService/interface/IOVProxy.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/ConnectionManager.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseException.hh"
#include "xgi/Utils.h"
#include "toolbox/string.h"
#include "occi.h"

using namespace oracle::occi;



HcalO2OManager::HcalO2OManager()
{
  edmplugin::PluginManager::configure(edmplugin::standard::config());
}


HcalO2OManager::~HcalO2OManager()
{
}


// inspired by cmscond_list_iov
//
std::vector<std::string> HcalO2OManager::getListOfPoolTags(std::string connect){
  //edmplugin::PluginManager::configure(edmplugin::standard::config()); // in the constructor for now
  //
  std::string user("");
  std::string pass("");
  std::string tag;
  cond::DBSession* session=new cond::DBSession;
  //
  std::string userenv(std::string("CORAL_AUTH_USER=")+user);
  std::string passenv(std::string("CORAL_AUTH_PASSWORD=")+pass);
  ::putenv(const_cast<char*>(userenv.c_str()));
  ::putenv(const_cast<char*>(passenv.c_str()));
   session->configuration().setAuthenticationMethod( cond::Env );    
   session->configuration().setMessageLevel( cond::Error );
  //
  session->open();
  cond::ConnectionHandler::Instance().registerConnection(connect,*session,-1);
  cond::Connection & myconnection = *cond::ConnectionHandler::Instance().getConnection(connect);
  
  std::vector<std::string> alltags;
  try{
    myconnection.connect(session);
    cond::CoralTransaction& coraldb=myconnection.coralTransaction();
    cond::MetaData metadata_svc(coraldb);
    coraldb.start(true);
    metadata_svc.listAllTags(alltags);
    coraldb.commit();
    myconnection.disconnect();
  }catch(cond::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(std::exception& er){
    std::cout<<er.what()<<std::endl;
  }
  delete session;
  return alltags;
}



// inspired by cmscond_list_iov
//
std::vector<uint32_t> HcalO2OManager::getListOfPoolIovs(std::string tag, std::string connect){
  //edmplugin::PluginManager::configure(edmplugin::standard::config()); // in the constructor for now
  std::string user("");
  std::string pass("");
  bool details=false;
  cond::DBSession* session=new cond::DBSession;
  //
  std::string userenv(std::string("CORAL_AUTH_USER=")+user);
  std::string passenv(std::string("CORAL_AUTH_PASSWORD=")+pass);
  ::putenv(const_cast<char*>(userenv.c_str()));
  ::putenv(const_cast<char*>(passenv.c_str()));
  session->configuration().setAuthenticationMethod( cond::Env );    
  session->configuration().setMessageLevel( cond::Error );
  //
  session->open();
  cond::ConnectionHandler::Instance().registerConnection(connect,*session,-1);
  cond::Connection & myconnection = *cond::ConnectionHandler::Instance().getConnection(connect);
  
  std::vector<uint32_t> allIovs;
  try{
    myconnection.connect(session);
    cond::CoralTransaction& coraldb=myconnection.coralTransaction();
    cond::MetaData metadata_svc(coraldb);
    std::string token;
    coraldb.start(true);
    token=metadata_svc.getToken(tag);
    coraldb.commit();
    cond::PoolTransaction& pooldb = myconnection.poolTransaction();
    {
      //cond::IOVProxy iov( pooldb, token, !details);
      // FIXME: after CMSSW_3_2_3 use this:
      cond::IOVProxy iov( myconnection, token, !details, details);
      //cond::IOVService iovservice(pooldb);
      unsigned int counter=0;

      //std::string payloadContainer=iovservice.payloadContainerName(token);
      // FIXME: after CMSSW_3_2_3 use this:
      std::string payloadContainer=iov.payloadContainerName();
      for (cond::IOVProxy::const_iterator ioviterator=iov.begin(); ioviterator!=iov.end(); ioviterator++) {
	allIovs.push_back(ioviterator->since());
	++counter;
      }
    }
    myconnection.disconnect();
  }catch(cond::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(std::exception& er){
    std::cout<<er.what()<<std::endl;
  }
  delete session;
  return allIovs;
}



std::vector<std::string> HcalO2OManager::getListOfOmdsTags(){
  std::vector<std::string> alltags;
  static ConnectionManager conn;
  conn.connect();
  std::string query = "select ";
  query            += "       channel_map_id,subdet,ieta,iphi,depth ";
  query            += "from ";
  query            += "       cms_hcl_hcal_cond.hcal_channels ";
  query            += "where ";
  query            += "       subdet='HB' or subdet='HE' or subdet='HF' or subdet='HO' ";
  int _n_tags = 0;
  try {
    oracle::occi::Statement* stmt = conn.getStatement(query);
    oracle::occi::ResultSet *rs = stmt->executeQuery();
    while (rs->next()) {
      _n_tags++;
      //alltags.push_back(rs->getString(1));
    }
  }
  catch (SQLException& e) {
    std::cerr << ::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()) << std::endl;
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
  }
  conn.disconnect();
  return alltags;
}



std::vector<uint32_t> HcalO2OManager::getListOfOmdsIovs(std::string tagname){
  std::vector<uint32_t> allIovs;

  return allIovs;
}
