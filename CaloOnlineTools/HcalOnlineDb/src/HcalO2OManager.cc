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
// $Id: HcalO2OManager.cc,v 1.4 2009/10/26 02:55:16 kukartse Exp $
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
int HcalO2OManager::getListOfPoolIovs(std::vector<uint32_t> & out, std::string tag, std::string connect){
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
  
  out.clear();
  try{
    myconnection.connect(session);
    cond::CoralTransaction& coraldb=myconnection.coralTransaction();
    cond::MetaData metadata_svc(coraldb);
    std::string token;
    coraldb.start(true);
    if(!metadata_svc.hasTag(tag)){
      //std::cout << "no such tag in the Pool database!" << std::endl;
      return -1;
    }
    token=metadata_svc.getToken(tag);
    coraldb.commit();
    cond::PoolTransaction& pooldb = myconnection.poolTransaction();
    {
      // FIXME: pre-CMSSW_33X
      cond::IOVProxy iov( pooldb, token, !details);
      cond::IOVService iovservice(pooldb);
      unsigned int counter=0;
      std::string payloadContainer=iovservice.payloadContainerName(token);
      //
      // FIXME: CMSSW_33X and later
      //cond::IOVProxy iov( myconnection, token, !details, details);
      //unsigned int counter=0;
      //std::string payloadContainer=iov.payloadContainerName();

      for (cond::IOVProxy::const_iterator ioviterator=iov.begin(); ioviterator!=iov.end(); ioviterator++) {
	out.push_back(ioviterator->since());
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
  return out.size();
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



int HcalO2OManager::getListOfOmdsIovs(std::vector<uint32_t> & out, std::string tagname){
  out.clear();
  static ConnectionManager conn;
  conn.connect();
  std::string query = " ";
  //query += "select iov, ";
  //query += "       i.cond_iov_record_id, ";
  //query += "       time ";
  //query += "from ( ";
  query += "      select iovs.interval_of_validity_begin as iov, ";
  query += "             min(iovs.record_insertion_time) time ";
  query += "      from cms_hcl_core_iov_mgmnt.cond_tags tags ";
  query += "      inner join cms_hcl_core_iov_mgmnt.cond_iov2tag_maps i2t ";
  query += "      on tags.cond_tag_id=i2t.cond_tag_id ";
  query += "      inner join cms_hcl_core_iov_mgmnt.cond_iovs iovs ";
  query += "      on i2t.cond_iov_record_id=iovs.cond_iov_record_id ";
  query += "where ";
  query += "      tags.tag_name=:1 ";
  query += "group by iovs.interval_of_validity_begin ";
  //query += "     ) ";
  //query += "inner join cms_hcl_core_iov_mgmnt.cond_iovs i ";
  //query += "on time=i.record_insertion_time ";
  query += "order by time asc ";
  int _n_iovs = 0;
  try {
    oracle::occi::Statement* stmt = conn.getStatement(query);
    //_____  set bind variables
    stmt->setString(1,tagname);
    oracle::occi::ResultSet *rs = stmt->executeQuery();
    while (rs->next()) {
      _n_iovs++;
      out.push_back(rs->getInt(1));
    }
  }
  catch (SQLException& e) {
    std::cerr << ::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()) << std::endl;
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Oracle  exception : %s",e.getMessage().c_str()));
  }
  conn.disconnect();
  return out.size();
}


void HcalO2OManager::getListOfNewIovs_test( void ){
  std::vector<uint32_t> omds, orcon, out;
  orcon.push_back(1);
  orcon.push_back(100);
  //orcon.push_back(80000);
  //orcon.push_back(90000);
  //orcon.push_back(100000);
  //orcon.push_back(199000);
  //orcon.push_back(199001);
  //orcon.push_back(199002);
  //orcon.push_back(199003);
  omds.push_back(1);
  omds.push_back(100);
  //omds.push_back(80000);
  //omds.push_back(90000);
  //omds.push_back(100000);
  //omds.push_back(199000);
  //omds.push_back(199001);
  //omds.push_back(199002);
  //omds.push_back(199004);
  if (getListOfNewIovs(out, omds, orcon) == -1){
    std::cout << "HcalO2OManager::getListOfNewIovs_test(): O2O is not possible" << std::endl;
  }
  else if (getListOfNewIovs(out, omds, orcon) == 0){
    std::cout << "HcalO2OManager::getListOfNewIovs_test(): O2O is not needed, the tag is up to date" << std::endl;
  }
  else{
    std::cout << "HcalO2OManager::getListOfNewIovs_test(): O2O is possible" << std::endl;
    std::cout << "HcalO2OManager::getListOfNewIovs_test(): " << out.size() << " IOVs to be copied to ORCON" << std::endl;
    std::copy (out.begin(),
	       out.end(),
	       std::ostream_iterator<uint32_t>(std::cout,"\n")
	       );
  }
}


int HcalO2OManager::getListOfNewIovs(std::vector<uint32_t> & iovs,
				     const std::vector<uint32_t> & omds_iovs,
				     const std::vector<uint32_t> & orcon_iovs){
  int result = -1; // default fail
  iovs.clear();
  if (omds_iovs.size() < orcon_iovs.size()) return result; // more IOVs in ORCON than in OMDS
  unsigned int _index = 0;
  for (std::vector<uint32_t>::const_iterator _iov = orcon_iovs.begin();
       _iov != orcon_iovs.end();
       ++_iov){
    _index = (int)(_iov - orcon_iovs.begin());
    if (omds_iovs[_index] != orcon_iovs[_index]){
      return result; // existing IOVs do not match
    }
    ++_index;
  }
  //
  //_____ loop over remaining OMDS IOVs
  //
  int _counter = 0; // count output IOVs
  for (;_index < omds_iovs.size();++_index){
    if (_index == 0){
      iovs.push_back(omds_iovs[_index]);
      ++_counter;
    }
    else if (omds_iovs[_index]>omds_iovs[_index-1]){
      iovs.push_back(omds_iovs[_index]);
      ++_counter;
    }
    else{
      return result;
    }
  }
  //if (_counter != 0) result = _counter;
  result = _counter;
  return result;
}
