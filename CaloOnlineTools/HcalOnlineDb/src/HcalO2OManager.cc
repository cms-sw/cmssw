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
// $Id: HcalO2OManager.cc,v 1.43 2012/02/15 15:26:07 andreasp Exp $
//


#include "CaloOnlineTools/HcalOnlineDb/interface/HcalO2OManager.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/DBCommon/interface/DbConnection.h" 	 
#include "CondCore/DBCommon/interface/DbSession.h" 	 
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"

#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

#include "CondCore/IOVService/interface/IOVProxy.h"

#include "CaloOnlineTools/HcalOnlineDb/interface/ConnectionManager.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseException.hh"
#include "xgi/Utils.h"
#include "toolbox/string.h"
#include "OnlineDB/Oracle/interface/Oracle.h"


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
std::vector<std::string> HcalO2OManager::getListOfPoolTags(std::string connect, std::string auth_path){
  //edmplugin::PluginManager::configure(edmplugin::standard::config()); // in the constructor for now
  //
  // FIXME: how to add auth_path authentication to this? See v1.25 for the functionality using old API
  std::cout << "===> WARNING! auth_path is specified as " << auth_path;
  std::cout << " but is not used explicitely. Is it being used at all?"  << std::endl;
  cond::DbConnection conn;
  //conn.configure( cond::CmsDefaults );
  conn.configuration().setAuthenticationPath(auth_path);
  //conn.configuration().setMessageLevel( coral::Debug );
  conn.configure();
  cond::DbSession session = conn.createSession();
  session.open(connect);
  std::vector<std::string> alltags;
  try{
    cond::MetaData metadata_svc(session);
    cond::DbScopedTransaction tr(session);
    tr.start(true);
    metadata_svc.listAllTags(alltags);
    tr.commit();
  }catch(cond::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(std::exception& er){
    std::cout<<er.what()<<std::endl;
  }
  return alltags;
}



// inspired by cmscond_list_iov
//
int HcalO2OManager::getListOfPoolIovs(std::vector<uint32_t> & out,
				      std::string tag, 
				      std::string connect,
				      std::string auth_path){
  //edmplugin::PluginManager::configure(edmplugin::standard::config()); // in the constructor for now
  // FIXME: how to add auth_path authentication to this? See v1.25 for the functionality using old API  
  std::cout << "===> WARNING! auth_path is specified as " << auth_path;
  std::cout << " but is not used explicitely. Is it being used at all?"  << std::endl;
  cond::DbConnection conn;
  //conn.configure( cond::CmsDefaults );
  conn.configuration().setAuthenticationPath(auth_path);
  //conn.configuration().setMessageLevel( coral::Debug );
  conn.configure();
  cond::DbSession session = conn.createSession();
  session.open(connect);
  out.clear();
  try{
    cond::MetaData metadata_svc(session);
    cond::DbScopedTransaction tr(session);
     tr.start(true);
     std::string token;
     if(!metadata_svc.hasTag(tag)){
       //std::cout << "no such tag in the Pool database!" << std::endl;
       return -1;
     }
     token=metadata_svc.getToken(tag);
     cond::IOVProxy iov(session, token);
     unsigned int counter=0;
     
     for (cond::IOVProxy::const_iterator ioviterator=iov.begin(); ioviterator!=iov.end(); ioviterator++) {
       out.push_back(ioviterator->since());
       ++counter;
     }
     tr.commit();
  }
  catch(cond::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(std::exception& er){
    std::cout<<er.what()<<std::endl;
  }
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

  // OMDS tag may not have the first IOV=1
  int _orcon_index_offset = 0;
  if (omds_iovs.size() > 0 &&
      orcon_iovs.size() > 0 &&
      orcon_iovs[0] == 1 &&
      omds_iovs[0] != 1){
    std::cout << std::endl << "HcalO2OManager: First IOV in the OMDS tag is not 1," << std::endl;
    std::cout << "HcalO2OManager: while it must be 1 in the offline tag." << std::endl;
    std::cout << "HcalO2OManager: O2O will assume that IOV=1 in the offline tag" << std::endl;
    std::cout << "HcalO2OManager: is filled with some safe default." << std::endl;
    std::cout << "HcalO2OManager: IOV=1 will be ignored, and O2O will proceeed" << std::endl;
    std::cout << "HcalO2OManager: as long as other inconsistencies are not detected." << std::endl << std::endl;
    _orcon_index_offset = 1; // skip the first IOV in ORCON because it
    //                       // is 1 while OMDS doesn't have IOV=1
  } 
  if (omds_iovs.size()+_orcon_index_offset < orcon_iovs.size()){
    std::cout << "HcalO2OManager: too many IOVs in the Pool tag, cannot sync, exiting..." << std::endl;
    return result;
  }

  // loop over all OMDS IOVs
  unsigned int _index = 0;

  bool enforce_strict_matching = false; // set to true if the strict IOV matching is desired, see description in the comments below

  for (std::vector<uint32_t>::const_iterator _iov = orcon_iovs.begin();
       _iov != orcon_iovs.end();
       ++_iov){
    _index = (int)(_iov - orcon_iovs.begin());

    // special case when the first Pool IOV = 1 (must always be true)
    // but OMDS IOVs do not start with IOV = 1
    // This can be a ligitimate mismatch when the OMDS tag is created for
    // current conditions without regard to the history
    // In such cases, when creating a copy of this tag in offline,
    // O2O copies the first IOV from OMDS and assigns it as IOV=1.
    // With later sync passes, O2O must skip the first offline IOV = 1
    if (_orcon_index_offset == 1 && _index == 0) continue;

    // current pair of OMDS-offline IOVs does not match
    // There are several options in such case:
    //
    //   - with strict matching, the program should quit, as it is not possible
    //     to keep the tag in sync between OMDS and offline because
    //     offline tags do not allow fixes, only updates
    //
    //   - intermediate solution is to find the latest IOV in the offline tag
    //     and append any IOVs from the OMDS tag that come after it
    //

    if (omds_iovs[_index-_orcon_index_offset] != orcon_iovs[_index]){

      std::cout << std::endl;
      std::cout << "HcalO2OManager: existing IOVs do not match, cannot sync in the strict sense." << std::endl;
      std::cout << "HcalO2OManager: mismatched pair is (OMDS/offline): " << omds_iovs[_index-_orcon_index_offset] << "/" << orcon_iovs[_index] << std::endl;
      std::cout << "HcalO2OManager: In the strict sense, the SYNCHRONIZATION OF THIS TAG HAS FAILED!" << std::endl;
      std::cout << "HcalO2OManager: As an interim solution, I will still copy to the offline tag" << std::endl;
      std::cout << "HcalO2OManager: those IOV from the OMDS tag that are newer than the last IOV" << std::endl;
      std::cout << "HcalO2OManager: currently in the offline tag. " << std::endl;

      // existing IOVs do not match

      if (enforce_strict_matching){
	return result;
      }
      else{
	break; // _index now contains the last "valid" OMDS IOV
      }

    }
    ++_index;
  }


  //
  //_____ loop over remaining OMDS IOVs
  //
  //std::cout << "HcalO2OManager: DEBUG: " << std::endl;
  int _counter = 0; // count output IOVs
  uint32_t _lastOrconIov = orcon_iovs[orcon_iovs.size()-1];

  for (;_index < omds_iovs.size();++_index){

    uint32_t _aIov = omds_iovs[_index];

    if (_index == 0 && _aIov > _lastOrconIov){ // can only copy later IOVs
      iovs.push_back(_aIov);
      ++_counter;
    }
    else if (omds_iovs[_index]>omds_iovs[_index-1] &&
	     _aIov > _lastOrconIov){  // can only copy later IOVs
      iovs.push_back(omds_iovs[_index]);
      ++_counter;
    }
    else{
      if (enforce_strict_matching){
	return result;
      }
      else{
	continue;
      }
    }
  }
  //if (_counter != 0) result = _counter;
  result = _counter;
  return result;
}


// get list of IOVs to update for a given tag, or report impossible
// return:
// list of IOVs as first argument,
// number of IOVs to update as return value
// -1 if tag is inconsistent with the update
// 0 if everything's up to date already
int HcalO2OManager::getListOfUpdateIovs(std::vector<uint32_t> & _iovs,
					std::string _tag,
					std::string pool_connect_string,
					std::string pool_auth_path
					){
  //std::cout << "DEBUG: " << pool_connect_string << std::endl;
  std::vector<uint32_t> omds_iovs;
  std::vector<uint32_t> pool_iovs;
  getListOfOmdsIovs(omds_iovs, _tag);
  getListOfPoolIovs(pool_iovs, _tag, pool_connect_string, pool_auth_path);
  int n_iovs = getListOfNewIovs(_iovs,
				omds_iovs,
				pool_iovs);
  if (n_iovs == -1){
    std::cout << "HcalO2OManager: O2O is not possible" << std::endl;
  }
  else if (n_iovs == 0){
    std::cout << "HcalO2OManager: O2O is not needed, the tag is up to date" << std::endl;
  }
  else{
    edm::LogInfo("HcalO2OManager") << "These IOVs are to be updated:" << std::endl;
    for (std::vector<uint32_t>::const_iterator iov = _iovs.begin();
	 iov != _iovs.end();
	 ++iov){
      std::cout << "O2O_IOV_LIST: " << *iov << std::endl;
    }
  }
  return n_iovs;
}
