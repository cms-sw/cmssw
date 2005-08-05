// -*- C++ -*-
//
// Package:     EDMProto
// Class  :     PedestalRetriever
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Mon Apr 18 16:43:35 EDT 2005
// $Id: HcalPedestalsRetriever.cc,v 1.1 2005/08/01 13:10:20 xiezhen Exp $
//

// system include files

// user include files
//#include "Reflection/Class.h"
#include "PluginManager/PluginManager.h"
#include "POOLCore/Token.h"
#include "FileCatalog/URIParser.h"
#include "FileCatalog/FCSystemTools.h"
#include "FileCatalog/IFileCatalog.h"
#include "StorageSvc/DbType.h"
#include "PersistencySvc/DatabaseConnectionPolicy.h"
#include "PersistencySvc/ISession.h"
#include "PersistencySvc/ITransaction.h"
#include "PersistencySvc/IDatabase.h"
#include "PersistencySvc/Placement.h"
#include "DataSvc/DataSvcFactory.h"
#include "DataSvc/IDataSvc.h"
#include "DataSvc/ICacheSvc.h"
#include "DataSvc/Ref.h"

//#include "SealUtil/SealTimer.h"
//#include "SealUtil/TimingReport.h"
//#include "SealUtil/SealHRRTChrono.h"
//#include "SealUtil/SealHRChrono.h"

#include "POOLCore/POOLContext.h"
#include "SealKernel/Exception.h"

#include "CondCore/ESSources/interface/HcalPedestalsRetriever.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondFormats/DataRecord/interface/HCalPedestalsRcd.h"
  using namespace edm::eventsetup;

HcalPedestalsRetriever::HcalPedestalsRetriever( const edm::ParameterSet&  pset)
{         
  //  std::cout<<"PedestalRetriever::PedestalRetriever"<<std::endl;
  //Tell Producer what we produce
  setWhatProduced(this);
  //Tell Finder what records we find
  findingRecord<HCalPedestalsRcd>();
  std::string dbConnection=pset.retrieve("connect").getString();
  std::string iovname=pset.retrieve("iovname").getString();
  cond::MetaData meta(dbConnection);
  iovAToken_=meta.getToken(iovname);
  seal::PluginManager::get()->initialise();
  //seal::TimingReport timRep;
  
  // NOTE: in future if get a POOL throw, show we convert it to one the EDM knows?
  
  try{
    pool::POOLContext::loadComponent( "SEAL/Services/MessageService" );
    pool::POOLContext::setMessageVerbosityLevel( seal::Msg::Error );
    // needed to connect to oracle
    pool::POOLContext::loadComponent( "POOL/Services/EnvironmentAuthenticationService" );
    
    pool::URIParser p;
    p.parse();
    
    // the required lifetime of the file catalog is the same of the  srv_ or longer  
    cat_.reset(new pool::IFileCatalog);
    cat_->addReadCatalog(p.contactstring());
    cat_->connect();
    
    cat_->start();
    
    svc_ = std::auto_ptr<pool::IDataSvc>(pool::DataSvcFactory::instance(&(*cat_)));
    // Define the policy for the implicit file handling
    pool::DatabaseConnectionPolicy policy;
    policy.setWriteModeForNonExisting(pool::DatabaseConnectionPolicy::CREATE);
    // policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::OVERWRITE);
    //policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::UPDATE);
    svc_->session().setDefaultConnectionPolicy(policy);
    //std::cout<<"iovAToken "<<iovAToken_<<std::endl;
    svc_->transaction().start(pool::ITransaction::READ);
    //std::cout<<"Reading "<<std::endl;
    pool::Ref<cond::IOV> iovped(svc_.get(),iovAToken_);
    iovped_=iovped;
    //std::cout<<"about to get out"<<std::endl;
  }catch(seal::Exception& e){
    std::cout << e.what() << std::endl;
  } catch ( std::exception& e ) {
    std::cout << e.what() << std::endl;
  } catch ( ... ) {
    std::cout << "Funny error" << std::endl;
  }
}
  
HcalPedestalsRetriever::~HcalPedestalsRetriever()
{
  std::cout<<"HcalPedestalRetriever::~HcalPedestalsRetriever"<<std::endl;
  svc_->transaction().commit();
  std::cout << "commit catalog" << std::endl;
  cat_->commit();
}

//
// member functions
//
const Pedestals*
HcalPedestalsRetriever::produce( const HCalPedestalsRcd& )
{
  std::cout<<"HcalPedestalRetriever::produce"<<std::endl;
  try{
    peds_ = pool::Ref<Pedestals>(svc_.get(),pedCid_);
    *peds_;
  }catch(seal::Exception& e){
    std::cout << e.what() << std::endl;
  } catch ( std::exception& e ) {
    std::cout << e.what() << std::endl;
  } catch ( ... ) {
    std::cout << "Funny error" << std::endl;
  }
  std::cout<<"about to get out produce"<<std::endl;
  return &(*peds_);
}

void
HcalPedestalsRetriever::setIntervalFor( const EventSetupRecordKey&,
				   const edm::IOVSyncValue& iTime, 
				   edm::ValidityInterval& oValidity)
{
  typedef std::map<int, std::string> IOVMap;
  typedef IOVMap::const_iterator iterator;
  try{
    unsigned long abtime=iTime.collisionID()-edm::IOVSyncValue::beginOfTime().collisionID();
    iterator iEnd = iovped_->iov.lower_bound( abtime );
    if( iEnd == iovped_->iov.end() ||  (*iEnd).second.empty() ) {
      //no valid data
      oValidity = edm::ValidityInterval(edm::IOVSyncValue::endOfTime(),edm::IOVSyncValue::endOfTime());
    } else {
      unsigned long starttime=edm::IOVSyncValue::beginOfTime().collisionID();
      if (iEnd != iovped_->iov.begin()) {
	iterator iStart(iEnd); iStart--;
      	starttime = (*iStart).first+edm::IOVSyncValue::beginOfTime().collisionID();
      }
      pedCid_ = (*iEnd).second;
      edm::IOVSyncValue start( starttime );
      edm::IOVSyncValue stop ((*iEnd).first+edm::IOVSyncValue::beginOfTime().collisionID());
      oValidity = edm::ValidityInterval( start, stop );
    }
  }catch(seal::Exception& e){
    std::cout << e.what() << std::endl;
  } catch ( std::exception& e ) {
    std::cout << e.what() << std::endl;
  } catch ( ... ) {
    std::cout << "Funny error" << std::endl;
  }
}
