// -*- C++ -*-
//
// Class  :     AlignmentRetriever
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Mon Apr 18 16:43:35 EDT 2005
// $Id: AlignmentRetriever.cc,v 1.3 2005/07/25 12:57:52 xiezhen Exp $
//

// system include files

// user include files
#include "CondCore/ESSources/interface/TrackerAlignmentRetriever.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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

//#include "FWCore/Framework/interface/SourceFactory.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentRcd.h"

  using namespace edm::eventsetup;
//
// constructors and destructor
//
TrackerAlignmentRetriever::TrackerAlignmentRetriever( const edm::ParameterSet&  pset)
{         
  std::cout<<"TrackerAlignmentRetriever::TrackerAlignmentRetriever"<<std::endl;
  //Tell Producer what we produce
  setWhatProduced(this);
  //Tell Finder what records we find
  findingRecord<TrackerAlignmentRcd>();
  // needed to connect to oracle
  //const std::string userNameEnv = "POOL_AUTH_USER=cms_vincenzo_dev";
  //::putenv( const_cast<char*>( userNameEnv.c_str() ) );
  //const std::string passwdEnv = "POOL_AUTH_PASSWORD=vinPass3";
  //::putenv( const_cast<char*>( passwdEnv.c_str() ) );
  std::string dbConnection=pset.retrieve("connect").getString();
  std::string iovname=pset.retrieve("iovname").getString();
  cond::MetaData meta(dbConnection);
  iovAToken_=meta.getToken(iovname);
  //  std::string m_dbConnection( "sqlite_file:trackeralign.db"); 
  seal::PluginManager::get()->initialise();
  
  // NOTE: in future if get a POOL throw, show we convert it to one the EDM knows?
  
  try{
    pool::POOLContext::loadComponent( "SEAL/Services/MessageService" );
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
    std::cout<<"iovAToken "<<iovAToken_<<std::endl;
    svc_->transaction().start(pool::ITransaction::READ);
    std::cout<<"Reading "<<std::endl;
    pool::Ref<cond::IOV> iovAlign(svc_.get(),iovAToken_);
    iovAlign_=iovAlign;
    std::cout<<"about to get out"<<std::endl;
  }catch(seal::Exception& e){
    std::cout << e.what() << std::endl;
  } catch ( std::exception& e ) {
    std::cout << e.what() << std::endl;
  } catch ( ... ) {
    std::cout << "Funny error" << std::endl;
  }
}
  
TrackerAlignmentRetriever::~TrackerAlignmentRetriever()
{
  svc_->transaction().commit();
  cat_->commit();
}

//
// member functions
//
const Alignments*
TrackerAlignmentRetriever::produce( const TrackerAlignmentRcd& )
{
  try{
    aligns_ = pool::Ref<Alignments>(svc_.get(),alignCid_);
    *aligns_;
  }catch(seal::Exception& e){
    std::cout << e.what() << std::endl;
  } catch ( std::exception& e ) {
    std::cout << e.what() << std::endl;
  } catch ( ... ) {
    std::cout << "Funny error" << std::endl;
  }
  std::cout<<"about to get out produce"<<std::endl;
  return &(*aligns_);
}

void
TrackerAlignmentRetriever::setIntervalFor( const EventSetupRecordKey&,
					  const edm::Timestamp& iTime, 
					  edm::ValidityInterval& oValidity)
{
  std::cout<<"TrackerAlignmentRetriever::setIntervalFor "<< iTime.value()<<std::endl;
  typedef std::map<int, std::string> IOVMap;
  typedef IOVMap::const_iterator iterator;
  try{
    unsigned long abtime=iTime.value()-edm::Timestamp::beginOfTime().value();
    std::cout<<"abtime "<<abtime<<std::endl;
    //iterator iEnd = iovAlign_->iov.lower_bound( iTime.value() );
    iterator iEnd = iovAlign_->iov.lower_bound( abtime );
    if( iEnd == iovAlign_->iov.end() ||  (*iEnd).second.empty() ) {
      //no valid data
      oValidity = edm::ValidityInterval(edm::Timestamp::endOfTime(),edm::Timestamp::endOfTime());
      std::cout<<"set to infinity"<<std::endl;
    } else {
      edm::Timestamp start=edm::Timestamp::beginOfTime();
      std::cout<<"beginoftime "<<start.value()<<std::endl;
      if (iEnd != iovAlign_->iov.begin()) {
	iterator iStart(iEnd); iStart--;
      	start = (*iStart).first+edm::Timestamp::beginOfTime().value();
      }
      std::cout<<"new start "<<start.value()<<std::endl;
      alignCid_ = (*iEnd).second;
      edm::Timestamp stop = (*iEnd).first+edm::Timestamp::beginOfTime().value();
      oValidity = edm::ValidityInterval( start, stop );
      std::cout<<"set to "<<start.value()<<" "<<stop.value()<<std::endl;
      std::cout << "align Cid " << alignCid_ << " valid from " << start.value() << " to " << stop.value() << std::endl;  
    }
  }catch(seal::Exception& e){
    std::cout << e.what() << std::endl;
  } catch ( std::exception& e ) {
    std::cout << e.what() << std::endl;
  } catch ( ... ) {
    std::cout << "Funny error" << std::endl;
  }
}
