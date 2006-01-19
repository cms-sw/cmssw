
/**
   \class HcalDbPOOL
   \brief IO for POOL instances of Hcal Calibrations
   \author Fedor Ratnikov Oct. 28, 2005
   $Id: HcalDbPool.h,v 1.3 2005/12/15 23:37:56 fedor Exp $
*/

// pool
#include "PluginManager/PluginManager.h"
#include "POOLCore/POOLContext.h"
#include "POOLCore/Token.h"
#include "FileCatalog/URIParser.h"
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
#include "RelationalAccess/RelationalException.h"
#include "Collection/Collection.h"
#include "AttributeList/AttributeList.h"
#include "FileCatalog/FCSystemTools.h"
#include "FileCatalog/FCException.h"
#include "FileCatalog/IFCAction.h"

// conditions
#include "CondCore/IOVService/interface/IOV.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

#include "CondFormats/HcalObjects/interface/AllObjects.h"

#include "CondTools/Hcal/interface/HcalDbPool.h"

template <class T>
bool HcalDbPool::storeObject (T* fObject, const std::string& fContainer, pool::Ref<T>* fRef) {
  if (!fRef->isNull ()) {
    std::cerr << "storeObject-> Ref is not empty. Ignore." << std::endl;
    return false;
  }
  try {
    service ()->transaction().start(pool::ITransaction::UPDATE);
    
    *fRef = pool::Ref <T> (service (), fObject);
    mPlacement->setContainerName (fContainer);
    fRef->markWrite (*mPlacement);
    service ()->transaction().commit();
  }
  catch (...) {
    std::cerr << "storeObject->  error "  << std::endl;
    return false;
  }
  return true;
} 

template <class T>
bool HcalDbPool::updateObject (T* fObject, pool::Ref<T>* fUpdate) {
  try {
    service ()->transaction().start(pool::ITransaction::UPDATE);
    if (fObject) *(fUpdate->ptr ()) = *fObject; // update object
    fUpdate->markUpdate();
    service ()->transaction().commit();
  }
  catch (...) {
    std::cerr << "updateObject->  error "  << std::endl;
    return false;
  }
  return true;
}

template <class T>
bool HcalDbPool::updateObject (pool::Ref<T>* fUpdate) {
  return updateObject ((T*)0, fUpdate);
}

template <class T>
bool HcalDbPool::storeIOV (const pool::Ref<T>& fObject, unsigned fMaxRun, pool::Ref<cond::IOV>* fIov) {
  unsigned maxRun = fMaxRun == 0 ? 0xffffffff : fMaxRun;
  if (fIov->isNull ()) {
    cond::IOV* newIov = new cond::IOV ();
    newIov->iov.insert (std::make_pair (maxRun, fObject.toString ()));
    return storeObject (newIov, "IOV", fIov);
  }
  else {
    (*fIov)->iov.insert (std::make_pair (maxRun, fObject.toString ()));
    return updateObject (fIov);
  }
}

template <class T>
bool HcalDbPool::getObject (const pool::Ref<cond::IOV>& fIOV, unsigned fRun, pool::Ref<T>* fObject) {
  if (!fIOV.isNull ()) {
    // scan IOV, search for valid data
    for (std::map<unsigned long,std::string>::iterator iovi = fIOV->iov.begin (); iovi != fIOV->iov.end (); iovi++) {
      if (fRun <= iovi->first) {
	std::string token = iovi->second;
	return getObject (token, fObject);
      }
    }
    std::cerr << "getObject-> no object for run " << fRun << " is found" << std::endl;
  }
  else {
    std::cerr << "getObject-> IOV reference is not set" << std::endl;
  }
  return false;
}

template <class T> 
bool HcalDbPool::getObject (const std::string& fToken, pool::Ref<T>* fObject) {
  try {
    *fObject = pool::Ref <T> (service (), fToken);
    service ()->transaction().start(pool::ITransaction::READ);
    fObject->isNull ();
    service ()->transaction().commit();
  }
  catch( const pool::RelationalTableNotFound& e ){
    std::cerr << "getObject-> pool::RelationalTableNotFound Exception" << std::endl;
  }
  catch (const seal::Exception& e) {
    std::cerr<<"getObject-> seal exception: " << e.what() << std::endl;
  }
  catch(...){
    std::cerr << "getObject-> Funny error" << std::endl;
  }
  return !(fObject->isNull ());
}

template <class T>
bool HcalDbPool::getObject_ (T* fObject, const std::string& fTag, int fRun) {
  std::string metadataToken = metadataGetToken (fTag);
  if (metadataToken.empty ()) {
    std::cerr << "HcalDbPool::getObject ERROR-> Can not find metadata for tag " << fTag << std::endl;
    return false;
  }
  pool::Ref<cond::IOV> iov;
  getObject (metadataToken, &iov);
  if (iov.isNull ()) {
    std::cerr << "HcalDbPool::getObject ERROR: can not find IOV for token " << metadataToken << std::endl;;
    return false;
  }
  pool::Ref<T> ref;
  if (getObject (iov, fRun, &ref)) {
    *fObject = *ref; // make copy
    return true;
  }
  return false;
}

template <class T>
bool HcalDbPool::putObject_ (T* fObject, const std::string& fClassName, const std::string& fTag, int fRun) {
  std::string metadataToken = metadataGetToken (fTag);
  pool::Ref<cond::IOV> iov;
  if (!metadataToken.empty ()) {
    getObject (metadataToken, &iov);
    if (iov.isNull ()) {
      std::cerr << "HcalDbPool::putObject ERROR: can not find IOV for token " << metadataToken << std::endl;;
      return false;
    }
  }
  bool create = iov.isNull ();
  pool::Ref<T> ref;
  if (!storeObject (fObject, fClassName, &ref) ||
      !storeIOV (ref, fRun, &iov)) {
    std::cerr << "ERROR: failed to store object or its IOV" << std::endl;
    return false;
  }
  if (create) {
    std::string token = iov.toString ();
    metadataSetTag (fTag, token);
  }
  return true;
}

HcalDbPool::HcalDbPool (const std::string& fConnect)
  : mConnect (fConnect) {
}

pool::IDataSvc* HcalDbPool::service ()
{
  
  if (!mService.get ()) {
    seal::PluginManager::get()->initialise();
    pool::POOLContext::loadComponent( "SEAL/Services/MessageService" );
    pool::POOLContext::loadComponent( "POOL/Services/EnvironmentAuthenticationService" );
    
    pool::URIParser parser;
    parser.parse();
    
    mCatalog.reset (new pool::IFileCatalog ());
    mCatalog->setWriteCatalog(parser.contactstring());
    mCatalog->connect();
    mCatalog->start();
    
    mService.reset (pool::DataSvcFactory::create (&*mCatalog));
    
    pool::DatabaseConnectionPolicy policy;  
    policy.setWriteModeForNonExisting(pool::DatabaseConnectionPolicy::CREATE);
    policy.setWriteModeForExisting(pool::DatabaseConnectionPolicy::UPDATE); 
    mService->session().setDefaultConnectionPolicy(policy);
    mPlacement.reset (new pool::Placement ());
    mPlacement->setDatabase(mConnect, pool::DatabaseSpecification::PFN); 
    mPlacement->setTechnology(pool::POOL_RDBMS_StorageType.type());
  }
  return mService.get ();
}

std::string HcalDbPool::metadataGetToken (const std::string& fTag) {
  std::string result;
  try {
    cond::MetaData md (mConnect);
    result = md.getToken (fTag);
  }
  catch (...) {
    result.clear ();
  }
  return result;
}

bool HcalDbPool::metadataSetTag (const std::string& fTag, const std::string& fToken) {
  bool result = false;
  try {
    cond::MetaData md (mConnect);
    result = md.addMapping (fTag, fToken);
  }
  catch (...) {
    result = false;
  }
  return result;
}

HcalDbPool::~HcalDbPool () {
  mService->session().disconnectAll();
  mCatalog->commit ();
  mCatalog->disconnect ();
}

bool HcalDbPool::getObject (HcalPedestals* fObject, const std::string& fTag, int fRun) {return getObject_ (fObject, fTag, fRun);}
bool HcalDbPool::putObject (HcalPedestals* fObject, const std::string& fTag, int fRun) {return putObject_ (fObject, "HcalPedestals", fTag, fRun);}
bool HcalDbPool::getObject (HcalPedestalWidths* fObject, const std::string& fTag, int fRun) {return getObject_ (fObject, fTag, fRun);}
bool HcalDbPool::putObject (HcalPedestalWidths* fObject, const std::string& fTag, int fRun) {return putObject_ (fObject, "HcalPedestalWidths", fTag, fRun);}
bool HcalDbPool::getObject (HcalGains* fObject, const std::string& fTag, int fRun) {return getObject_ (fObject, fTag, fRun);}
bool HcalDbPool::putObject (HcalGains* fObject, const std::string& fTag, int fRun) {return putObject_ (fObject, "HcalGains", fTag, fRun);}
bool HcalDbPool::getObject (HcalGainWidths* fObject, const std::string& fTag, int fRun) {return getObject_ (fObject, fTag, fRun);}
bool HcalDbPool::putObject (HcalGainWidths* fObject, const std::string& fTag, int fRun) {return putObject_ (fObject, "HcalGainWidths", fTag, fRun);}
bool HcalDbPool::getObject (HcalQIEData* fObject, const std::string& fTag, int fRun) {return getObject_ (fObject, fTag, fRun);}
bool HcalDbPool::putObject (HcalQIEData* fObject, const std::string& fTag, int fRun) {return putObject_ (fObject, "HcalQIEData", fTag, fRun);}
bool HcalDbPool::getObject (HcalChannelQuality* fObject, const std::string& fTag, int fRun) {return getObject_ (fObject, fTag, fRun);}
bool HcalDbPool::putObject (HcalChannelQuality* fObject, const std::string& fTag, int fRun) {return putObject_ (fObject, "HcalChannelQuality", fTag, fRun);}
bool HcalDbPool::getObject (HcalElectronicsMap* fObject, const std::string& fTag, int fRun) {return getObject_ (fObject, fTag, fRun);}
bool HcalDbPool::putObject (HcalElectronicsMap* fObject, const std::string& fTag, int fRun) {return putObject_ (fObject, "HcalElectronicsMap", fTag, fRun);}
