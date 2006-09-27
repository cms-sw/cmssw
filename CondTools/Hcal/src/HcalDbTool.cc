
/**
   \class HcalDbTool
   \brief IO for POOL instances of Hcal Calibrations
   \author Fedor Ratnikov Oct. 28, 2005
   $Id: HcalDbTool.cc,v 1.1 2006/09/26 20:49:01 fedor Exp $
*/

#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

// conditions
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/ServiceLoader.h"
#include "CondCore/IOVService/interface/IOV.h"
#include "DataSvc/Ref.h"
#include "CondCore/DBCommon/interface/DBWriter.h"


#include "CondFormats/HcalObjects/interface/AllObjects.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "CondTools/Hcal/interface/HcalDbTool.h"

namespace {
  pool::Ref<cond::IOV> iovCache;
}

typedef std::map<HcalDbTool::IOVRun,std::string> IOVCollection;

template <class T>
bool HcalDbTool::storeObject (T* fObject, const std::string& fContainer, pool::Ref<T>* fRef) {
  if (mVerbose) std::cout << "HcalDbTool::storeObject-> start..." << std::endl;
  if (!fRef->isNull ()) {
    std::cerr << "storeObject-> Ref is not empty. Ignore." << std::endl;
    return false;
  }
  try {
    mSession->startUpdateTransaction();
    if (mVerbose) std::cout << "transaction ---> start" << std::endl;
    cond::DBWriter writer (*(mSession), fContainer);
    std::string token = writer.markWrite (fObject);
    *fRef = pool::Ref <T> (&(mSession->DataSvc()), token);
    if (mVerbose) std::cout << "commit write/read back operation" << std::endl;
    mSession->commit ();
  }
  catch (cond::Exception& e) {
    std::cerr << "storeObject->  COND error: "  << e.what() << std::endl;
    return false;
  }
  catch (pool::Exception& e) {
    std::cerr << "storeObject->  POOL error: "  << e.what() << std::endl;
    return false;
  }
  catch (cms::Exception& e) {
    std::cerr << "storeObject->  CMS error: "  << e.what() << std::endl;
    return false;
  }
  catch (const std::exception& e) {
    std::cerr << "storeObject->  standard error: "  << e.what() << std::endl;
    return false;
  }
   catch (...) {
     std::cerr << "storeObject->  not standard error "  << std::endl;
     return false;
   }
  if (mVerbose) std::cout << "HcalDbTool::storeObject-> end..." << std::endl;
  return true;
} 

template <class T>
bool HcalDbTool::updateObject (T* fObject, pool::Ref<T>* fUpdate, const std::string& fContainer) {
  typedef T D;
  if (mVerbose) std::cout << "HcalDbTool::updateObject-> start..." << std::endl;
  try {
    mSession->startUpdateTransaction();
    if (mVerbose) std::cout << "transaction ---> start" << std::endl;
    cond::DBWriter writer (*(mSession), fContainer);
    if (fObject) *(fUpdate->ptr ()) = *fObject; // update object
    std::string token = fUpdate->toString();
    writer.template markUpdate<T>(token);
    if (mVerbose) std::cout << "transaction ---> commit" << std::endl;
    mSession->commit ();
  }
  catch (cond::Exception& e) {
    std::cerr << "updateObject->  COND error: "  << e.what() << std::endl;
    return false;
  }
  catch (pool::Exception& e) {
    std::cerr << "updateObject->  POOL error: "  << e.what() << std::endl;
    return false;
  }
  catch (cms::Exception& e) {
    std::cerr << "updateObject->  CMS error: "  << e.what() << std::endl;
    return false;
  }
  catch (std::exception& e) {
    std::cerr << "updateObject->  error: " << e.what () << std::endl;
    return false;
  }
  catch (...) {
    std::cerr << "HcalDbTool::updateObject-> General error" << std::endl;
  }
  if (mVerbose) std::cout << "HcalDbTool::updateObject-> end..." << std::endl;
  return true;
}

template <class T>
void markDelete (cond::DBWriter& writer, const std::string& token) {
  writer.template markDelete<T>  (token);
}

template <class T>
bool HcalDbTool::deleteObject (pool::Ref<T>* fRef, const std::string& fContainer) {
  if (mVerbose) std::cout << "HcalDbTool::deleteObject-> start..." << std::endl;
  try {
    if (mVerbose) std::cout << "transaction ---> start" << std::endl;
    mSession->startUpdateTransaction();
    cond::DBWriter writer (*(mSession), fContainer);
    //    writer.markDelete<T> (fRef->toString());
    markDelete<T> (writer, fRef->toString());
    mSession->commit ();
    if (mVerbose) std::cout << "transaction ---> commit" << std::endl;
  }
  catch (const cond::Exception& e) {
    std::cerr<<"HcalDbTool::deleteObject-> COND error: " << e.what() << std::endl;
    return false;
  }
  catch (const pool::Exception& e) {
    std::cerr<<"HcalDbTool::deleteObject-> POOL error: " << e.what() << std::endl;
    return false;
  }
  catch (const cms::Exception& e) {
    std::cerr<<"HcalDbTool::deleteObject-> CMS error: " << e.what() << std::endl;
    return false;
  }
  catch (std::exception& e) {
    std::cerr << "HcalDbTool::deleteObject->  error: " << e.what () << std::endl;
    return false;
  }
  catch (...) {
    std::cerr << "HcalDbTool::deleteObject-> General error" << std::endl;
  }
  if (mVerbose) std::cout << "HcalDbTool::deleteObject-> end..." << std::endl;
  return true;
}

template <class T>
bool HcalDbTool::updateObject (pool::Ref<T>* fUpdate) {
  return updateObject ((T*)0, fUpdate);
}

template <class T>
bool HcalDbTool::storeIOV (const pool::Ref<T>& fObject, unsigned fMaxRun, pool::Ref<cond::IOV>* fIov) {
  unsigned maxRun = fMaxRun == 0 ? 0xffffffff : fMaxRun;
  if (fIov->isNull ()) {
    cond::IOV* newIov = new cond::IOV ();
    newIov->iov.insert (std::make_pair (maxRun, fObject.toString ()));
    return storeObject (newIov, "cond::IOV", fIov);
  }
  else {
    (*fIov)->iov.insert (std::make_pair (maxRun, fObject.toString ()));
    return updateObject (fIov);
  }
}

template <class T>
bool HcalDbTool::getObject (const pool::Ref<cond::IOV>& fIOV, unsigned fRun, pool::Ref<T>* fObject) {
  if (!fIOV.isNull ()) {
    // scan IOV, search for valid data
    for (IOVCollection::iterator iovi = fIOV->iov.begin (); iovi != fIOV->iov.end (); iovi++) {
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
bool HcalDbTool::getObject (const std::string& fToken, pool::Ref<T>* fObject) {
  if (mVerbose) std::cout << "HcalDbTool::getObject-> start for token: " << fToken << std::endl;
  try {
    *fObject = pool::Ref <T> (&(mSession->DataSvc()), fToken);
    if (mVerbose) std::cout << "transaction ---> start" << std::endl;
    mSession->startReadOnlyTransaction();
    fObject->isNull ();
    mSession->commit ();
    if (mVerbose) std::cout << "transaction ---> commit" << std::endl;
  }
  catch (const cond::Exception& e) {
    std::cerr<<"getObject-> COND error: " << e.what() << std::endl;
  }
  catch (const pool::Exception& e) {
    std::cerr<<"getObject-> POOL error: " << e.what() << std::endl;
  }
  catch (const cms::Exception& e) {
    std::cerr<<"getObject-> CMS error: " << e.what() << std::endl;
  }
  catch(...){
    std::cerr << "getObject-> Funny error" << std::endl;
  }
  if (mVerbose) std::cout << "HcalDbTool::getObject-> end..." << std::endl;
  return !(fObject->isNull ());
}

template <class T>
bool HcalDbTool::getObject_ (T* fObject, const std::string& fTag, IOVRun fRun) {
  std::string metadataToken = metadataGetToken (fTag);
  if (metadataToken.empty ()) {
    std::cerr << "HcalDbTool::getObject ERROR-> Can not find metadata for tag " << fTag << std::endl;
    return false;
  }
  if (iovCache.toString () != metadataToken) {
    getObject (metadataToken, &iovCache);
  }
  if (iovCache.isNull ()) {
    std::cerr << "HcalDbTool::getObject ERROR: can not find IOV for token " << metadataToken << std::endl;;
    return false;
  }
  pool::Ref<T> ref;
  if (getObject (iovCache, fRun, &ref)) {
    *fObject = *ref; // make copy
    return true;
  }
  return false;
}

template <class T>
bool HcalDbTool::putObject_ (T* fObject, const std::string& fClassName, const std::string& fTag, IOVRun fRun) {
  std::string metadataToken = metadataGetToken (fTag);
  pool::Ref<cond::IOV> iov;
  if (!metadataToken.empty ()) {
    getObject (metadataToken, &iov);
    if (iov.isNull ()) {
      std::cerr << "HcalDbTool::putObject ERROR: can not find IOV for token " << metadataToken << std::endl;;
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

template <class T>
bool HcalDbTool::deleteObject_ (T* fObject, const std::string& fTag, IOVRun fRun) {
  std::string metadataToken = metadataGetToken (fTag);
  pool::Ref<cond::IOV> iov;
  if (!metadataToken.empty () && iovCache.toString () != metadataToken) {
    getObject (metadataToken, &iovCache);
    if (iovCache.isNull ()) {
      std::cerr << "HcalDbTool::deleteObject ERROR: can not find IOV for token " << metadataToken << std::endl;;
      return false;
    }
  }
  pool::Ref<T> ref;
  if (getObject (iovCache, fRun, &ref)) {
    metadataToken = ref.toString ();
    if (deleteObject (&ref) && cleanAllIov (metadataToken)) {
      return true;
    }
    else {
      std::cerr << "HcalDbTool::deleteObject ERROR: can not clearly delete object for token " << metadataToken << std::endl;;
    }
  }
  return false;
}

bool HcalDbTool::cleanAllIov (const std::string& fToken) {
  std::vector<std::string> tags = metadataAllTags ();
  for (unsigned i = 0; i < tags.size (); i++) {
    if (mVerbose) std::cout << "HcalDbTool::cleanAllIov-> processing tag " << tags [i] << std::endl;
    std::string token = metadataGetToken (tags [i]);
    pool::Ref<cond::IOV> iov;
    if (getObject (token, &iov)) {
      bool tbUpdated = false;
      // search for object's token
      IOVCollection::iterator iovi = iov->iov.begin ();
      while (iovi != iov->iov.end ()) {
	IOVCollection::iterator tmpiov = iovi;
	iovi++;
	if (tmpiov->second == fToken) {
	  std::cerr << "HcalDbTool::cleanAllIov-> CONSISTENCY WARNING: removing run " << tmpiov->first
		    << ", tag " << tags [i] << " reference for object " << fToken 
		    << " may make data internally inconsistent." << std::endl;
	  iov->iov.erase (tmpiov);
	  tbUpdated = true;
	}
      }
      if (tbUpdated) {
	if (mVerbose) std::cout << "HcalDbTool::cleanAllIov-> updating IOV for tag" << tags [i] << std::endl;
	if (!updateObject (&iov)) {
	  std::cerr << "HcalDbTool::cleanAllIov-> Can not update IOV for tag " << tags [i] << std::endl;
	}
      }
    }
    else {
      std::cerr << "HcalDbTool::cleanAllIov-> Can not get IOV for token " << token << std::endl;
    }
  }
  return true;
}

HcalDbTool::HcalDbTool (const std::string& fConnect, bool fVerbose)
  : mConnect (fConnect),
    mVerbose (fVerbose) {
  try {
    // services
    mLoader=new cond::ServiceLoader;
    mLoader->loadAuthenticationService(cond::Env);
    mLoader->loadMessageService(mVerbose ? cond::Debug : cond::Error);
    // make session
    mSession = new cond::DBSession (mConnect);
    const char* catalog = ::getenv ("POOL_CATALOG");
    if (!catalog) {
      if (mVerbose) std::cout << "HcalDbTool::HcalDbTool-> using default catalog" << std::endl;
      catalog = "file:PoolFileCatalog.xml";
    }
    mSession->setCatalog (catalog);
    mSession->connect (cond::ReadWriteCreate);
    if (mVerbose) std::cout << "HcalDbTool::HcalDbTool-> using catalog: " << catalog << std::endl;
    // make metadata
    mMetadata = new cond::MetaData (mConnect, *mLoader);
  }
  catch (cond::Exception& e) {
    std::cerr << "HcalDbTool::HcalDbTool->  COND error: "  << e.what() << std::endl;
  }
  catch (pool::Exception& e) {
    std::cerr << "HcalDbTool::HcalDbTool->  POOL error: "  << e.what() << std::endl;
  }
  catch (cms::Exception& e) {
    std::cerr << "HcalDbTool::HcalDbTool->  CMS error: "  << e.what() << std::endl;
  }
  catch (const std::exception& e) {
    std::cerr << "HcalDbTool::HcalDbTool->  standard error: "  << e.what() << std::endl;
  }
   catch (...) {
     std::cerr << "HcalDbTool::HcalDbTool->  not standard error "  << std::endl;
   }
}

HcalDbTool::~HcalDbTool () {
  if (mVerbose) std::cout << "HcalDbTool::~HcalDbTool started..." << std::endl;
  delete mSession;
  delete mLoader;
  delete mMetadata;
  if (mVerbose) std::cout << "HcalDbTool::~HcalDbTool done..." << std::endl;
}


std::vector<std::string> HcalDbTool::metadataAllTags () {
  std::vector<std::string> result;
  try {    
    mMetadata = new cond::MetaData (mConnect, *mLoader);
    mMetadata->connect (cond::ReadOnly);
    mMetadata->listAllTags (result);
    mMetadata->disconnect ();
    delete mMetadata;
    mMetadata = 0;
  }
  catch (const std::exception& e) {
    std::cerr << "metadataAllTags->  standard error: "  << e.what() << std::endl;
  }
  catch (...) {
    std::cerr << "metadataAllTags->  not standard error "  << std::endl;
  }
  return result;
}

const std::string& HcalDbTool::metadataGetToken (const std::string& fTag) {
    if (mTag == fTag) return mToken;
    mTag = fTag;
    try {
      mMetadata = new cond::MetaData (mConnect, *mLoader);
      mMetadata->connect (cond::ReadOnly);
      mToken = mMetadata->getToken (fTag);
      mMetadata->disconnect ();
      delete mMetadata;
      mMetadata = 0;
    }
    catch (const std::exception& e) {
      std::cerr << "metadataGetTag->  standard error: "  << e.what() << std::endl;
      mToken.clear ();
    }
    catch (...) {
      std::cerr << "metadataGetTag->  not standard error "  << std::endl;
      mToken.clear ();
    }
    return mToken;
}

bool HcalDbTool::metadataSetTag (const std::string& fTag, const std::string& fToken) {
  if (mVerbose) std::cout << "HcalDbTool::metadataSetTag->begin..." << std::endl;
  bool result = false;
  try {
    mMetadata = new cond::MetaData (mConnect, *mLoader);
    mMetadata->connect (cond::ReadWriteCreate);
    result = mMetadata->addMapping (fTag, fToken);;
    mMetadata->disconnect ();
    delete mMetadata;
    mMetadata = 0;
  }
  catch (const std::exception& e) {
    std::cerr << "metadataSetTag->  standard error: "  << e.what() << std::endl;
    mToken.clear ();
  }
  catch (...) {
    std::cerr << "metadataSetTag->  not standard error "  << std::endl;
    mToken.clear ();
  }
  return result;
}


bool HcalDbTool::getObject (cond::IOV* fObject, const std::string& fTag) {
  if (!fObject) return false;
  std::string metadataToken = metadataGetToken (fTag);
  if (metadataToken.empty ()) {
    std::cerr << "HcalDbTool::getObject IOV ERROR-> Can not find metadata for tag " << fTag << std::endl;
    return false;
  }
  if (iovCache.toString () != metadataToken) {
    getObject (metadataToken, &iovCache);
  }
  if (iovCache.isNull ()) {
    std::cerr << "HcalDbTool::getObject ERROR: can not find IOV for token " << metadataToken << std::endl;;
    return false;
  }
  *fObject = *iovCache;
  return true;
}

bool HcalDbTool::putObject (cond::IOV* fObject, const std::string& fTag) {
  std::string metadataToken = metadataGetToken (fTag);
  if (metadataToken.empty ()) {
    pool::Ref<cond::IOV> iov;
    if (storeObject (fObject, "cond::IOV", &iov)) {
      metadataToken = iov.toString ();
      return metadataSetTag (fTag, metadataToken);
    }
    else {
      return false;
    }
  }
  else if (iovCache.toString () != metadataToken) {
    getObject (metadataToken, &iovCache);
  }
  if (iovCache.isNull ()) {
    std::cerr << "HcalDbTool::putObject ERROR: can not find IOV for token " << metadataToken << std::endl;;
    return false;
  }
  //  return updateObject (fObject, &iovCache);
  return false;
}

bool HcalDbTool::getObject (HcalPedestals* fObject, const std::string& fTag, IOVRun fRun) {return getObject_ (fObject, fTag, fRun);}
bool HcalDbTool::putObject (HcalPedestals* fObject, const std::string& fTag, IOVRun fRun) {return putObject_ (fObject, "HcalPedestals", fTag, fRun);}
bool HcalDbTool::deleteObject (HcalPedestals* fObject, const std::string& fTag, IOVRun fRun) {return deleteObject_ (fObject, fTag, fRun);}
bool HcalDbTool::getObject (HcalPedestalWidths* fObject, const std::string& fTag, IOVRun fRun) {return getObject_ (fObject, fTag, fRun);}
bool HcalDbTool::putObject (HcalPedestalWidths* fObject, const std::string& fTag, IOVRun fRun) {return putObject_ (fObject, "HcalPedestalWidths", fTag, fRun);}
bool HcalDbTool::getObject (HcalGains* fObject, const std::string& fTag, IOVRun fRun) {return getObject_ (fObject, fTag, fRun);}
bool HcalDbTool::putObject (HcalGains* fObject, const std::string& fTag, IOVRun fRun) {return putObject_ (fObject, "HcalGains", fTag, fRun);}
bool HcalDbTool::getObject (HcalGainWidths* fObject, const std::string& fTag, IOVRun fRun) {return getObject_ (fObject, fTag, fRun);}
bool HcalDbTool::putObject (HcalGainWidths* fObject, const std::string& fTag, IOVRun fRun) {return putObject_ (fObject, "HcalGainWidths", fTag, fRun);}
bool HcalDbTool::getObject (HcalQIEData* fObject, const std::string& fTag, IOVRun fRun) {return getObject_ (fObject, fTag, fRun);}
bool HcalDbTool::putObject (HcalQIEData* fObject, const std::string& fTag, IOVRun fRun) {return putObject_ (fObject, "HcalQIEData", fTag, fRun);}
bool HcalDbTool::getObject (HcalCalibrationQIEData* fObject, const std::string& fTag, IOVRun fRun) {return getObject_ (fObject, fTag, fRun);}
bool HcalDbTool::putObject (HcalCalibrationQIEData* fObject, const std::string& fTag, IOVRun fRun) {return putObject_ (fObject, "HcalQIEData", fTag, fRun);}
bool HcalDbTool::getObject (HcalChannelQuality* fObject, const std::string& fTag, IOVRun fRun) {return getObject_ (fObject, fTag, fRun);}
bool HcalDbTool::putObject (HcalChannelQuality* fObject, const std::string& fTag, IOVRun fRun) {return putObject_ (fObject, "HcalChannelQuality", fTag, fRun);}
bool HcalDbTool::getObject (HcalElectronicsMap* fObject, const std::string& fTag, IOVRun fRun) {return getObject_ (fObject, fTag, fRun);}
bool HcalDbTool::putObject (HcalElectronicsMap* fObject, const std::string& fTag, IOVRun fRun) {return putObject_ (fObject, "HcalElectronicsMap", fTag, fRun);}
