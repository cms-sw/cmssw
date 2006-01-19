//
// F.Ratnikov (UMd), Jul. 19, 2005
//
#ifndef HcalDbPool_h
#define HcalDbPool_h

#include <memory>
#include <string>

#include "CondFormats/HcalObjects/interface/AllClasses.h"

/**
   \class HcalDbPOOL
   \brief IO for POOL instances of Hcal Calibrations
   \author Fedor Ratnikov Oct. 28, 2005
   $Id: HcalDbPool.h,v 1.3 2005/12/15 23:37:56 fedor Exp $
*/

namespace cond {
  class MetaData;
  class IOV;
}
namespace pool {
  class IFileCatalog;
  class IDataSvc;
  class Placement;
  template <class T> class Ref;
}
   
class HcalDbPool {
 public:
  HcalDbPool (const std::string& fConnect);
  ~HcalDbPool ();

  pool::IDataSvc* service ();
  std::string metadataGetToken (const std::string& fTag);
  bool metadataSetTag (const std::string& fTag, const std::string& fToken);

  bool getObject (HcalPedestals* fObject, const std::string& fTag, int fRun);
  bool putObject (HcalPedestals* fObject, const std::string& fTag, int fRun);
  bool getObject (HcalPedestalWidths* fObject, const std::string& fTag, int fRun);
  bool putObject (HcalPedestalWidths* fObject, const std::string& fTag, int fRun);
  bool getObject (HcalGains* fObject, const std::string& fTag, int fRun);
  bool putObject (HcalGains* fObject, const std::string& fTag, int fRun);
  bool getObject (HcalGainWidths* fObject, const std::string& fTag, int fRun);
  bool putObject (HcalGainWidths* fObject, const std::string& fTag, int fRun);
  bool getObject (HcalQIEData* fObject, const std::string& fTag, int fRun);
  bool putObject (HcalQIEData* fObject, const std::string& fTag, int fRun);
  bool getObject (HcalChannelQuality* fObject, const std::string& fTag, int fRun);
  bool putObject (HcalChannelQuality* fObject, const std::string& fTag, int fRun);
  bool getObject (HcalElectronicsMap* fObject, const std::string& fTag, int fRun);
  bool putObject (HcalElectronicsMap* fObject, const std::string& fTag, int fRun);
 private:
  std::string mConnect;
  std::auto_ptr<cond::MetaData> mMetaData;
  std::auto_ptr<pool::IFileCatalog> mCatalog;
  std::auto_ptr<pool::IDataSvc> mService;
  std::auto_ptr<pool::Placement> mPlacement;

  template <class T>
  bool storeObject (T* fObject, const std::string& fContainer, pool::Ref<T>* fObject);

  template <class T>
  bool updateObject (T* fObject, pool::Ref<T>* fUpdate);

  template <class T>
  bool updateObject (pool::Ref<T>* fUpdate);

  template <class T>
  bool storeIOV (const pool::Ref<T>& fObject, unsigned fMaxRun, pool::Ref<cond::IOV>* fIov);

  template <class T>
  bool getObject (const pool::Ref<cond::IOV>& fIOV, unsigned fRun, pool::Ref<T>* fObject);

  template <class T> 
  bool getObject (const std::string& fToken, pool::Ref<T>* fObject);

  template <class T>
  bool getObject_ (T* fObject, const std::string& fTag, int fRun);

  template <class T>
  bool putObject_ (T* fObject, const std::string& fClassName, const std::string& fTag, int fRun);

};
#endif
