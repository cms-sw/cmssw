//
// F.Ratnikov (UMd), Jul. 19, 2005
//
#ifndef HcalDbTool_h
#define HcalDbTool_h

#include <memory>
#include <string>

#include "CondFormats/HcalObjects/interface/AllClasses.h"
#include "DataSvc/Ref.h"

/**
   \class HcalDbTool
   \brief IO for POOL instances of Hcal Calibrations
   \author Fedor Ratnikov Oct. 28, 2005
   $Id: HcalDbTool.h,v 1.3 2006/10/06 21:38:38 fedor Exp $
*/

namespace cond {
  class IOV;
  class DBSession;
  class MetaData;
  class ServiceLoader;
}

   
class HcalDbTool {
 public:
  typedef unsigned long long IOVRun;
  HcalDbTool (const std::string& fConnect, bool fVerbose = false, bool fXmlAuth = false, const char* fCatalog = 0);
  ~HcalDbTool ();

  const std::string& metadataGetToken (const std::string& fTag);
  bool metadataSetTag (const std::string& fTag, const std::string& fToken);
  std::vector<std::string> metadataAllTags ();

  bool getObject (HcalPedestals* fObject, const std::string& fTag, IOVRun fRun);
  bool putObject (HcalPedestals* fObject, const std::string& fTag, IOVRun fRun, bool fAppend = false);
  bool deleteObject (HcalPedestals* fObject, const std::string& fTag, IOVRun fRun);
  bool getObject (HcalPedestalWidths* fObject, const std::string& fTag, IOVRun fRun);
  bool putObject (HcalPedestalWidths* fObject, const std::string& fTag, IOVRun fRun, bool fAppend = false);
  bool getObject (HcalGains* fObject, const std::string& fTag, IOVRun fRun);
  bool putObject (HcalGains* fObject, const std::string& fTag, IOVRun fRun, bool fAppend = false);
  bool getObject (HcalGainWidths* fObject, const std::string& fTag, IOVRun fRun);
  bool putObject (HcalGainWidths* fObject, const std::string& fTag, IOVRun fRun, bool fAppend = false);
  bool getObject (HcalQIEData* fObject, const std::string& fTag, IOVRun fRun);
  bool putObject (HcalQIEData* fObject, const std::string& fTag, IOVRun fRun, bool fAppend = false);
  bool getObject (HcalCalibrationQIEData* fObject, const std::string& fTag, IOVRun fRun);
  bool putObject (HcalCalibrationQIEData* fObject, const std::string& fTag, IOVRun fRun, bool fAppend = false);
  bool getObject (HcalChannelQuality* fObject, const std::string& fTag, IOVRun fRun);
  bool putObject (HcalChannelQuality* fObject, const std::string& fTag, IOVRun fRun, bool fAppend = false);
  bool getObject (HcalElectronicsMap* fObject, const std::string& fTag, IOVRun fRun);
  bool putObject (HcalElectronicsMap* fObject, const std::string& fTag, IOVRun fRun, bool fAppend = false);
  bool getObject (cond::IOV* fObject, const std::string& fTag);
  bool putObject (cond::IOV* fObject, const std::string& fTag);
 private:
  std::string mConnect;
  cond::DBSession* mSession;
  cond::MetaData* mMetadata;
  cond::ServiceLoader* mLoader;
  std::string mTag;
  std::string mToken;
  bool mVerbose;
  template <class T>
  bool storeObject (T* fObject, const std::string& fContainer, pool::Ref<T>* fObject);

  template <class T>
  bool updateObject (T* fObject, pool::Ref<T>* fUpdate, const std::string& fContainer = "");

  template <class T>
  bool updateObject (pool::Ref<T>* fUpdate);

  template <class T>
  bool deleteObject (pool::Ref<T>* fObject, const std::string& fContainer = "");

  template <class T>
  bool storeIOV (const pool::Ref<T>& fObject, IOVRun fMaxRun, pool::Ref<cond::IOV>* fIov, bool Append);

  template <class T>
  bool getObject (const pool::Ref<cond::IOV>& fIOV, IOVRun fRun, pool::Ref<T>* fObject);

  template <class T> 
  bool getObject (const std::string& fToken, pool::Ref<T>* fObject);

  template <class T>
  bool getObject_ (T* fObject, const std::string& fTag, IOVRun fRun);

  template <class T>
  bool putObject_ (T* fObject, const std::string& fClassName, const std::string& fTag, IOVRun fRun, bool Append);

  template <class T>
  bool deleteObject_ (T* fObject, const std::string& fTag, IOVRun fRun);

  bool cleanAllIov (const std::string& fToken);
};
#endif
