//
// F.Ratnikov (UMd), Jan. 6, 2006
//
#ifndef HcalDbOnline_h
#define HcalDbOnline_h

#include <memory>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalCalibrationQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

/**

   \class HcalDbOnline
   \brief Gather conditions data from online DB
   \author Fedor Ratnikov
   
*/

namespace oracle {
  namespace occi {
    class Environment;
    class Connection;
    class Statement;
  }
}

class HcalDbOnline {
 public:
  typedef unsigned long long IOVTime;
  typedef std::pair <IOVTime, IOVTime> IntervalOV;

  HcalDbOnline (const std::string& fDb, bool fVerbose = false);
  ~HcalDbOnline ();

  bool getObject (HcalPedestals* fObject, HcalPedestalWidths* fWidths, const std::string& fTag, IOVTime fTime);
  bool getObject (HcalPedestals* fObject, const std::string& fTag, IOVTime fTime);
  bool getObject (HcalGains* fObject, HcalGainWidths* fWidths, const std::string& fTag, IOVTime fTime);
  bool getObject (HcalGains* fObject, const std::string& fTag, IOVTime fTime);
  bool getObject (HcalPedestalWidths* fObject, const std::string& fTag, IOVTime fTime);
  bool getObject (HcalGainWidths* fObject, const std::string& fTag, IOVTime fTime);
  bool getObject (HcalElectronicsMap* fObject, const std::string& fTag, IOVTime fTime);
  bool getObject (HcalQIEData* fObject, const std::string& fTag, IOVTime fTime);
  bool getObject (HcalCalibrationQIEData* fObject, const std::string& fTag, IOVTime fTime);

  std::vector<std::string> metadataAllTags ();
  std::vector<IntervalOV> getIOVs (const std::string& fTag);
  
  
 private:
  oracle::occi::Environment* mEnvironment;
  oracle::occi::Connection* mConnect;
  oracle::occi::Statement* mStatement;
  template <class T> bool getObjectGeneric (T* fObject, const std::string& fTag);
  bool mVerbose;
};
#endif
