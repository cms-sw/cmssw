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
  HcalDbOnline (const std::string& fDb, bool fVerbose = false);
  ~HcalDbOnline ();

  bool getObject (HcalPedestals* fObject, HcalPedestalWidths* fWidths, const std::string& fTag);
  bool getObject (HcalPedestals* fObject, const std::string& fTag);
  bool getObject (HcalGains* fObject, const std::string& fTag);
  bool getObject (HcalPedestalWidths* fObject, const std::string& fTag);
  bool getObject (HcalGainWidths* fObject, const std::string& fTag) {return false;}
  bool getObject (HcalElectronicsMap* fObject, const std::string& fTag);
  bool getObject (HcalQIEData* fObject, const std::string& fTag);
  bool getObject (HcalCalibrationQIEData* fObject, const std::string& fTag);
 private:
  oracle::occi::Environment* mEnvironment;
  oracle::occi::Connection* mConnect;
  oracle::occi::Statement* mStatement;
  template <class T> bool getObjectGeneric (T* fObject, const std::string& fTag);
  bool mVerbose;
};
#endif
