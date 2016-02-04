//
// F.Ratnikov (UMd), Jan. 6, 2006
//
#ifndef HcalDbPoolOCCI_h
#define HcalDbPoolOCCI_h

#include <memory>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

/**

   \class HcalDbPoolOCCI
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

class HcalDbPoolOCCI {
 public:
  HcalDbPoolOCCI (const std::string& fDb);
  ~HcalDbPoolOCCI ();

  bool getObject (HcalPedestals* fObject, const std::string& fTag, unsigned long fRun);
  bool getObject (HcalGains* fObject, const std::string& fTag, unsigned long fRun);
  bool getObject (HcalElectronicsMap* fObject, const std::string& fTag, unsigned long fRun);
 private:
  oracle::occi::Environment* mEnvironment;
  oracle::occi::Connection* mConnect;
  oracle::occi::Statement* mStatement;
  std::string getMetadataToken (const std::string& fTag);
  std::string getDataToken (const std::string& fIov, unsigned long fRun);
  template <class T, class S> bool getObjectGeneric (T* fObject, S* fCondObject, const std::string& fTag, unsigned long fRun);

};
#endif
