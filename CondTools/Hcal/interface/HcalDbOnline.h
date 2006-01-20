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
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"

/**

   \class HcalDbOnline
   \brief Gather conditions data from online DB
   \author Fedor Ratnikov
   
*/

namespace oracle {
  namespace occi {
    class Environment;
    class Connection;
  }
}

class HcalDbOnline {
 public:
  HcalDbOnline (const std::string& fDb);
  ~HcalDbOnline ();

  std::auto_ptr <HcalPedestals> getPedestals (const std::string& fTag);
  std::auto_ptr <HcalGains> getGains (const std::string& fTag);
 private:
  oracle::occi::Environment* mEnvironment;
  oracle::occi::Connection* mConnect;
};
#endif
