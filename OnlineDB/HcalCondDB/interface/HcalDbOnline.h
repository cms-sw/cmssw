//
// F.Ratnikov (UMd), Jan. 6, 2006
//
#ifndef HcalDbOnline_h
#define HcalDbOnline_h

#include <memory>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

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

  std::vector<std::string> metadataAllTags ();
  std::vector<IntervalOV> getIOVs (const std::string& fTag);
  
  
 private:
  oracle::occi::Environment* mEnvironment;
  oracle::occi::Connection* mConnect;
  oracle::occi::Statement* mStatement;
  bool mVerbose;
};
#endif
