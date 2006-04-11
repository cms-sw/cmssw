#ifndef OnlineDB_SiStripESSources_SiStripFedCablingBuilderFromDb_H
#define OnlineDB_SiStripESSources_SiStripFedCablingBuilderFromDb_H

#include "CalibTracker/SiStripConnectivity/interface/SiStripFedCablingESSource.h"
#include "boost/cstdint.hpp"
#include <string>

class SiStripConfigDb;

using namespace std;

class SiStripFedCablingBuilderFromDb : public SiStripFedCablingESSource {
  
 public:
  
  SiStripFedCablingBuilderFromDb( const edm::ParameterSet& );
  ~SiStripFedCablingBuilderFromDb(); 
  
 private:
  
  /** Builds FED cabling using information from TK config DB. */
  virtual SiStripFedCabling* makeFedCabling();

  /** Interface to TK online configuration database. */
  SiStripConfigDb* db_;
  
};

#endif // OnlineDB_SiStripESSources_SiStripFedCablingBuilderFromDb_H
