#ifndef OnlineDB_SiStripESSources_SiStripFedCablingBuilderFromDb_H
#define OnlineDB_SiStripESSources_SiStripFedCablingBuilderFromDb_H

#include "CalibTracker/SiStripConnectivity/interface/SiStripFedCablingESSource.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "boost/cstdint.hpp"
#include <string>
#include <vector>

class SiStripConfigDb;

using namespace std;

class SiStripFedCablingBuilderFromDb : public SiStripFedCablingESSource {
  
 public:

  typedef vector<FedChannelConnectionDescription*> FedConnections;
  
  SiStripFedCablingBuilderFromDb( const edm::ParameterSet& );
  ~SiStripFedCablingBuilderFromDb(); 

 protected:

  /** Builds FED cabling using information from TK config DB. */
  virtual SiStripFedCabling* makeFedCabling();

  /** */
  void createFecCablingFromConnections( const FedConnections&, SiStripFecCabling& );
  /** */
  void createFecCablingFromDevices( SiStripConfigDb* const, SiStripFecCabling& );
  
  /** Interface to TK online configuration database. */
  SiStripConfigDb* db_;
  
};

#endif // OnlineDB_SiStripESSources_SiStripFedCablingBuilderFromDb_H
