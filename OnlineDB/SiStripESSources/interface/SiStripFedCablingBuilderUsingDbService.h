// Last commit: $Id:$
// Latest tag:  $Name:$
// Location:    $Source:$

#ifndef OnlineDB_SiStripESSources_SiStripFedCablingBuilderUsingDbService_H
#define OnlineDB_SiStripESSources_SiStripFedCablingBuilderUsingDbService_H

#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "boost/cstdint.hpp"
#include <vector>
#include <string>

class SiStripFedCablingBuilderUsingDbService : public SiStripFedCablingBuilderFromDb {
  
 public:

  SiStripFedCablingBuilderUsingDbService( const edm::ParameterSet& );
  virtual ~SiStripFedCablingBuilderUsingDbService(); 
  
  /** Builds FED cabling using information from TK config DB. */
  virtual SiStripFedCabling* makeFedCabling();

};

#endif // OnlineDB_SiStripESSources_SiStripFedCablingBuilderUsingDbService_H

