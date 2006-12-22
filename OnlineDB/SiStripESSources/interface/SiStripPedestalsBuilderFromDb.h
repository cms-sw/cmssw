// Last commit: $Id: $
// Latest tag:  $Name: $
// Location:    $Source: $

#ifndef OnlineDB_SiStripESSources_SiStripPedestalsBuilderFromDb_H
#define OnlineDB_SiStripESSources_SiStripPedestalsBuilderFromDb_H

#include "CalibTracker/SiStripPedestals/interface/SiStripPedestalsESSource.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "boost/cstdint.hpp"
#include <vector>
#include <string>

class SiStripFecCabling;
class SiStripDetCabling;
class SiStripPedestals;
class DcuDetIdMap;

class SiStripPedestalsBuilderFromDb : public SiStripPedestalsESSource {
  
 public:

  SiStripPedestalsBuilderFromDb( const edm::ParameterSet& );
  virtual ~SiStripPedestalsBuilderFromDb();
  
  /** Builds pedestals using info from configuration database. */
  virtual SiStripPedestals* makePedestals();
  
  /** Builds pedestals using FED descriptions and cabling info
      retrieved from configuration database. */
  static void buildPedestals( SiStripConfigDb* const,
			      const SiStripDetCabling&,
			      SiStripPedestals& );
  
 protected:
  
  /** Virtual method that is called by makePedestals() to allow
      pedestals to be written to the conditions database. */
  virtual void writePedestalsToCondDb( const SiStripPedestals& ) {;}
  
  /** Access to the configuration DB interface class. */
  SiStripConfigDb* db_;
  
  /** Container for DB connection parameters. */
  SiStripConfigDb::DbParams dbParams_;
  
};

#endif // OnlineDB_SiStripESSources_SiStripPedestalsBuilderFromDb_H

