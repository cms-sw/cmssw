// Last commit: $Id: $
// Latest tag:  $Name: $
// Location:    $Source: $

#ifndef OnlineDB_SiStripESSources_SiStripNoiseBuilderFromDb_H
#define OnlineDB_SiStripESSources_SiStripNoiseBuilderFromDb_H

#include "CalibTracker/SiStripPedestals/interface/SiStripNoiseESSource.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "boost/cstdint.hpp"
#include <vector>
#include <string>

class SiStripFecCabling;
class SiStripDetCabling;
class SiStripNoises;
class DcuDetIdMap;

class SiStripNoiseBuilderFromDb : public SiStripNoiseESSource {
  
 public:

  SiStripNoiseBuilderFromDb( const edm::ParameterSet& );
  virtual ~SiStripNoiseBuilderFromDb();
  
  /** Builds pedestals using info from configuration database. */
  virtual SiStripNoises* makeNoise();
  
  /** Builds pedestals using FED descriptions and cabling info
      retrieved from configuration database. */
  static void buildNoise( SiStripConfigDb* const,
			  const SiStripDetCabling&,
			  SiStripNoises& );
  
 protected:
  
  /** Virtual method that is called by makeNoise() to allow
      pedestals to be written to the conditions database. */
  virtual void writeNoiseToCondDb( const SiStripNoises& ) {;}
  
  /** Access to the configuration DB interface class. */
  SiStripConfigDb* db_;
  
  /** Container for DB connection parameters. */
  SiStripConfigDb::DbParams dbParams_;
  
};

#endif // OnlineDB_SiStripESSources_SiStripNoiseBuilderFromDb_H

