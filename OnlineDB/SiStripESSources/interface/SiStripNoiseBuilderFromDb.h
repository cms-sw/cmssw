// Last commit: $Id: SiStripNoiseBuilderFromDb.h,v 1.1 2006/12/22 12:20:35 bainbrid Exp $
// Latest tag:  $Name: TIF_190307 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripESSources/interface/SiStripNoiseBuilderFromDb.h,v $

#ifndef OnlineDB_SiStripESSources_SiStripNoiseBuilderFromDb_H
#define OnlineDB_SiStripESSources_SiStripNoiseBuilderFromDb_H

#include "CalibTracker/SiStripPedestals/interface/SiStripNoiseESSource.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
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

