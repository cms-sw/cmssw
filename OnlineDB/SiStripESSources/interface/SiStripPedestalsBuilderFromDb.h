// Last commit: $Id: SiStripPedestalsBuilderFromDb.h,v 1.3 2008/04/08 09:33:49 bainbrid Exp $
// Latest tag:  $Name: V02-00-00 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripESSources/interface/SiStripPedestalsBuilderFromDb.h,v $

#ifndef OnlineDB_SiStripESSources_SiStripPedestalsBuilderFromDb_H
#define OnlineDB_SiStripESSources_SiStripPedestalsBuilderFromDb_H

#include "CalibTracker/SiStripESProducers/interface/SiStripPedestalsESSource.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripDbParams.h"
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
  SiStripDbParams dbParams_;
  
};

#endif // OnlineDB_SiStripESSources_SiStripPedestalsBuilderFromDb_H

