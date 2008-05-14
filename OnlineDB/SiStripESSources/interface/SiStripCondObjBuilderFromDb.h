// Last commit: $Id: SiStripCondObjBuilderFromDb.h,v 1.2 2007/03/19 13:23:06 bainbrid Exp $
// Latest tag:  $Name: V01-01-00 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripESSources/interface/SiStripCondObjBuilderFromDb.h,v $

#ifndef OnlineDB_SiStripESSources_SiStripCondObjBuilderFromDb_H
#define OnlineDB_SiStripESSources_SiStripCondObjBuilderFromDb_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "boost/cstdint.hpp"
#include <vector>
#include <string>

class SiStripFecCabling;
class SiStripDetCabling;
class SiStripPedestals;
class SiStripNoises;
class SiStripQuality;
class SiStripThreshold;
class DcuDetIdMap;

class SiStripCondObjBuilderFromDb {
  
 public:

  SiStripCondObjBuilderFromDb();
  virtual ~SiStripCondObjBuilderFromDb();
  
  
  /** Builds pedestals using FED descriptions and cabling info
      retrieved from configuration database. */
  void buildCondObj();
  void buildStripRelatedObjects( SiStripConfigDb* const db,
				 const SiStripDetCabling& det_cabling);

  SiStripFedCabling* getFedCabling() {return fed_cabling_;}
  SiStripPedestals *  getPedestals() {return pedestals_;}  
  SiStripNoises    *  getNoises()    {return noises_;}  
  SiStripThreshold *  getThreshold() {return threshold_;}  
  SiStripQuality   *  getQuality()   {return quality_;}  
 
 protected:
  
  
  /** Access to the configuration DB interface class. */
  SiStripConfigDb* db_;
  
  /** Container for DB connection parameters. */
  SiStripConfigDb::DbParams dbParams_;

  SiStripFedCabling* fed_cabling_;
  SiStripPedestals* pedestals_;  
  SiStripNoises* noises_;  
  SiStripThreshold* threshold_;  
  SiStripQuality* quality_;  
  
};

#endif // OnlineDB_SiStripESSources_SiStripCondObjBuilderFromDb_H

