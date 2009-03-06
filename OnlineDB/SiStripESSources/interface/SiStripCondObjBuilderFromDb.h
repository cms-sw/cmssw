// Last commit: $Id: SiStripCondObjBuilderFromDb.h,v 1.2 2008/05/16 15:30:07 bainbrid Exp $
// Latest tag:  $Name: V02-00-02 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripESSources/interface/SiStripCondObjBuilderFromDb.h,v $

#ifndef OnlineDB_SiStripESSources_SiStripCondObjBuilderFromDb_H
#define OnlineDB_SiStripESSources_SiStripCondObjBuilderFromDb_H

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripDbParams.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"

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
  SiStripCondObjBuilderFromDb(const edm::ParameterSet&,
			      const edm::ActivityRegistry&);
  virtual ~SiStripCondObjBuilderFromDb();
  
  /** Returns database connection parameters. */
  inline const SiStripDbParams& dbParams() const {return db_->dbParams();} 

  /** Builds pedestals using FED descriptions and cabling info
      retrieved from configuration database. */
  void buildCondObj();
  void buildStripRelatedObjects( SiStripConfigDb* const db,
				 const SiStripDetCabling& det_cabling);

  SiStripFedCabling*  getFedCabling() {checkUpdate(); return fed_cabling_;}
  SiStripPedestals *  getPedestals()  {checkUpdate(); return pedestals_;}  
  SiStripNoises    *  getNoises()     {checkUpdate(); return noises_;}  
  SiStripThreshold *  getThreshold()  {checkUpdate(); return threshold_;}  
  SiStripQuality   *  getQuality()    {checkUpdate(); return quality_;}  

  void getValue(SiStripFedCabling* & val){ val = getFedCabling();}
  void getValue(SiStripPedestals * & val){ val = getPedestals(); }  
  void getValue(SiStripNoises    * & val){ val = getNoises();    }  
  void getValue(SiStripThreshold * & val){ val = getThreshold(); }  
  void getValue(SiStripQuality   * & val){ val = getQuality();   }  
  void getValue(SiStripBadStrip  * & val){ val = new SiStripBadStrip(* (const SiStripBadStrip*) getQuality());   }  
  

 protected:
  
  void checkUpdate();
  
  /** Access to the configuration DB interface class. */
  // Build and retrieve SiStripConfigDb object using service
  edm::Service<SiStripConfigDb> db_;
  
  /** Container for DB connection parameters. */
  SiStripDbParams dbParams_;

  SiStripFedCabling  *fed_cabling_;
  SiStripPedestals   *pedestals_;  
  SiStripNoises      *noises_;  
  SiStripThreshold   *threshold_;  
  SiStripQuality     *quality_;  
  
};

#endif // OnlineDB_SiStripESSources_SiStripCondObjBuilderFromDb_H

