//-------------------------------------------------
//
/**  \class DTConfigManager
 *
 *   DTTPG Configuration manager 
 *   Includes config classes for every single chip
 *
 *   \author  C. Battilana
 *   april 07 : SV DTConfigTrigUnit added
 *   april 07 : CB Removed DTGeometry dependecies
 *   september 08 : SV LUTs added
 *   091106 SV flags for DB/geometry lut or bti acceptance compute
 */
//
//--------------------------------------------------
#ifndef DT_CONFIG_MANAGER_H
#define DT_CONFIG_MANAGER_H

//---------------
// C++ Headers --
//---------------
#include <map>

//----------------------
// Base Class Headers --
//----------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigBti.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTraco.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTSTheta.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTSPhi.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTrigUnit.h" 
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigSectColl.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigLUTs.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTBtiId.h"
#include "DataFormats/MuonDetId/interface/DTTracoId.h"
#include "DataFormats/MuonDetId/interface/DTSectCollId.h"

#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigPedestals.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//              ---------------------
//              -- Class Interface --
//              ---------------------


class DTConfigManager {

 public:
  
  typedef std::map<DTBtiId,DTConfigBti> innerBtiMap;
  typedef std::map<DTTracoId,DTConfigTraco> innerTracoMap;
  typedef std::map<DTChamberId,innerBtiMap> BtiMap;
  typedef std::map<DTChamberId,innerTracoMap> TracoMap;
  typedef std::map<DTChamberId,DTConfigTSTheta> TSThetaMap;
  typedef std::map<DTChamberId,DTConfigTSPhi> TSPhiMap;
  typedef std::map<DTChamberId,DTConfigTrigUnit> TrigUnitMap;
  typedef std::map<DTChamberId,DTConfigLUTs> LUTMap;
  typedef std::map<DTSectCollId,DTConfigSectColl> SectCollMap;

 public:
  
  //! Constructor
  DTConfigManager();
  
  //! Destructor 
  ~DTConfigManager();

  //! Get desired BTI configuration
  DTConfigBti* getDTConfigBti(DTBtiId) const;

  //! Get desired BTI configuration map for a given DTChamber
  const std::map<DTBtiId,DTConfigBti>& getDTConfigBtiMap(DTChamberId) const;
  
  //! Get desired TRACO configuration
  DTConfigTraco* getDTConfigTraco(DTTracoId) const;

  //! Get desired TRACO configuration map for a given DTChamber
  const std::map<DTTracoId,DTConfigTraco>& getDTConfigTracoMap(DTChamberId) const;

  //! Get desired Trigger Server Theta configuration
  DTConfigTSTheta* getDTConfigTSTheta(DTChamberId) const;

  //! Get desired Trigger Server Phi configuration
  DTConfigTSPhi* getDTConfigTSPhi(DTChamberId) const;

  //! Get desired Trigger Unit configuration 
  DTConfigTrigUnit* getDTConfigTrigUnit(DTChamberId) const;
  
   //! Get desired LUT configuration 
  DTConfigLUTs* getDTConfigLUTs(DTChamberId) const;

  //! Get desired SectorCollector configuration
  DTConfigSectColl* getDTConfigSectColl(DTSectCollId) const;

  //! Get desired Pedestals configuration
  DTConfigPedestals* getDTConfigPedestals() const;
 
  //! Get global debug flag
  inline bool getDTTPGDebug() const { return my_dttpgdebug; };

  //! Get BX Offset for a given vdrift config
  int getBXOffset() const;

  //! Lut from DB flag
  inline bool lutFromDB() const { return my_lutfromdb; }

  //! Use Bti acceptance parameters (LL,LH,CL,CH,RL,RH)
  inline bool useAcceptParam() const { return my_acceptparam; }

  //! flag for CCB configuration validity
  inline bool CCBConfigValidity() const { return my_CCBvalid; }
 

  //! Set DTConfigBti for desired chip
  void setDTConfigBti(DTBtiId,DTConfigBti);

  //! Set DTConfigTraco for desired chip
  void setDTConfigTraco(DTTracoId,DTConfigTraco);

  //! Set DTConfigTSTheta for desired chip
  inline void setDTConfigTSTheta(DTChamberId chambid ,DTConfigTSTheta conf) { my_tsthetamap[chambid] = conf; };

  //! Set DTConfigTSPhi for desired chip
  inline void setDTConfigTSPhi(DTChamberId chambid,DTConfigTSPhi conf) { my_tsphimap[chambid] = conf; };

  //! Set DTConfigTrigUnit for desired chamber
  void setDTConfigTrigUnit(DTChamberId chambid,DTConfigTrigUnit conf) { my_trigunitmap[chambid] = conf; };

  //! Set DTConfigLUTs for desired chamber
  void setDTConfigLUTs(DTChamberId chambid,DTConfigLUTs conf) { my_lutmap[chambid] = conf; };

  //! Set DTConfigSectColl for desired chip
  void setDTConfigSectColl(DTSectCollId sectcollid ,DTConfigSectColl conf){ my_sectcollmap[sectcollid] = conf; };

  //! Set DTConfigPedestals configuration 
  void setDTConfigPedestals(DTConfigPedestals pedestals) { my_pedestals = pedestals; };

  //! SetGlobalDebug flag
  inline void setDTTPGDebug(bool debug) { my_dttpgdebug = debug; }
   
  //! Set lut from DB flag
  inline void setLutFromDB(bool lutFromDB) { my_lutfromdb = lutFromDB; }

  //! Set the use of Bti acceptance parameters (LL,LH,CL,CH,RL,RH)
  inline void setUseAcceptParam(bool acceptparam) { my_acceptparam = acceptparam; }

  //! Set the flag for CCB configuration validity
  inline void setCCBConfigValidity(bool CCBValid) { my_CCBvalid = CCBValid; }

  //! Dump luts string commands from configuration parameters
  void dumpLUTParam(DTChamberId &chambid) const; /* SV 091111 */ 


 private:

  // maps for the whole config structure
  BtiMap       my_btimap;
  TracoMap     my_tracomap;
  TSThetaMap   my_tsthetamap;
  TSPhiMap     my_tsphimap;
  TrigUnitMap  my_trigunitmap; 
  LUTMap       my_lutmap;
  SectCollMap  my_sectcollmap;
  DTConfigPedestals my_pedestals;
  
  bool my_dttpgdebug;

  bool my_lutfromdb;
  bool my_acceptparam;
  bool my_CCBvalid;
};

#endif
