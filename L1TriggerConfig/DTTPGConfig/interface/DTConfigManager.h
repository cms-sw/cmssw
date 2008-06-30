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
 *
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
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTBtiId.h"
#include "DataFormats/MuonDetId/interface/DTTracoId.h"
#include "DataFormats/MuonDetId/interface/DTSectCollId.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//              ---------------------
//              -- Class Interface --
//              ---------------------


class DTConfigManager {

 public:
  
  typedef std::map<DTBtiId,DTConfigBti>            innerBtiMap;
  typedef std::map<DTTracoId,DTConfigTraco>        innerTracoMap;
  typedef std::map<DTChamberId,innerBtiMap>        BtiMap;
  typedef std::map<DTChamberId,innerTracoMap>      TracoMap;
  typedef std::map<DTChamberId,DTConfigTSTheta>    TSThetaMap;
  typedef std::map<DTChamberId,DTConfigTSPhi>      TSPhiMap;
  typedef std::map<DTChamberId,DTConfigTrigUnit>   TrigUnitMap;
  typedef std::map<DTSectCollId,DTConfigSectColl>  SectCollMap;

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

  //! Get desired SectorCollector configuration
  DTConfigSectColl* getDTConfigSectColl(DTSectCollId) const;

  //! Set DTConfigBti for desired chip
  void setDTConfigBti(DTBtiId,DTConfigBti);

  //! Set DTConfigTraco for desired chip
  void setDTConfigTraco(DTTracoId,DTConfigTraco);

  //! Set DTConfigTSTheta for desired chip
  inline void setDTConfigTSTheta(DTChamberId chambid ,DTConfigTSTheta conf) { my_tsthetamap[chambid] = conf; };

  //! Set DTConfigTSPhi for desired chip
  inline void setDTConfigTSPhi(DTChamberId chambid,DTConfigTSPhi conf) { my_tsphimap[chambid] = conf; };

  //! Set DTConfigTrigUnit for desired chip
  void setDTConfigTrigUnit(DTChamberId chambid,DTConfigTrigUnit conf) { my_trigunitmap[chambid] = conf; };

  //! Set DTConfigSectColl for desired chip
  void setDTConfigSectColl(DTSectCollId sectcollid ,DTConfigSectColl conf){ my_sectcollmap[sectcollid] = conf; };

  //! Get global debug flag
  inline bool getDTTPGDebug() const { return my_dttpgdebug; };

  //! SetGlobalDebug flag
  inline void setDTTPGDebug(bool debug) { my_dttpgdebug = debug; };
  
  //! Get BX Offset
  int getBXOffset() const;

 private:

  // maps for the whole config structure
  // BTI & TRACO use map<..,map<..,..> > to optimize access
  BtiMap       my_btimap;
  TracoMap     my_tracomap;
  TSThetaMap   my_tsthetamap;
  TSPhiMap     my_tsphimap;
  TrigUnitMap  my_trigunitmap; 
  SectCollMap  my_sectcollmap;
  
  int my_bxoffset;
  bool my_dttpgdebug;
};

#endif
