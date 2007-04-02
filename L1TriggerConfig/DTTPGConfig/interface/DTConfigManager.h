//-------------------------------------------------
//
/**  \class DTConfigManager
 *
 *   DTTPG Configuration manager 
 *   Includes config classes for every single chip
 *
 *   \author  C. Battilana
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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigBti.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTraco.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTSTheta.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTSPhi.h"
//#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTrigUnit.h" CB togli se nn serve!!!!!
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigSectColl.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "L1Trigger/DTUtilities/interface/DTBtiId.h"
#include "L1Trigger/DTUtilities/interface/DTTracoId.h"
#include "L1Trigger/DTUtilities/interface/DTSectCollId.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTGeometry;

//              ---------------------
//              -- Class Interface --
//              ---------------------


class DTConfigManager {

 public:
  
  typedef std::map<DTBtiId,DTConfigBti*>            innerBtiMap;
  typedef std::map<DTTracoId,DTConfigTraco*>        innerTracoMap;
  typedef std::map<DTChamberId,innerBtiMap>         BtiMap;
  typedef std::map<DTChamberId,innerTracoMap>       TracoMap;
  typedef std::map<DTChamberId,DTConfigTSTheta* >   TSThetaMap;
  typedef std::map<DTChamberId,DTConfigTSPhi* >     TSPhiMap;
  typedef std::map<DTSectCollId,DTConfigSectColl* > SectCollMap;

 public:
  
  //! Constructor
  DTConfigManager(edm::ParameterSet& , edm::ESHandle<DTGeometry>&);
  
  //! Destructor 
  ~DTConfigManager();

  //! Get desired BTI configuration
  DTConfigBti* getDTConfigBti(DTBtiId) const;

  //! Get desired BTI configuration map for a given DTChamber
  const std::map<DTBtiId,DTConfigBti* >& getDTConfigBtiMap(DTChamberId) const;
  
  //! Get desired TRACO configuration
  DTConfigTraco* getDTConfigTraco(DTTracoId) const;

  //! Get desired TRACO configuration map for a given DTChamber
  const std::map<DTTracoId,DTConfigTraco* >& getDTConfigTracoMap(DTChamberId) const;

  //! Get desired Trigger Server Theta configuration
  DTConfigTSTheta* getDTConfigTSTheta(DTChamberId) const;

  //! Get desired Trigger Server Phi configuration
  DTConfigTSPhi* getDTConfigTSPhi(DTChamberId) const;

/*   //! Get desired Trigger Unit configuration */
/*   DTConfigTrigUnit* getDTConfigTrigUnit(DTChamberId) const; */

  //! Get desired SectorCollector configuration
  DTConfigSectColl* getDTConfigSectColl(DTSectCollId) const;

  //! Get global debug flag
  inline bool getDTTPGDebug() const { return my_dttpgdebug; };

  //! Get global debug flag
  inline bool getDTTrigUnitDebug() const { return my_trigunitdebug; };

  //! Get BX Offset
  int getBXOffset() const;

 private:

  // maps for the whole config structure
  // BTI & TRACO use map<..,map<..,..> > to optimize access
  BtiMap       my_btimap;
  TracoMap     my_tracomap;
  TSThetaMap   my_tsthetamap;
  TSPhiMap     my_tsphimap;
  //std::map<DTChamberId,DTConfigTrigUnit*>     my_tumap; CB togliere!!!!!
  SectCollMap  my_sectcollmap;

  // pointer to the config classes to simplify construction/destruction
  DTConfigBti*      my_bticonf;
  DTConfigTraco*    my_tracoconf;
  DTConfigTSTheta*  my_tsthetaconf;
  DTConfigTSPhi*    my_tsphiconf;
  DTConfigSectColl* my_sectcollconf;
  
  int my_bxoffset;
  bool my_dttpgdebug;
  bool my_trigunitdebug;

};

#endif
