#ifndef _EcalElectronicsMapper_HH_
#define _EcalElectronicsMapper_HH_ 


/*
 *\ Class EcalElectronicsMapper
 *
 * Mapper for the ECAL electronics 
 
 * \file EcalElectronicsMapper.h
 *
 * $Date: 2009/06/04 18:39:33 $
 * $Revision: 1.2 $
 * \author N. Almeida
 * \author G. Franzoni
 *
*/


#include <iostream>                    
#include <fstream>
#include <string>
#include <map>


#include "DCCRawDataDefinitions.h"
#include <DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h>
#include <DataFormats/EcalDetId/interface/EcalElectronicsId.h>
#include <DataFormats/EcalDetId/interface/EcalScDetId.h>
#include <DataFormats/EcalDigi/interface/EcalDataFrame.h>
#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h>
#include <DataFormats/EcalDigi/interface/EcalPseudoStripInputDigi.h>
#include <DataFormats/EcalDigi/interface/EcalSrFlag.h>
#include <iostream>
#include <sstream>

#include <DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h>

class EcalElectronicsMapping;

class EcalElectronicsMapper{
  
public:


  /**
   * Constructor
   */
  EcalElectronicsMapper(uint numbOfXtalTSamples, uint numbOfTriggerTSamples);


  /**
   * Destructor
   */
  ~EcalElectronicsMapper();


  void setEcalElectronicsMapping(const EcalElectronicsMapping *); 


  /**
  * Set DCC id that is going to be unpacked for the event
  */
  bool setActiveDCC(uint dccId); 


  /**
   * Receives a string with a path and checks if file is accessible
   */
  bool setDCCMapFilePath(std::string );


  /**
   * Retrieves current path do the map file
   */
  std::string getDCCMapFilePath() const { return pathToMapFile_; }
  
  
  /**
   * Read map file (returns false if an error ocurred)
   *  deprecated by HLT environment
   */
  //  bool readDCCMapFile();
  //  bool readDCCMapFile(std::string );

  /**
   *  HLT friendly: load default mapping or, for non standatd mapping,
   *  use 2 vectors from cfg
   */
  bool makeMapFromVectors(std::vector<int>&, std::vector<int>&);
  
  /**
   * Get methods for DCCId/SMId and map
   */
  const std::map<uint ,uint>& getDCCMap() const { return myDCCMap_; }
  
  DetId  * getDetIdPointer(uint feChannel, uint strip, uint xtal){  return  xtalDetIds_[smId_-1][feChannel-1][strip-1][xtal-1];}
	 
  EcalTrigTowerDetId * getTTDetIdPointer(uint tccId, uint tower){ return ttDetIds_[tccId-1][tower-1];}
	 
  EcalElectronicsId  * getTTEleIdPointer(uint tccId, uint tower){ return ttEleIds_[tccId-1][tower-1];}

  EcalTriggerPrimitiveDigi * getTPPointer(uint tccId, uint tower){ return ttTPIds_[tccId-1][tower-1];}

  EcalScDetId  * getSCDetIdPointer(uint smId, uint feChannel){ return  scDetIds_[smId-1][feChannel-1];}

  EcalElectronicsId  * getSCElectronicsPointer(uint smId, uint feChannel){ return  scEleIds_[smId-1][feChannel-1];}

  EcalPseudoStripInputDigi  * getPSInputDigiPointer(uint tccId, uint towerId, uint psId){ return psInput_[tccId-1][towerId-1][psId-1];}
    
  EcalPseudoStripInputDigi  * getPSInputDigiPointer(uint tccId, uint psCounter){
      return getPSInputDigiPointer(tccId, tTandPs_[tccId-1][psCounter-1][0],tTandPs_[tccId-1][psCounter-1][1]);}

    
  // this getter method needs be clarified.
  // Changed by Ph.G. on July 1, 09: return a vector instead of a single
  // element. One SRF can be associated to two  supercrystals, because of
  // channel grouping.
  std::vector<EcalSrFlag*> getSrFlagPointer(uint feChannel){ return srFlags_[smId_-1][feChannel-1]; }
  
  std::vector<uint> * getTccs(uint smId){ return mapSmIdToTccIds_[smId];}
	
  uint getActiveDCC()                 { return dccId_;                      }
 
  uint getActiveSM()                  { return smId_;                       }

  uint numbXtalTSamples()             { return numbXtalTSamples_;           }

  uint numbTriggerTSamples()          { return numbTriggerTSamples_;        }
  
  uint getUnfilteredTowerBlockLength(){ return unfilteredFEBlockLength_;    }

  uint getEBTCCBlockLength()          { return ebTccBlockLength_;           }
  
  uint getEETCCBlockLength()          { return eeTccBlockLength_;           }
  
  uint getSRPBlockLength()            { return srpBlockLength_;             }
    
  uint getDCCId(uint aSMId) const;

  uint getSMId(uint aDCCId) const;
  
  uint getNumChannelsInDcc(uint aDCCId){return numChannelsInDcc_[aDCCId-1];}

  const EcalElectronicsMapping * mapping(){return mappingBuilder_;} 

  bool isTCCExternal(uint TCCId);
  
  /**
   * Print current map
   */
  friend std::ostream& operator<< (std::ostream &o, const EcalElectronicsMapper& aEcalElectronicsMapper);
  

  // Mantain this here as long as everything is moved to a general mapping
  enum SMGeom_t {
    kModules = 4,           // Number of modules per supermodule
    kTriggerTowers = 68,    // Number of trigger towers per supermodule
    kTowersInPhi = 4,       // Number of trigger towers in phi
    kTowersInEta = 17,      // Number of trigger towers in eta
    kCrystals = 1700,       // Number of crystals per supermodule
    kPns = 10,              // Number of PN laser monitoring diodes per supermodule
    kCrystalsInPhi = 20,    // Number of crystals in phi
    kCrystalsInEta = 85,    // Number of crystals in eta
    kCrystalsPerTower = 25, // Number of crystals per trigger tower
    kCardsPerTower = 5,     // Number of VFE cards per trigger tower
    kChannelsPerCard = 5,   // Number of channels per VFE card
    TTMAPMASK = 100
  };


private:

  void fillMaps();
  uint computeUnfilteredFEBlockLength();
  uint computeEBTCCBlockLength();
  uint computeEETCCBlockLength();

  std::string pathToMapFile_;
  
  uint numbXtalTSamples_;
  
  uint numbTriggerTSamples_;

  std::map<uint,uint> myDCCMap_;
  
  std::map< uint, std::vector<uint> * > mapSmIdToTccIds_;
  
  uint dccId_;
  
  uint smId_;
  
  uint unfilteredFEBlockLength_;
  
  uint srpBlockLength_;
  
  uint ebTccBlockLength_, eeTccBlockLength_;

  static const uint numChannelsInDcc_[NUMB_SM];

    
  // ARRAYS OF DetId  
  DetId                     * xtalDetIds_[NUMB_SM][NUMB_FE][NUMB_STRIP][NUMB_XTAL];
  EcalScDetId               * scDetIds_[NUMB_SM][NUMB_FE];
  EcalElectronicsId         * scEleIds_[NUMB_SM][NUMB_FE];
  EcalTrigTowerDetId        * ttDetIds_[NUMB_TCC][NUMB_FE];
  EcalElectronicsId         * ttEleIds_[NUMB_TCC][NUMB_FE];
  EcalTriggerPrimitiveDigi  * ttTPIds_[NUMB_TCC][NUMB_FE];
  std::vector<EcalSrFlag*>  srFlags_[NUMB_SM][NUMB_FE];
  EcalPseudoStripInputDigi  * psInput_[NUMB_TCC][TCC_EB_NUMBTTS][NUMB_STRIP];
    
  short tTandPs_[NUMB_TCC][5*EcalTrigTowerDetId::kEBTowersPerSM][2];
    
  const EcalElectronicsMapping    * mappingBuilder_;
  

};

#endif
