#ifndef _EcalElectronicsMapper_HH_
#define _EcalElectronicsMapper_HH_ 


/*
 *\ Class EcalElectronicsMapper
 *
 * Mapper for the ECAL electronics 
 
 * \file EcalElectronicsMapper.h
 *
 * $Date: 2010/09/30 16:45:32 $
 * $Revision: 1.6 $
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

class EcalElectronicsMapper {

public:

  /**
   * Constructor
   */
  EcalElectronicsMapper(unsigned int numbOfXtalTSamples, unsigned int numbOfTriggerTSamples);


  /**
   * Destructor
   */
  ~EcalElectronicsMapper();


  void setEcalElectronicsMapping(const EcalElectronicsMapping *); 

  void deletePointers();
  void resetPointers();

  /**
  * Set DCC id that is going to be unpacked for the event
  */
  bool setActiveDCC(unsigned int dccId); 


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
  const std::map<unsigned int ,unsigned int>& getDCCMap() const { return myDCCMap_; }
  
  DetId  * getDetIdPointer(unsigned int feChannel, unsigned int strip, unsigned int xtal){  return  xtalDetIds_[smId_-1][feChannel-1][strip-1][xtal-1];}
	 
  EcalTrigTowerDetId * getTTDetIdPointer(unsigned int tccId, unsigned int tower){ return ttDetIds_[tccId-1][tower-1];}
	 
  EcalElectronicsId  * getTTEleIdPointer(unsigned int tccId, unsigned int tower){ return ttEleIds_[tccId-1][tower-1];}

  EcalTriggerPrimitiveDigi * getTPPointer(unsigned int tccId, unsigned int tower){ return ttTPIds_[tccId-1][tower-1];}

  EcalScDetId  * getSCDetIdPointer(unsigned int smId, unsigned int feChannel){ return  scDetIds_[smId-1][feChannel-1];}

  EcalElectronicsId  * getSCElectronicsPointer(unsigned int smId, unsigned int feChannel){ return  scEleIds_[smId-1][feChannel-1];}

  EcalPseudoStripInputDigi  * getPSInputDigiPointer(unsigned int tccId, unsigned int towerId, unsigned int psId){ return psInput_[tccId-1][towerId-1][psId-1];}
    
  EcalPseudoStripInputDigi  * getPSInputDigiPointer(unsigned int tccId, unsigned int psCounter){
      return getPSInputDigiPointer(tccId, tTandPs_[tccId-1][psCounter-1][0],tTandPs_[tccId-1][psCounter-1][1]);}

    
  // this getter method needs be clarified.
  // Changed by Ph.G. on July 1, 09: return a vector instead of a single
  // element. One SRF can be associated to two  supercrystals, because of
  // channel grouping.
  std::vector<EcalSrFlag*> getSrFlagPointer(unsigned int feChannel){ return srFlags_[smId_-1][feChannel-1]; }
  
  std::vector<unsigned int> * getTccs(unsigned int smId){ return mapSmIdToTccIds_[smId];}
	
  unsigned int getActiveDCC()                 { return dccId_;                      }
 
  unsigned int getActiveSM()                  { return smId_;                       }

  unsigned int numbXtalTSamples()             { return numbXtalTSamples_;           }

  unsigned int numbTriggerTSamples()          { return numbTriggerTSamples_;        }
  
  unsigned int getUnfilteredTowerBlockLength(){ return unfilteredFEBlockLength_;    }

  unsigned int getEBTCCBlockLength()          { return ebTccBlockLength_;           }
  
  unsigned int getEETCCBlockLength()          { return eeTccBlockLength_;           }
  
  unsigned int getSRPBlockLength()            { return srpBlockLength_;             }
    
  unsigned int getDCCId(unsigned int aSMId) const;

  unsigned int getSMId(unsigned int aDCCId) const;
  
  unsigned int getNumChannelsInDcc(unsigned int aDCCId){return numChannelsInDcc_[aDCCId-1];}

  const EcalElectronicsMapping * mapping(){return mappingBuilder_;} 

  bool isTCCExternal(unsigned int TCCId);
  
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
  unsigned int computeUnfilteredFEBlockLength();
  unsigned int computeEBTCCBlockLength();
  unsigned int computeEETCCBlockLength();

  std::string pathToMapFile_;
  
  unsigned int numbXtalTSamples_;
  
  unsigned int numbTriggerTSamples_;

  std::map<unsigned int,unsigned int> myDCCMap_;
  
  std::map< unsigned int, std::vector<unsigned int> * > mapSmIdToTccIds_;
  
  unsigned int dccId_;
  
  unsigned int smId_;
  
  unsigned int unfilteredFEBlockLength_;
  
  unsigned int srpBlockLength_;
  
  unsigned int ebTccBlockLength_, eeTccBlockLength_;

  static const unsigned int numChannelsInDcc_[NUMB_SM];

    
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
  

// functions and fields to work with 'ghost' VFEs:
public:
  // check, does the given [FED (dcc), CCU (tower), VFE (strip)] belongs
  // to the list of VFEs with 'ghost' channels
  bool isGhost(const int FED, const int CCU, const int VFE);
  
private:
  void setupGhostMap();
  std::map<int, std::map<int, std::map<int, bool> > > ghost_;
};

#endif
