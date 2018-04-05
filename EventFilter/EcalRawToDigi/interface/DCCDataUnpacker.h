#ifndef DCCDATAUNPACKER_HH
#define DCCDATAUNPACKER_HH


/*
 *\ Class DCCDataUnpacker
 *
 * This class takes care of unpacking ECAL's raw data info.
 * A gateway for all blocks unpackers and committing collections to the Event
 * DCCEBEventBlock and DCCEEEventBlock are used here
 *
 * \file DCCDataUnpacker.h
 *
 * \author N. Almeida
 * \author G. Franzoni
 *
*/
//C++
#include <fstream>                   
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cstdio>                     
#include <cstdint>
#include <atomic>

//DATA DECODER

#include "DCCEventBlock.h"

#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalDigi/interface/EcalPnDiodeDigi.h>

#include <DataFormats/EcalDetId/interface/EcalDetIdCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

class EcalElectronicsMapper;
class DCCEventBlock;
class DCCEBEventBlock;
class DCCEEEventBlock;
class EcalRawToDigi;

class DCCDataUnpacker{

public : 
  
  DCCDataUnpacker(EcalElectronicsMapper *, bool hU,bool srpU, bool tccU, bool feU, bool memU, bool syncCheck, bool feIdCheck, bool forceToKeepFRdata);
  ~DCCDataUnpacker();
  /**
     Unpack data from a buffer
  */
  void unpack(const uint64_t* buffer, size_t bufferSize, unsigned int smId, unsigned int fedId);


  /**
    Set the collection pointers
  */

  void setEBDigisCollection( std::unique_ptr<EBDigiCollection>                         * x )
  { ebDigis_                = x; } 
 
  void setEEDigisCollection( std::unique_ptr<EEDigiCollection>                         * x )
  { eeDigis_                = x; } 
 
  void setDccHeadersCollection( std::unique_ptr<EcalRawDataCollection>                 * x )
  { dccHeaders_             = x; }
 
  void setEBSrFlagsCollection( std::unique_ptr<EBSrFlagCollection>                     * x )
  { ebSrFlags_              = x; } 
  
  void setEESrFlagsCollection( std::unique_ptr<EESrFlagCollection>                     * x )
  { eeSrFlags_              = x; }

  void setEcalTpsCollection( std::unique_ptr<EcalTrigPrimDigiCollection>                  * x )
  { ecalTps_                  = x; }

  void  setEcalPSsCollection( std::unique_ptr<EcalPSInputDigiCollection>                  * x )
  { ecalPSs_                  = x; }

  void setInvalidGainsCollection( std::unique_ptr<EBDetIdCollection>                    * x )
  { invalidGains_           = x; }
 
  void setInvalidGainsSwitchCollection( std::unique_ptr<EBDetIdCollection>              * x )
  { invalidGainsSwitch_     = x; }
 
  void setInvalidChIdsCollection( std::unique_ptr<EBDetIdCollection>                    * x )
  { invalidChIds_           = x; }

  // EE 
  void setInvalidEEGainsCollection( std::unique_ptr<EEDetIdCollection>                    * x )
  { invalidEEGains_           = x; }
  
  void setInvalidEEGainsSwitchCollection( std::unique_ptr<EEDetIdCollection>              * x )
  { invalidEEGainsSwitch_     = x; }
 
  void setInvalidEEChIdsCollection( std::unique_ptr<EEDetIdCollection>                    * x )
  { invalidEEChIds_           = x; }
  // EE 
 
  void setInvalidTTIdsCollection( std::unique_ptr<EcalElectronicsIdCollection>         * x )
  { invalidTTIds_           = x; }

  void setInvalidZSXtalIdsCollection( std::unique_ptr<EcalElectronicsIdCollection>     * x )
  { invalidZSXtalIds_           = x; }

  void setInvalidBlockLengthsCollection( std::unique_ptr<EcalElectronicsIdCollection>  * x )
  { invalidBlockLengths_    = x; }
 
  void setPnDiodeDigisCollection( std::unique_ptr<EcalPnDiodeDigiCollection>            * x )
  { pnDiodeDigis_           = x; }
 
  void setInvalidMemTtIdsCollection( std::unique_ptr<EcalElectronicsIdCollection>      * x )
  { invalidMemTtIds_        = x; }
 
  void setInvalidMemBlockSizesCollection( std::unique_ptr<EcalElectronicsIdCollection> * x )
  { invalidMemBlockSizes_   = x; }
 
  void setInvalidMemChIdsCollection( std::unique_ptr<EcalElectronicsIdCollection>      * x )
  { invalidMemChIds_        = x; }
 
  void setInvalidMemGainsCollection( std::unique_ptr<EcalElectronicsIdCollection>      * x )
  { invalidMemGains_        = x; }
  
 
  /**
   Get the collection pointers
  */
  
  std::unique_ptr<EBDigiCollection>             * ebDigisCollection()
  { return ebDigis_;               }
  
  std::unique_ptr<EEDigiCollection>             * eeDigisCollection()
  { return eeDigis_;               }
  
  std::unique_ptr<EcalTrigPrimDigiCollection>   * ecalTpsCollection()
  { return ecalTps_;                 } 
  
  std::unique_ptr<EcalPSInputDigiCollection>    * ecalPSsCollection()
  { return ecalPSs_;                 } 

  std::unique_ptr<EBSrFlagCollection>           * ebSrFlagsCollection()
  { return ebSrFlags_;             }  
  
  std::unique_ptr<EESrFlagCollection>           * eeSrFlagsCollection()
  { return eeSrFlags_;             } 
  
  std::unique_ptr<EcalRawDataCollection>        * dccHeadersCollection()
  { return dccHeaders_;            }
  
  std::unique_ptr<EBDetIdCollection>            * invalidGainsCollection()
  { return invalidGains_;          }
  
  std::unique_ptr<EBDetIdCollection>            * invalidGainsSwitchCollection()
  { return invalidGainsSwitch_;    }
  
  std::unique_ptr<EBDetIdCollection>            * invalidChIdsCollection()
  { return invalidChIds_;          }

  //EE
  std::unique_ptr<EEDetIdCollection>            * invalidEEGainsCollection()
  { return invalidEEGains_;          }
  
  std::unique_ptr<EEDetIdCollection>            * invalidEEGainsSwitchCollection()
  { return invalidEEGainsSwitch_;    }
  
  std::unique_ptr<EEDetIdCollection>            * invalidEEChIdsCollection()
  { return invalidEEChIds_;          }
  //EE

  std::unique_ptr<EcalElectronicsIdCollection> * invalidTTIdsCollection()
  { return invalidTTIds_;          }

  std::unique_ptr<EcalElectronicsIdCollection> * invalidZSXtalIdsCollection()
  { return invalidZSXtalIds_;          }  
  
  std::unique_ptr< EcalElectronicsIdCollection> * invalidBlockLengthsCollection()
  { return invalidBlockLengths_;   }
     
  std::unique_ptr<EcalElectronicsIdCollection>  * invalidMemTtIdsCollection()
  { return invalidMemTtIds_;       }
 
  std::unique_ptr<EcalElectronicsIdCollection>  * invalidMemBlockSizesCollection()
  { return invalidMemBlockSizes_;  }
  
  std::unique_ptr<EcalElectronicsIdCollection>  * invalidMemChIdsCollection()
  { return invalidMemChIds_;       }
  
  std::unique_ptr<EcalElectronicsIdCollection>  * invalidMemGainsCollection()
  { return invalidMemGains_;       }

  std::unique_ptr<EcalPnDiodeDigiCollection>    * pnDiodeDigisCollection()
  { return pnDiodeDigis_;          }
  

  /**
   Get the ECAL electronics Mapper
  */
  const EcalElectronicsMapper * electronicsMapper() const { return electronicsMapper_; }
  
  
  /**
   Functions to work with Channel Status DB
  */
  void setChannelStatusDB(const EcalChannelStatusMap* chdb) { chdb_ = chdb; }
  // return status of given crystal
  // https://twiki.cern.ch/twiki/bin/view/CMS/EcalChannelStatus#Assigning_Channel_status
  uint16_t getChannelStatus(const DetId& id) const;
  // return low 5 bits of status word
  uint16_t getChannelValue(const DetId& id) const;
  uint16_t getChannelValue(const int fed, const int ccu, const int strip, const int xtal) const;
  // return status of given CCU
  uint16_t getCCUValue(const int fed, const int ccu) const;
  
  
  /**
  Get the associated event
  */
  DCCEventBlock * currentEvent(){ return currentEvent_;}

  static std::atomic<bool> silentMode_; 
 
protected :

  // Data collections pointers
  std::unique_ptr<EBDigiCollection>            * ebDigis_;
  std::unique_ptr<EEDigiCollection>            * eeDigis_;
  std::unique_ptr<EcalTrigPrimDigiCollection>  * ecalTps_;
  std::unique_ptr<EcalPSInputDigiCollection>   * ecalPSs_;
  std::unique_ptr<EcalRawDataCollection>       * dccHeaders_;
  std::unique_ptr<EBDetIdCollection>           * invalidGains_;
  std::unique_ptr<EBDetIdCollection>           * invalidGainsSwitch_;
  std::unique_ptr<EBDetIdCollection>           * invalidChIds_;
  //EE
  std::unique_ptr<EEDetIdCollection>           * invalidEEGains_;
  std::unique_ptr<EEDetIdCollection>           * invalidEEGainsSwitch_;
  std::unique_ptr<EEDetIdCollection>           * invalidEEChIds_;
  //EE
  std::unique_ptr<EBSrFlagCollection>          * ebSrFlags_;
  std::unique_ptr<EESrFlagCollection>          * eeSrFlags_;
  std::unique_ptr<EcalElectronicsIdCollection> * invalidTTIds_;
  std::unique_ptr<EcalElectronicsIdCollection> * invalidZSXtalIds_;
  std::unique_ptr<EcalElectronicsIdCollection> * invalidBlockLengths_; 
  
  std::unique_ptr<EcalElectronicsIdCollection> * invalidMemTtIds_ ;
  std::unique_ptr<EcalElectronicsIdCollection> * invalidMemBlockSizes_ ;
  std::unique_ptr<EcalElectronicsIdCollection> * invalidMemChIds_ ;
  std::unique_ptr<EcalElectronicsIdCollection> * invalidMemGains_ ;
  std::unique_ptr<EcalPnDiodeDigiCollection>   * pnDiodeDigis_;

  EcalElectronicsMapper  * electronicsMapper_;
  const EcalChannelStatusMap* chdb_;
  DCCEventBlock          * currentEvent_;
  DCCEBEventBlock        * ebEventBlock_;
  DCCEEEventBlock        * eeEventBlock_;
		
};

#endif
