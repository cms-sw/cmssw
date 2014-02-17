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
 * $Date: 2012/08/06 21:51:35 $
 * $Revision: 1.6 $
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
#include <stdio.h>                     
#include <stdint.h>

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

  void setEBDigisCollection( std::auto_ptr<EBDigiCollection>                         * x )
  { ebDigis_                = x; } 
 
  void setEEDigisCollection( std::auto_ptr<EEDigiCollection>                         * x )
  { eeDigis_                = x; } 
 
  void setDccHeadersCollection( std::auto_ptr<EcalRawDataCollection>                 * x )
  { dccHeaders_             = x; }
 
  void setEBSrFlagsCollection( std::auto_ptr<EBSrFlagCollection>                     * x )
  { ebSrFlags_              = x; } 
  
  void setEESrFlagsCollection( std::auto_ptr<EESrFlagCollection>                     * x )
  { eeSrFlags_              = x; }

  void setEcalTpsCollection( std::auto_ptr<EcalTrigPrimDigiCollection>                  * x )
  { ecalTps_                  = x; }

  void  setEcalPSsCollection( std::auto_ptr<EcalPSInputDigiCollection>                  * x )
  { ecalPSs_                  = x; }

  void setInvalidGainsCollection( std::auto_ptr<EBDetIdCollection>                    * x )
  { invalidGains_           = x; }
 
  void setInvalidGainsSwitchCollection( std::auto_ptr<EBDetIdCollection>              * x )
  { invalidGainsSwitch_     = x; }
 
  void setInvalidChIdsCollection( std::auto_ptr<EBDetIdCollection>                    * x )
  { invalidChIds_           = x; }

  // EE 
  void setInvalidEEGainsCollection( std::auto_ptr<EEDetIdCollection>                    * x )
  { invalidEEGains_           = x; }
  
  void setInvalidEEGainsSwitchCollection( std::auto_ptr<EEDetIdCollection>              * x )
  { invalidEEGainsSwitch_     = x; }
 
  void setInvalidEEChIdsCollection( std::auto_ptr<EEDetIdCollection>                    * x )
  { invalidEEChIds_           = x; }
  // EE 
 
  void setInvalidTTIdsCollection( std::auto_ptr<EcalElectronicsIdCollection>         * x )
  { invalidTTIds_           = x; }

  void setInvalidZSXtalIdsCollection( std::auto_ptr<EcalElectronicsIdCollection>     * x )
  { invalidZSXtalIds_           = x; }

  void setInvalidBlockLengthsCollection( std::auto_ptr<EcalElectronicsIdCollection>  * x )
  { invalidBlockLengths_    = x; }
 
  void setPnDiodeDigisCollection( std::auto_ptr<EcalPnDiodeDigiCollection>            * x )
  { pnDiodeDigis_           = x; }
 
  void setInvalidMemTtIdsCollection( std::auto_ptr<EcalElectronicsIdCollection>      * x )
  { invalidMemTtIds_        = x; }
 
  void setInvalidMemBlockSizesCollection( std::auto_ptr<EcalElectronicsIdCollection> * x )
  { invalidMemBlockSizes_   = x; }
 
  void setInvalidMemChIdsCollection( std::auto_ptr<EcalElectronicsIdCollection>      * x )
  { invalidMemChIds_        = x; }
 
  void setInvalidMemGainsCollection( std::auto_ptr<EcalElectronicsIdCollection>      * x )
  { invalidMemGains_        = x; }
  
 
  /**
   Get the collection pointers
  */
  
  std::auto_ptr<EBDigiCollection>             * ebDigisCollection()
  { return ebDigis_;               }
  
  std::auto_ptr<EEDigiCollection>             * eeDigisCollection()
  { return eeDigis_;               }
  
  std::auto_ptr<EcalTrigPrimDigiCollection>   * ecalTpsCollection()
  { return ecalTps_;                 } 
  
  std::auto_ptr<EcalPSInputDigiCollection>    * ecalPSsCollection()
  { return ecalPSs_;                 } 

  std::auto_ptr<EBSrFlagCollection>           * ebSrFlagsCollection()
  { return ebSrFlags_;             }  
  
  std::auto_ptr<EESrFlagCollection>           * eeSrFlagsCollection()
  { return eeSrFlags_;             } 
  
  std::auto_ptr<EcalRawDataCollection>        * dccHeadersCollection()
  { return dccHeaders_;            }
  
  std::auto_ptr<EBDetIdCollection>            * invalidGainsCollection()
  { return invalidGains_;          }
  
  std::auto_ptr<EBDetIdCollection>            * invalidGainsSwitchCollection()
  { return invalidGainsSwitch_;    }
  
  std::auto_ptr<EBDetIdCollection>            * invalidChIdsCollection()
  { return invalidChIds_;          }

  //EE
  std::auto_ptr<EEDetIdCollection>            * invalidEEGainsCollection()
  { return invalidEEGains_;          }
  
  std::auto_ptr<EEDetIdCollection>            * invalidEEGainsSwitchCollection()
  { return invalidEEGainsSwitch_;    }
  
  std::auto_ptr<EEDetIdCollection>            * invalidEEChIdsCollection()
  { return invalidEEChIds_;          }
  //EE

  std::auto_ptr<EcalElectronicsIdCollection> * invalidTTIdsCollection()
  { return invalidTTIds_;          }

  std::auto_ptr<EcalElectronicsIdCollection> * invalidZSXtalIdsCollection()
  { return invalidZSXtalIds_;          }  
  
  std::auto_ptr< EcalElectronicsIdCollection> * invalidBlockLengthsCollection()
  { return invalidBlockLengths_;   }
     
  std::auto_ptr<EcalElectronicsIdCollection>  * invalidMemTtIdsCollection()
  { return invalidMemTtIds_;       }
 
  std::auto_ptr<EcalElectronicsIdCollection>  * invalidMemBlockSizesCollection()
  { return invalidMemBlockSizes_;  }
  
  std::auto_ptr<EcalElectronicsIdCollection>  * invalidMemChIdsCollection()
  { return invalidMemChIds_;       }
  
  std::auto_ptr<EcalElectronicsIdCollection>  * invalidMemGainsCollection()
  { return invalidMemGains_;       }

  std::auto_ptr<EcalPnDiodeDigiCollection>    * pnDiodeDigisCollection()
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

  static bool silentMode_; 
 
protected :

  // Data collections pointers
  std::auto_ptr<EBDigiCollection>            * ebDigis_;
  std::auto_ptr<EEDigiCollection>            * eeDigis_;
  std::auto_ptr<EcalTrigPrimDigiCollection>  * ecalTps_;
  std::auto_ptr<EcalPSInputDigiCollection>   * ecalPSs_;
  std::auto_ptr<EcalRawDataCollection>       * dccHeaders_;
  std::auto_ptr<EBDetIdCollection>           * invalidGains_;
  std::auto_ptr<EBDetIdCollection>           * invalidGainsSwitch_;
  std::auto_ptr<EBDetIdCollection>           * invalidChIds_;
  //EE
  std::auto_ptr<EEDetIdCollection>           * invalidEEGains_;
  std::auto_ptr<EEDetIdCollection>           * invalidEEGainsSwitch_;
  std::auto_ptr<EEDetIdCollection>           * invalidEEChIds_;
  //EE
  std::auto_ptr<EBSrFlagCollection>          * ebSrFlags_;
  std::auto_ptr<EESrFlagCollection>          * eeSrFlags_;
  std::auto_ptr<EcalElectronicsIdCollection> * invalidTTIds_;
  std::auto_ptr<EcalElectronicsIdCollection> * invalidZSXtalIds_;
  std::auto_ptr<EcalElectronicsIdCollection> * invalidBlockLengths_; 
  
  std::auto_ptr<EcalElectronicsIdCollection> * invalidMemTtIds_ ;
  std::auto_ptr<EcalElectronicsIdCollection> * invalidMemBlockSizes_ ;
  std::auto_ptr<EcalElectronicsIdCollection> * invalidMemChIds_ ;
  std::auto_ptr<EcalElectronicsIdCollection> * invalidMemGains_ ;
  std::auto_ptr<EcalPnDiodeDigiCollection>   * pnDiodeDigis_;

  EcalElectronicsMapper  * electronicsMapper_;
  const EcalChannelStatusMap* chdb_;
  DCCEventBlock          * currentEvent_;
  DCCEBEventBlock        * ebEventBlock_;
  DCCEEEventBlock        * eeEventBlock_;
		
};

#endif
