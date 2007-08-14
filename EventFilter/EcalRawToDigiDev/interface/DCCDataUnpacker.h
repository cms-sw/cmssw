#ifndef DCCDATAUNPACKER_HH
#define DCCDATAUNPACKER_HH


/*
 *\ Class DCCDataUnpacker
 *
 * This class takes care of unpacking ECAL's raw data info
 *
 * \file DCCDataUnpacker.h
 *
 * $Date: 2007/07/24 11:39:35 $
 * $Revision: 1.5 $
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

class EcalElectronicsMapper;
class DCCEventBlock;
class DCCEBEventBlock;
class DCCEEEventBlock;
class EcalRawToDigi;

class DCCDataUnpacker{

public : 
  
  DCCDataUnpacker(EcalElectronicsMapper *, bool hU,bool srpU, bool tccU, bool feU, bool memU, bool syncCheck);
  ~DCCDataUnpacker();
  /**
     Unpack data from a buffer
  */
  void unpack( uint64_t * buffer, uint bufferSize, uint smId, uint fedId);


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
 
  void setEBTpsCollection( std::auto_ptr<EcalTrigPrimDigiCollection>                  * x )
  { ebTps_                  = x; }
  
  void setEETpsCollection( std::auto_ptr<EcalTrigPrimDigiCollection>                  * x )
  { eeTps_                  = x; }
 
  void setInvalidGainsCollection( std::auto_ptr<EBDetIdCollection>                    * x )
  { invalidGains_           = x; }
 
  void setInvalidGainsSwitchCollection( std::auto_ptr<EBDetIdCollection>              * x )
  { invalidGainsSwitch_     = x; }
 
  void setInvalidGainsSwitchStayCollection( std::auto_ptr<EBDetIdCollection>          * x )
  { invalidGainsSwitchStay_ = x; }
 
  void setInvalidChIdsCollection( std::auto_ptr<EBDetIdCollection>                    * x )
  { invalidChIds_           = x; }
 
  void setInvalidTTIdsCollection( std::auto_ptr<EcalTrigTowerDetIdCollection>         * x )
  { invalidTTIds_           = x; }
 
  void setInvalidBlockLengthsCollection( std::auto_ptr<EcalTrigTowerDetIdCollection>  * x )
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
  
  std::auto_ptr<EcalTrigPrimDigiCollection>   * ebTpsCollection()
  { return ebTps_;                 } 
  
  std::auto_ptr<EcalTrigPrimDigiCollection>   * eeTpsCollection()
  { return eeTps_;                 } 
  
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
  
  std::auto_ptr<EBDetIdCollection>            * invalidGainsSwitchStayCollection()
  { return invalidGainsSwitchStay_;}
  
  std::auto_ptr<EBDetIdCollection>            * invalidChIdsCollection()
  { return invalidChIds_;          }
      
  std::auto_ptr<EcalTrigTowerDetIdCollection> * invalidTTIdsCollection()
  { return invalidTTIds_;          }  
  
  std::auto_ptr<EcalTrigTowerDetIdCollection> * invalidBlockLengthsCollection()
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
  EcalElectronicsMapper * electronicsMapper(){return electronicsMapper_;}
  
  /**
  Get the associated event
  */
  DCCEventBlock * currentEvent(){ return currentEvent_;}
 
protected :

  // Data collections pointers
  std::auto_ptr<EBDigiCollection>                       * ebDigis_;
  std::auto_ptr<EEDigiCollection>                       * eeDigis_;
  std::auto_ptr<EcalTrigPrimDigiCollection >      * ebTps_;
  std::auto_ptr<EcalTrigPrimDigiCollection >      * eeTps_;
  std::auto_ptr<EcalRawDataCollection>             * dccHeaders_;
  std::auto_ptr<EBDetIdCollection>                    * invalidGains_;
  std::auto_ptr<EBDetIdCollection>                    * invalidGainsSwitch_;
  std::auto_ptr<EBDetIdCollection>                    * invalidGainsSwitchStay_;
  std::auto_ptr<EBDetIdCollection>                    * invalidChIds_;
  std::auto_ptr<EBSrFlagCollection>                   * ebSrFlags_;
  std::auto_ptr<EESrFlagCollection>                   * eeSrFlags_;
  std::auto_ptr<EcalTrigTowerDetIdCollection> * invalidTTIds_;
  std::auto_ptr<EcalTrigTowerDetIdCollection> * invalidBlockLengths_; 
  
  
  std::auto_ptr<EcalElectronicsIdCollection>  * invalidMemTtIds_ ;
  std::auto_ptr<EcalElectronicsIdCollection>  * invalidMemBlockSizes_ ;
  std::auto_ptr<EcalElectronicsIdCollection>  * invalidMemChIds_ ;
  std::auto_ptr<EcalElectronicsIdCollection>  * invalidMemGains_ ;
  std::auto_ptr<EcalPnDiodeDigiCollection>   * pnDiodeDigis_;

  EcalElectronicsMapper                  * electronicsMapper_;
  DCCEventBlock                            * currentEvent_;
  DCCEBEventBlock                        * ebEventBlock_;
  DCCEEEventBlock                        * eeEventBlock_;
		
};

#endif

