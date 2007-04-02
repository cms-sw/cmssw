#ifndef DCCDATAUNPACKER_HH
#define DCCDATAUNPACKER_HH


/*
 *\ Class DCCDataUnpacker
 *
 * This class takes care of unpacking ECAL's raw data info
 *
 * \file DCCDataUnpacker.h
 *
 * $Date: 2007/03/28 00:43:16 $
 * $Revision: 1.1.2.2 $
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

#include "ECALUnpackerException.h"      
#include "DCCEventBlock.h"

#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalDigi/interface/EcalPnDiodeDigi.h>

#include <DataFormats/EcalDetId/interface/EcalDetIdCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>


using namespace std;

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

  void setEBDigisCollection( auto_ptr<EBDigiCollection>                         * x )
  { ebDigis_                = x; } 
 
  void setEEDigisCollection( auto_ptr<EEDigiCollection>                         * x )
  { eeDigis_                = x; } 
 
  void setDccHeadersCollection( auto_ptr<EcalRawDataCollection>                 * x )
  { dccHeaders_             = x; }
 
  void setEBSrFlagsCollection( auto_ptr<EBSrFlagCollection>                     * x )
  { ebSrFlags_              = x; } 
  
  void setEESrFlagsCollection( auto_ptr<EESrFlagCollection>                     * x )
  { eeSrFlags_              = x; }
 
  void setEBTpsCollection(auto_ptr<EcalTrigPrimDigiCollection>                  * x )
  { ebTps_                  = x; }
  
  void setEETpsCollection(auto_ptr<EcalTrigPrimDigiCollection>                  * x )
  { eeTps_                  = x; }
 
  void setInvalidGainsCollection(auto_ptr<EBDetIdCollection>                    * x )
  { invalidGains_           = x; }
 
  void setInvalidGainsSwitchCollection(auto_ptr<EBDetIdCollection>              * x )
  { invalidGainsSwitch_     = x; }
 
  void setInvalidGainsSwitchStayCollection(auto_ptr<EBDetIdCollection>          * x )
  { invalidGainsSwitchStay_ = x; }
 
  void setInvalidChIdsCollection(auto_ptr<EBDetIdCollection>                    * x )
  { invalidChIds_           = x; }
 
  void setInvalidTTIdsCollection(auto_ptr<EcalTrigTowerDetIdCollection>         * x )
  { invalidTTIds_           = x; }
 
  void setInvalidBlockLengthsCollection(auto_ptr<EcalTrigTowerDetIdCollection>  * x )
  { invalidBlockLengths_    = x; }
 
  void setPnDiodeDigisCollection(auto_ptr<EcalPnDiodeDigiCollection>            * x )
  { pnDiodeDigis_           = x; }
 
  void setInvalidMemTtIdsCollection( auto_ptr<EcalElectronicsIdCollection>      * x )
  { invalidMemTtIds_        = x; }
 
  void setInvalidMemBlockSizesCollection( auto_ptr<EcalElectronicsIdCollection> * x )
  { invalidMemBlockSizes_   = x; }
 
  void setInvalidMemChIdsCollection( auto_ptr<EcalElectronicsIdCollection>      * x )
  { invalidMemChIds_        = x; }
 
  void setInvalidMemGainsCollection( auto_ptr<EcalElectronicsIdCollection>      * x )
  { invalidMemGains_        = x; }
  
 
  /**
   Get the collection pointers
  */
  
  auto_ptr<EBDigiCollection>             * ebDigisCollection()
  { return ebDigis_;               }
  
  auto_ptr<EEDigiCollection>             * eeDigisCollection()
  { return eeDigis_;               }
  
  auto_ptr<EcalTrigPrimDigiCollection>   * ebTpsCollection()
  { return ebTps_;                 } 
  
  auto_ptr<EcalTrigPrimDigiCollection>   * eeTpsCollection()
  { return eeTps_;                 } 
  
  auto_ptr<EBSrFlagCollection>           * ebSrFlagsCollection()
  { return ebSrFlags_;             }  
  
  auto_ptr<EESrFlagCollection>           * eeSrFlagsCollection()
  { return eeSrFlags_;             } 
  
  auto_ptr<EcalRawDataCollection>        * dccHeadersCollection()
  { return dccHeaders_;            }
  
  auto_ptr<EBDetIdCollection>            * invalidGainsCollection()
  { return invalidGains_;          }
  
  auto_ptr<EBDetIdCollection>            * invalidGainsSwitchCollection()
  { return invalidGainsSwitch_;    }
  
  auto_ptr<EBDetIdCollection>            * invalidGainsSwitchStayCollection()
  { return invalidGainsSwitchStay_;}
  
  auto_ptr<EBDetIdCollection>            * invalidChIdsCollection()
  { return invalidChIds_;          }
      
  auto_ptr<EcalTrigTowerDetIdCollection> * invalidTTIdsCollection()
  { return invalidTTIds_;          }  
  
  auto_ptr<EcalTrigTowerDetIdCollection> * invalidBlockLengthsCollection()
  { return invalidBlockLengths_;   }
     
  auto_ptr<EcalElectronicsIdCollection>  * invalidMemTtIdsCollection()
  { return invalidMemTtIds_;       }
 
  auto_ptr<EcalElectronicsIdCollection>  * invalidMemBlockSizesCollection()
  { return invalidMemBlockSizes_;  }
  
  auto_ptr<EcalElectronicsIdCollection>  * invalidMemChIdsCollection()
  { return invalidMemChIds_;       }
  
  auto_ptr<EcalElectronicsIdCollection>  * invalidMemGainsCollection()
  { return invalidMemGains_;       }

  auto_ptr<EcalPnDiodeDigiCollection>    * pnDiodeDigisCollection()
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
  auto_ptr<EBDigiCollection>             * ebDigis_;
  auto_ptr<EEDigiCollection>             * eeDigis_;
  auto_ptr<EcalTrigPrimDigiCollection >  * ebTps_;
  auto_ptr<EcalTrigPrimDigiCollection >  * eeTps_;
  auto_ptr<EcalRawDataCollection>        * dccHeaders_;
  auto_ptr<EBDetIdCollection>            * invalidGains_;
  auto_ptr<EBDetIdCollection>            * invalidGainsSwitch_;
  auto_ptr<EBDetIdCollection>            * invalidGainsSwitchStay_;
  auto_ptr<EBDetIdCollection>            * invalidChIds_;
  auto_ptr<EBSrFlagCollection>           * ebSrFlags_;
  auto_ptr<EESrFlagCollection>           * eeSrFlags_;
  auto_ptr<EcalTrigTowerDetIdCollection> * invalidTTIds_;
  auto_ptr<EcalTrigTowerDetIdCollection> * invalidBlockLengths_; 
  
  
  auto_ptr<EcalElectronicsIdCollection>  * invalidMemTtIds_ ;
  auto_ptr<EcalElectronicsIdCollection>  * invalidMemBlockSizes_ ;
  auto_ptr<EcalElectronicsIdCollection>  * invalidMemChIds_ ;
  auto_ptr<EcalElectronicsIdCollection>  * invalidMemGains_ ;
  auto_ptr<EcalPnDiodeDigiCollection>    * pnDiodeDigis_;

  EcalElectronicsMapper                  * electronicsMapper_;
  DCCEventBlock                          * currentEvent_;
  DCCEBEventBlock                        * ebEventBlock_;
  DCCEEEventBlock                        * eeEventBlock_;
		
};

#endif

