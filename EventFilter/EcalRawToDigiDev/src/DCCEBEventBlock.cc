
#include "EventFilter/EcalRawToDigiDev/interface/DCCEBEventBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCDataUnpacker.h"
#include "EventFilter/EcalRawToDigiDev/interface/EcalDCCHeaderRuntypeDecoder.h"
#include "EventFilter/EcalRawToDigiDev/interface/EcalElectronicsMapper.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCTowerBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCEBTCCBlock.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCEBSRPBlock.h"
#include <sys/time.h>

#include <iomanip>
#include <sstream>


DCCEBEventBlock::DCCEBEventBlock( DCCDataUnpacker * u, EcalElectronicsMapper * m , bool hU, bool srpU, bool tccU, bool feU , bool memU) : 
  DCCEventBlock(u,m,hU,srpU,tccU,feU,memU)
{

  //Builds a tower unpacker block
  towerBlock_ = new DCCTowerBlock(u,m,this,feUnpacking_); 
  
  //Builds a srp unpacker block
  srpBlock_   = new DCCEBSRPBlock(u,m,this,srpUnpacking_);
  
  //Builds a tcc unpacker block
  tccBlock_   = new DCCEBTCCBlock(u,m,this,tccUnpacking_);
  
 
}




 // Unpack TCC blocks
int DCCEBEventBlock::unpackTCCBlocks(){


    if(tccChStatus_[0] != CH_TIMEOUT && tccChStatus_[0] != CH_DISABLED)
      return tccBlock_->unpack(&data_,&dwToEnd_);
	else return BLOCK_UNPACKED;


}

