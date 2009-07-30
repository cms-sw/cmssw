
#include "EventFilter/AutoBU/interface/TriggerGeneratorWithPayload.h"
#include "EventFilter/Utilities/interface/DebugUtils.h"


#include <interface/evb/i2oEVBMsgs.h>
#include <toolbox/mem/Reference.h>

#include <interface/shared/frl_header.h>
#include <interface/shared/fed_trailer.h>

using namespace evf;

// bitToSet encoding -1=patterns; [0,63]=set ONE specific bit in physics algo 1, [64,127]=set one specific in physics algo 2)
// bitPatterns: use p1 for odd and p2 for even events for each 64-bit set in alternation
toolbox::mem::Reference *TriggerGeneratorWithPayload::generate( toolbox::mem::MemoryPoolFactory *poolFactory,
								toolbox::mem::Pool              *pool,
								I2O_TID                   initiatorAddress,
								I2O_TID                   targetAddress,
								uint32_t                  triggerSourceId,
								U32                       eventNumber,
								U32                       eventType,
								l1cond                   *toSet,
								uint32_t                  orbit) 
{
  using namespace evf::evtn;
  toolbox::mem::Reference            *ref_retval = 0;
  
  I2O_MESSAGE_FRAME                  *stdMsg     = 0;
  I2O_PRIVATE_MESSAGE_FRAME          *pvtMsg     = 0;
  I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *block      = 0;
  char                               *payload    = 0;
 
  size_t payloadSize                             = toSet->recordScheme - sizeof(fedh_t) - sizeof(fedt_t); //size of evm block from GT, used to decide the bst scheme and no of bxs
  ref_retval = Base::generate(poolFactory,pool,initiatorAddress,targetAddress,triggerSourceId,eventNumber,payloadSize);

  stdMsg = (I2O_MESSAGE_FRAME*)ref_retval->getDataLocation();
  pvtMsg = (I2O_PRIVATE_MESSAGE_FRAME*)stdMsg;
  block  = (I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*)stdMsg;
  
  payload = ((char*)block) + sizeof(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME) + sizeof(frlh_t);

  //set offsets based on record scheme 
  evm_board_setformat(toSet->recordScheme);

  

  fillPayload(payload, eventNumber, eventType, orbit, toSet);
  
  return ref_retval;
}
 void TriggerGeneratorWithPayload::fillPayload(char *ptr, uint32_t eventNumber, uint32_t evty, uint32_t orbit, l1cond *toSet){
  using namespace evf::evtn;

  uint32_t lsn = orbit/0x00100000 + 1;
  uint32_t bx = FAKE_FIXED_BX; // don't know yet how to set bx meaningfully;
  ptr += sizeof(fedh_t);

  // get current time to set bst gps field
  timeval tv;
  gettimeofday(&tv,0);
  //board id
  char *pptr = ptr + EVM_BOARDID_OFFSET * SLINK_HALFWORD_SIZE;
  *((uint32_t*)(pptr)) = (EVM_BOARDID_VALUE << EVM_BOARDID_SHIFT);
  //gps time
  pptr = ptr + EVM_GTFE_BSTGPS_OFFSET * SLINK_HALFWORD_SIZE;
  *((uint32_t*)(pptr)) = tv.tv_usec;
  pptr += SLINK_HALFWORD_SIZE;
  *((uint32_t*)(pptr)) = tv.tv_sec;
  //event number
  pptr = ptr + (EVM_GTFE_BLOCK*2 + EVM_TCS_TRIGNR_OFFSET) * SLINK_HALFWORD_SIZE;
  *((uint32_t*)(pptr)) = eventNumber;
  //orbit number
  pptr = ptr + (EVM_GTFE_BLOCK*2 + EVM_TCS_ORBTNR_OFFSET) * SLINK_HALFWORD_SIZE;
  *((uint32_t*)(pptr)) = orbit;
  //lumi section
  pptr = ptr + (EVM_GTFE_BLOCK*2 + EVM_TCS_LSBLNR_OFFSET) * SLINK_HALFWORD_SIZE;
  *((uint32_t*)(pptr)) = lsn + ((evty << EVM_TCS_EVNTYP_SHIFT) & EVM_TCS_EVNTYP_MASK);
  // bunch crossing in fdl bx+0 (-1,0,1) for nbx=3 i.e. offset by one full FDB block and leave -1 alone (it will be full of zeros)
  pptr = ptr + ((EVM_GTFE_BLOCK + EVM_TCS_BLOCK + EVM_FDL_BLOCK)*2+EVM_FDL_BCNRIN_OFFSET) * SLINK_HALFWORD_SIZE;
  *((uint32_t*)(pptr)) = bx & EVM_TCS_BCNRIN_MASK;
  // tech trig 64-bit set

  pptr = ptr + ((EVM_GTFE_BLOCK + EVM_TCS_BLOCK + EVM_FDL_BLOCK)*2 + EVM_FDL_TECTRG_OFFSET) * SLINK_HALFWORD_SIZE;
  *((tbits*)(pptr)) = toSet->t;
//   pptr = ptr + ((EVM_GTFE_BLOCK + EVM_TCS_BLOCK + EVM_FDL_BLOCK)*2 + EVM_FDL_ALGOB1_OFFSET) * SLINK_HALFWORD_SIZE;
//   *((uint64_t*)(pptr)) = a.ta1;
//   pptr = ptr + ((EVM_GTFE_BLOCK + EVM_TCS_BLOCK + EVM_FDL_BLOCK)*2 + EVM_FDL_ALGOB2_OFFSET) * SLINK_HALFWORD_SIZE;
//   *((uint64_t*)(pptr)) = a.ta2;
  
}
