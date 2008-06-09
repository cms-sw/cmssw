#include "EventFilter/RPCRawToDigi/interface/DataRecord.h"

using namespace rpcrawtodigi;

DataRecord::DataRecord()
{
 theData = (controlWordFlag << RECORD_TYPE_SHIFT);
 theData |= (EmptyOrDCCDiscardedFlag << CONTROL_TYPE_SHIFT);
 theData |= (EmptyWordFlag << EMPTY_OR_DCCDISCARDED_SHIFT);

}

DataRecord::recordName  DataRecord::type() const 
{
  recordName wordType = UndefinedType;
  if ( (int)((theData >> RECORD_TYPE_SHIFT) & RECORD_TYPE_MASK) <= MaxLBFlag) wordType = LinkBoardData;
  if ( (int)((theData >> RECORD_TYPE_SHIFT) & RECORD_TYPE_MASK) == controlWordFlag) {
    if ( (int)((theData >> BX_TYPE_SHIFT) & BX_TYPE_MASK) == BXFlag) wordType = StartOfBXData;
    if ( (int)((theData >> CONTROL_TYPE_SHIFT) & CONTROL_TYPE_MASK) == StartOfLBInputDataFlag) wordType =  StartOfTbLinkInputNumberData;
    if ( (int) ((theData >> CONTROL_TYPE_SHIFT) & CONTROL_TYPE_MASK) == RMBDiscardedDataFlag  ) wordType = RMBDiscarded;
    if ( (int) ((theData >> CONTROL_TYPE_SHIFT) & CONTROL_TYPE_MASK) == RMBCorruptedDataFlag  ) wordType = RMBCorrupted;
    // Empty or DCC Discarded
      if ( (int)((theData >> CONTROL_TYPE_SHIFT) & CONTROL_TYPE_MASK) == EmptyOrDCCDiscardedFlag){

        if ( (int)((theData >>  EMPTY_OR_DCCDISCARDED_SHIFT) & EMPTY_OR_DCCDISCARDED_MASK) == EmptyWordFlag) wordType = EmptyWord;
        if ( (int) ((theData >> EMPTY_OR_DCCDISCARDED_SHIFT) & EMPTY_OR_DCCDISCARDED_MASK) == DCCDiscardedFlag) wordType = DCCDiscarded;
        if ( (int) ((theData >> RMB_DISABLED_SHIFT) & RMB_DISABLED_MASK) == RMBDisabledDataFlag) wordType = RMBDisabled;
      }
  }
  return wordType;
}
