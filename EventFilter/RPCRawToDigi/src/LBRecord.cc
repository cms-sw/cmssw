#include "EventFilter/RPCRawToDigi/interface/LBRecord.h"

#include <vector>

using namespace rpcrawtodigi;
using namespace std;

LBRecord::LBRecord(const RPCLinkBoardData & lbData) : DataRecord(0)
{
  theData = 0;

  int eod = lbData.eod();
  theData |= (eod<<EOD_SHIFT );

  int halfP = lbData.halfP();
  theData |= (halfP<<HALFP_SHIFT);

  int partitionNumber = lbData.partitionNumber();
  theData |= (partitionNumber<<PARTITION_NUMBER_SHIFT);

  int lbNumber = lbData.lbNumber();
  theData |= (lbNumber<<LB_SHIFT);

  std::vector<int> bitsOn = lbData.bitsOn();
  int partitionData = 0;
  for (vector<int>::const_iterator iv = bitsOn.begin(); iv != bitsOn.end(); iv++ ) {
    int ibit = (partitionNumber)? (*iv)%(partitionNumber*BITS_PER_PARTITION) : (*iv);
    partitionData |= (1<<ibit);
  }
  theData |= (partitionData<<PARTITION_DATA_SHIFT);
}

LBRecord::LBRecord(RecordType lbData) : DataRecord(0) { theData = lbData; }

RPCLinkBoardData LBRecord::lbData() const
{
   
  int partitionData= (theData>>PARTITION_DATA_SHIFT)&PARTITION_DATA_MASK;
  int halfP = (theData >> HALFP_SHIFT ) & HALFP_MASK;
  int eod = (theData >> EOD_SHIFT ) & EOD_MASK;
  int partitionNumber = (theData >> PARTITION_NUMBER_SHIFT ) & PARTITION_NUMBER_MASK;
  int lbNumber = (theData >> LB_SHIFT ) & LB_MASK ;

  std::vector<int> bits; bits.clear();
  for(int bb=0; bb<8;++bb) {
    if(partitionNumber>11){continue;} //Temporasry FIX. Very dirty. AK
    if ((partitionData>>bb)& 0X1) bits.push_back( partitionNumber* BITS_PER_PARTITION + bb);
  }
   
  return RPCLinkBoardData (bits,halfP,eod,partitionNumber,lbNumber);
}
