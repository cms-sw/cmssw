#include "DataFormats/RPCDigi/interface/RecordCD.h"

#include <vector>

using namespace rpcrawtodigi;
using namespace std;

RecordCD::RecordCD(int lbInLink, int partitionNumber, int eod, int halfP, 
  const std::vector<int> & packedStrips) : DataRecord(0)
{
  theData = 0;

  theData |= (lbInLink <<CHAMBER_SHIFT);

  theData |= (partitionNumber<<PARTITION_NUMBER_SHIFT);

  theData |= (eod<<EOD_SHIFT );

  theData |= (halfP<<HALFP_SHIFT);

  int partitionData = 0;
  for (vector<int>::const_iterator iv = packedStrips.begin(); iv != packedStrips.end(); iv++ ) {
    int ibit = (partitionNumber) ? (*iv)%(partitionNumber*BITS_PER_PARTITION) : (*iv);
    partitionData |= (1<<ibit);
  }
  theData |= (partitionData<<PARTITION_DATA_SHIFT);
}

int RecordCD::lbInLink() const
{
  return (theData >> CHAMBER_SHIFT ) & CHAMBER_MASK ;
}

int RecordCD::partitionNumber() const
{
  return (theData >> PARTITION_NUMBER_SHIFT ) & PARTITION_NUMBER_MASK;
}

int RecordCD::eod() const
{
  return (theData >> EOD_SHIFT ) & EOD_MASK; 
}

int RecordCD::halfP() const
{
  return (theData >> HALFP_SHIFT ) & HALFP_MASK;
}

int RecordCD::partitionData() const
{
  return (theData>>PARTITION_DATA_SHIFT)&PARTITION_DATA_MASK; 
}

std::vector<int> RecordCD::packedStrips() const
{
  int partitionNumber = this->partitionNumber();
  int partitionData = this->partitionData();
  std::vector<int> strips;
  for (int ib=0; ib <8; ++ib) {
    if ((partitionData>>ib)& 1) strips.push_back( partitionNumber* BITS_PER_PARTITION + ib); 
  }
  return strips;
}

std::string RecordCD::print() const
{
  std::ostringstream str;
  str <<" DATA";  
  return str.str();
}
