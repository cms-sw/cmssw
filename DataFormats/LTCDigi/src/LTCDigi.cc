#include "DataFormats/LTCDigi/interface/LTCDigi.h"


LTCDigi::LTCDigi(const unsigned char *data)
{
  // six 64 bit words
  uint64_t *ld = (uint64_t*)data;

  trigType_   = (ld[0]>>56)&       0xFUL; // 4 bits

  eventID_     = (ld[0]>>32)&0x00FFFFFFULL; // 24 bits
  runNumber_   = (ld[2]>>32)&0xFFFFFFFFULL; // 32 bits
  eventNumber_ = (ld[2])    &0xFFFFFFFFULL; // 32 bits
  
  sourceID_      = (ld[1]>> 8)&0x00000FFFULL; // 12 bits
  // this should always be 815?
  //assert(sourceID_ == 815);

  bunchNumber_   = (ld[0]>>20)&     0xFFFULL; // 12 bits
  orbitNumber_   = (ld[1]>>32)&0xFFFFFFFFULL; // 32 bits

  versionNumber_ = (ld[1]>>24)&0xFFULL;       // 8 bits
  
  daqPartition_  = (ld[1]    )&0xFULL;        // 4 bits


  trigInputStat_ = (ld[3]    )&0xFFFFFFFFULL; // 32 bits

  trigInhibitNumber_ = (ld[3]>>32)&0xFFFFFFFFULL; // 32 bits

  bstGpsTime_    = ld[4]; // 64 bits

}
//static
uint32_t LTCDigi::GetEventNumberFromBuffer(const unsigned char *data) 
{
  // six 64 bit words
  uint64_t *ld = (uint64_t*)data;
  uint32_t eventNumber = (ld[2])    &0xFFFFFFFFULL; // 32 bits
  return eventNumber;
}
//static
uint32_t LTCDigi::GetRunNumberFromBuffer(const unsigned char *data) 
{
  // six 64 bit words
  uint64_t *ld = (uint64_t*)data;
  uint32_t runNumber   = (ld[2]>>32)&0xFFFFFFFFULL; // 32 bits
  return runNumber;
}
