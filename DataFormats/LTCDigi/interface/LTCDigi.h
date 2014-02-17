#ifndef DataFormats_LTCDigi_h
#define DataFormats_LTCDigi_h

#include <vector>
#include <ostream>
#include <string>

#include "FWCore/Utilities/interface/typedefs.h"

/** \class LTCDigi
 *  Data from the Local Trigger Controller (LTC)
 *
 * $Id: LTCDigi.h,v 1.5 2011/08/25 20:18:06 wmtan Exp $
 */


class LTCDigi
{
 public:
  // need a default constructor!
  LTCDigi() {}
  LTCDigi(const unsigned char* data);
  
  // STATIC functions
  // These are to allow the event builder to grab the run number 
  // from a butter it knows points at LTC data.
  static cms_uint32_t GetEventNumberFromBuffer(const unsigned char *databuffer);
  static cms_uint32_t  GetRunNumberFromBuffer(const unsigned char *databuffer);
  static std::string utcTime(cms_uint64_t t);
  static std::string locTime(cms_uint64_t t);

  // right now these are just silly but maybe if we pack the internals
  // then this won't seem as silly
  unsigned int eventNumber() const { return eventNumber_; };
  unsigned int eventID() const { return eventID_; };
  unsigned int runNumber() const   { return runNumber_;   };

  unsigned int bunchNumber() const { return bunchNumber_;};
  cms_uint32_t     orbitNumber() const { return orbitNumber_;};

  int version() const { return versionNumber_; } ;
  int sourceID() const { return sourceID_; };

  int daqPartition() const { return daqPartition_; };

  cms_uint32_t triggerInputStatus() const { return trigInputStat_; };
  cms_uint32_t triggerInhibitNumber() const { return trigInhibitNumber_; };

  cms_uint64_t bstGpsTime() const { return bstGpsTime_;};

  unsigned int bxMask() const { return ((triggerInputStatus()>>28)&0x1); } ;
  unsigned int vmeTrigger() const 
  { 
    return ((triggerInputStatus()>>27)&0x1); 
  } ;
  unsigned int ramTrigger() const 
  { 
    return ((triggerInputStatus()>>26)&0x1); 
  } ;
  unsigned char externTriggerMask() const // six bits
  { 
    return ((triggerInputStatus()>>20)&0x3FU); 
  } ;
  unsigned char cyclicTriggerMask() const // six bits
  { 
    return (triggerInputStatus()&0x3FU); 
  } ;
  
  
  
  bool HasTriggered( int i ) const {
    if ( i > 5 ) return false; // throw exception?
    return ((externTriggerMask()&(0x1U<<i))!=0);
  }

  

 private:
  // unpacked for now
  unsigned int trigType_; // 4 bits

  unsigned int eventID_; // 24 bits
  unsigned int runNumber_; // 32 bits

  unsigned int sourceID_;    // 12 bits - 0xCBB constant for LTC

  unsigned int bunchNumber_; // 12 bits
  cms_uint32_t     orbitNumber_; // 32 bits

  int          versionNumber_; // 8 bits - Slink data version number  

  int          daqPartition_;  // 4 bits
  
  cms_uint32_t     eventNumber_; // 32 bits
                               // same as event number up to resets
  cms_uint32_t     trigInputStat_; // 32 bits
  cms_uint32_t     trigInhibitNumber_; // 32 bits
  cms_uint64_t     bstGpsTime_; // 64 bits - is standard unix time in seconds


    
};

std::ostream & operator<<(std::ostream & stream, 
			  const LTCDigi & myDigi);

typedef std::vector<LTCDigi> LTCDigiCollection;


#endif  // DataFormats_LTCDigi_h
