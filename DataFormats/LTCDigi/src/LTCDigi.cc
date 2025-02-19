// $Id: LTCDigi.cc,v 1.4 2011/08/25 20:18:06 wmtan Exp $
#include "DataFormats/LTCDigi/interface/LTCDigi.h"


LTCDigi::LTCDigi(const unsigned char *data)
{
  // six 64 bit words
  cms_uint64_t *ld = (cms_uint64_t*)data;

  trigType_   = (ld[0]>>56)&        0xFULL; // 4 bits

  eventID_     = (ld[0]>>32)&0x00FFFFFFULL; // 24 bits
  runNumber_   = (ld[2]>>32)&0xFFFFFFFFULL; // 32 bits
  eventNumber_ = (ld[2])    &0xFFFFFFFFULL; // 32 bits
  
  sourceID_      = (ld[0]>> 8)&0x00000FFFULL; // 12 bits
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
cms_uint32_t LTCDigi::GetEventNumberFromBuffer(const unsigned char *data) 
{
  // six 64 bit words
  cms_uint64_t *ld = (cms_uint64_t*)data;
  cms_uint32_t eventNumber = (ld[2])    &0xFFFFFFFFULL; // 32 bits
  return eventNumber;
}
//static
cms_uint32_t LTCDigi::GetRunNumberFromBuffer(const unsigned char *data) 
{
  // six 64 bit words
  cms_uint64_t *ld = (cms_uint64_t*)data;
  cms_uint32_t runNumber   = (ld[2]>>32)&0xFFFFFFFFULL; // 32 bits
  return runNumber;
}


std::ostream & 
operator<<(std::ostream & stream, const LTCDigi & myDigi)
{
   stream << "----------------------------------------"<< std::endl;
   stream << "Dumping LTC digi. " << std::endl;
   stream << "Source ID: " << myDigi.sourceID() << std::endl;
   stream << "Run, event: " << myDigi.runNumber()
	  << ", " << myDigi.eventNumber () << std::endl;
   stream << "N_Inhibit:" << myDigi.triggerInhibitNumber() << std::endl;
   stream << LTCDigi::utcTime(myDigi.bstGpsTime()) << std::endl;
   stream << LTCDigi::locTime(myDigi.bstGpsTime()) << std::endl;
   ///
   stream << "Partition: " << myDigi.daqPartition() << std::endl;
   stream << "Bunch #:   " << myDigi.bunchNumber()  << std::endl;
   stream << "Orbit #:   " << myDigi.orbitNumber()  << std::endl;

   // Trigger information
   stream << "Trigger Bits(0-5):" ;
   for (int i = 0; i < 6; ++i ) {
      if ( myDigi.HasTriggered(i) )
	 stream << "1";
      else 
	 stream << "0";
      stream << " ";
   }
   stream << std::endl;

   //
   stream << "Ram trigger: " << myDigi.ramTrigger() << std::endl;
   stream << "VME trigger: " << myDigi.vmeTrigger() << std::endl;

   stream << "++++++++++++++++++++++++++++++++++++++++"<< std::endl;
   
   stream << "Raw Data" << std::endl;
   stream << "Trigger Input status: 0x" 
	  << std::hex << myDigi.triggerInputStatus() << std::endl;
   stream << "GPS time:             0x"
	  << std::hex << myDigi.bstGpsTime() << std::endl;

   stream << "----------------------------------------"<< std::endl;
   stream << std::dec << std::endl;

   return stream;
}


std::string LTCDigi::utcTime(cms_uint64_t t) //const
{
  // note that gmtime isn't reentrant and we don't own the data
  time_t tsmall = t/1000000;
  tm *utct = gmtime(&tsmall);
  std::string tstr("UTC: ");
  tstr += asctime(utct);
  tstr.replace(tstr.find("\n"),tstr.size(), "");
  return tstr;
}

std::string LTCDigi::locTime(cms_uint64_t t) //const
{
  time_t tsmall = t/1000000;
  std::string a("LOC: ");
  a+= std::string(ctime(&tsmall));
  a.replace(a.find("\n"),a.size(), "");
  return a;
}




