#include "FWCore/MessageService/test/MemoryTestClient_A.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <iomanip>
#include <string>

namespace edmtest
{
 
int  MemoryTestClient_A::nevent = 0;

MemoryTestClient_A::MemoryTestClient_A( edm::ParameterSet const & ps)
  : memoryPattern(), vsize(0), last_allocation(0) 
{
  int pattern = ps.getUntrackedParameter<int>("pattern",1);
  edm::LogWarning("memoryPattern") << "Memory Pattern selected: " << pattern;
  initializeMemoryPattern(pattern);
}

void
  MemoryTestClient_A::analyze( edm::Event      const & /*unused*/
                           , edm::EventSetup const & /*unused*/
                              )
{
  nevent++;
  double v = memoryPattern[nevent%memoryPattern.size()];
  edm::LogVerbatim("memoryUsage") << "Event " << nevent 
  	<< " uses "<< v << " Mbytes";
  if ( v > vsize ) {
    int leaksize = static_cast<int>((v-vsize)*1048576);
    char* leak = new  char[leaksize];
    edm::LogPrint("memoryIncrease") << "Event " << nevent 
  	<< " increases vsize by "<< v-vsize << " Mbytes";
    vsize = v;
    last_allocation = leak;
  }
  // DO NOT delete[] leak; the point is to increment vsize!
  
}  // MessageLoggerClient::analyze()

void  MemoryTestClient_A::initializeMemoryPattern(int pattern) {
  switch(pattern) {
    case 1:		// A general pattern
	memoryPattern.push_back(   2.1   );
	memoryPattern.push_back(   3.1   );
	memoryPattern.push_back(   4.1   );
	memoryPattern.push_back(   2.1   );
	memoryPattern.push_back(   1.7   );
	memoryPattern.push_back(   8.4   );
	memoryPattern.push_back(   3.4   );
	memoryPattern.push_back(  43.1   );
	memoryPattern.push_back(  17.1   );
	memoryPattern.push_back(   2.1   );
	memoryPattern.push_back(  47.9   );
	memoryPattern.push_back(   8.3   );
	memoryPattern.push_back(  56.3   );
	memoryPattern.push_back(   1.1   );
	memoryPattern.push_back(  19.1   );
	memoryPattern.push_back(   2.1   );
	memoryPattern.push_back(  22.0   );
	memoryPattern.push_back(   9.1   );
	memoryPattern.push_back(   2.1   );
	memoryPattern.push_back(  57.9   );
	memoryPattern.push_back(   2.1   );
	memoryPattern.push_back(  59.5   );
	memoryPattern.push_back(   4.1   );
	memoryPattern.push_back(   6.1   );
	memoryPattern.push_back(  61.5   );
	memoryPattern.push_back(   4.2   );
	memoryPattern.push_back(   6.3   );
    break;
    case 2:		// Here, there is a swap with R1 insteadd of L1
	memoryPattern.push_back(   2.1   );
	memoryPattern.push_back(   3.1   );
	memoryPattern.push_back(   4.1   );
	memoryPattern.push_back(   2.1   );
	memoryPattern.push_back(   1.7   );
	memoryPattern.push_back(   8.4   );
	memoryPattern.push_back(   3.4   );
	memoryPattern.push_back(   2.1   );
	memoryPattern.push_back(   1.7   );
	memoryPattern.push_back(   3.4   );
	memoryPattern.push_back(  43.1   );
	memoryPattern.push_back(  17.1   );
	memoryPattern.push_back(   2.1   );
	memoryPattern.push_back(  47.9   );
	memoryPattern.push_back(   8.3   );
	memoryPattern.push_back(  56.3   );
	memoryPattern.push_back(   1.1   );
	memoryPattern.push_back(  69.1   );
	memoryPattern.push_back(   2.1   );
	memoryPattern.push_back(  22.0   );
	memoryPattern.push_back(   9.1   );
	memoryPattern.push_back(   2.1   );
	memoryPattern.push_back( 117.9   );
	memoryPattern.push_back(   2.1   );
	memoryPattern.push_back( 119.5   );
	memoryPattern.push_back(   4.1   );
	memoryPattern.push_back(   6.1   );
	memoryPattern.push_back( 120.8   );
	memoryPattern.push_back(  19.5   );
  break;
    case 3:		// Here, there are few increments
	memoryPattern.push_back(   2.1   );
	memoryPattern.push_back(   3.1   );
	memoryPattern.push_back(   4.1   );
	memoryPattern.push_back(   2.1   );
	memoryPattern.push_back(   1.7   );
	memoryPattern.push_back(   8.4   );
	memoryPattern.push_back(   3.4   );
	memoryPattern.push_back(   2.1   );
	memoryPattern.push_back(   1.7   );
	memoryPattern.push_back(   3.4   );
	memoryPattern.push_back(  43.1   );
	memoryPattern.push_back(  17.1   );
    break;
    default:
	memoryPattern.push_back(   2.1   );
	memoryPattern.push_back(   3.1   );
	memoryPattern.push_back(   4.1   );
	memoryPattern.push_back(   2.1   );
  }
}

}  // namespace edmtest


using edmtest::MemoryTestClient_A;
DEFINE_FWK_MODULE(MemoryTestClient_A);
