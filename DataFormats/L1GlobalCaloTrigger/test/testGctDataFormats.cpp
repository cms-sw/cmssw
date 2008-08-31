// test 

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHFBitCounts.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHFRingEtSums.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternHFData.h"


#include <iostream>

using namespace std;

int main() {

  // test HF bit counts set methods
  L1GctHFBitCounts b;
  unsigned c[4];
  for (unsigned i=0; i<4; ++i) c[i] = 0;
  for (unsigned i=0; i<4; ++i) {
    for (unsigned j=0; j<0x7; ++j) {
      b.setBitCount(i, j);
      c[i] = j;
      
      // check all counts are as expected
      for (unsigned k=0; k<4; ++k) {
	bool check = ( b.bitCount(k) == c[k] );
	if (!check) {
	  std::cout << "L1GctHFBitCounts failed : ";
	  std::cout << " bitCount(" << std::dec << k << ") = " << std::hex << b.bitCount(k);
	  std::cout << ", expected " << std::hex << c[k] << std::endl;
	  exit(1);
	}
      }
      
    }
  }

  // test HF ring sums set methods
  L1GctHFRingEtSums s;
  for (unsigned i=0; i<4; ++i) c[i] = 0;
  for (unsigned i=0; i<4; ++i) {
    for (unsigned j=0; j<0x7; ++j) {
      s.setEtSum(i, j);
      c[i] = j;
      
      // check all counts are as expected
      for (unsigned k=0; k<4; ++k) {
	bool check = ( s.etSum(k) == c[k] );
	if (!check) {
	  std::cout << "L1GctHFBitCounts failed : ";
	  std::cout << " bitCount(" << std::dec << k << ") = " << std::hex << s.etSum(k);
	  std::cout << ", expected " << std::hex << c[k] << std::endl;
	  exit(1);
	}
      }
      
    }
  }

  // test intern HF data
  L1GctInternHFData d;
  for (unsigned i=0; i<4; ++i) c[i] = 0;
  for (unsigned i=0; i<4; ++i) {
    for (unsigned j=0; j<0xff; ++j) {
      d.setValue(i, j);
      c[i] = j;
      
      // check all counts are as expected
      for (unsigned k=0; k<4; ++k) {
	bool check = ( d.value(k) == c[k] );
	if (!check) {
	  std::cout << "L1GctHFBitCounts failed : ";
	  std::cout << " bitCount(" << std::dec << k << ") = " << std::hex << d.value(k);
	  std::cout << ", expected " << std::hex << c[k] << std::endl;
	  exit(1);
	}
      }
      
    }
  }



}

