/**
  * \file
  * A test for EEDetId::hashedIndex()
  *
  */

#include <iostream>
#include <string>
#include <stdexcept>

#include "DataFormats/EcalDetId/interface/EEDetId.h"

const int nBegin[EEDetId::IX_MAX] = { 41, 41, 41, 36, 36, 26, 26, 26, 21, 21, 21, 21, 21, 16, 16, 14, 14, 14, 14, 14, 9, 9, 9, 9, 9, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 9, 9, 9, 14, 14, 14, 14, 14, 16, 16, 21, 21, 21, 21, 21, 26, 26, 26, 36, 36, 41, 41, 41 };

int main(int argc, char* argv[]) {
  try {
	  for (int iz = -1; iz<2; iz+=2) {
		  for (int ix = EEDetId::IX_MIN; ix <= EEDetId::IX_MAX; ix++) {
			  for (int iy=nBegin[ix-1]; iy<=100-nBegin[ix-1]+1; iy++) {
				  if (EEDetId::validDetId(ix, iy, iz)) {
					  EEDetId id = EEDetId( ix, iy, iz );
					  EEDetId ud;
					  assert( ud.unhashIndex( id.hashedIndex() ) == id );
					  std::cout << id << " " << id.hashedIndex() << " " << ud.unhashIndex( id.hashedIndex() ) << std::endl;
				  } else {
					  std::cout << "Invalid detId " << ix << " " << iy << " " << iz << std::endl;
				  }
			  }
		  }
	  }
  } catch (std::exception &e) {
	  std::cerr << e.what();
  }
}
