//FAMOS headers
#include "FastSimulation/Utilities/interface/Looses.h"

#include <iomanip>
#include <iostream>

Looses* Looses::myself = 0;

Looses::Looses() {}

Looses* Looses::instance() {
  if (!myself) myself = new Looses();
  return myself;
}

Looses::~Looses() { summary(); }

void 
Looses::count(const std::string& name, unsigned cut) { 
  
  if ( theLosses.find(name) == theLosses.end() ) { 

    std::vector<unsigned> myCounts;
    for ( unsigned i=0; i<20; ++i ) myCounts.push_back(0);
    theLosses[name] = myCounts;

  } 

  if ( cut < 20 ) ++theLosses[name][cut];

}

void 
Looses::summary() { 

  std::map< std::string, std::vector<unsigned> >::const_iterator lossItr;
  std::cout << "***** From LOOSES ***** : Cuts effects" << std::endl << std::endl;

  for ( lossItr=theLosses.begin(); lossItr != theLosses.end(); ++lossItr ) {
    std::cout << lossItr->first << ":" << std::endl;
    for ( unsigned i=0; i<4; ++i ) {
      for ( unsigned j=0; j<5; ++j ) {
	std::cout << std::setw(8) << lossItr->second[5*i+j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
     
}

