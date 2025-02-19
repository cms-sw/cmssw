//-------------------------------------------------
//
//   Class: DTSectCollThCand.cpp
//
//   Description: A Sector Collector Theta Candidate
//
//
//   Author List:
//   C. Battilana
//   Modifications: 
//   
//
//
//--------------------------------------------------


//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTSectorCollector/interface/DTSectCollThCand.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>

//----------------
// Constructors -- 
//----------------
DTSectCollThCand::DTSectCollThCand(DTSC* tsc, const DTChambThSegm* tstsegm) 
  : _tsc(tsc), _tstsegm(tstsegm) {
}


DTSectCollThCand::DTSectCollThCand() {
 
}

//--------------
// Destructor --
//--------------
DTSectCollThCand::~DTSectCollThCand(){
}


//--------------
// Operations --
//--------------

DTSectCollThCand& 
DTSectCollThCand::operator=(const DTSectCollThCand& tsccand) {
  if(this != &tsccand){
    _tsc = tsccand._tsc;
    _tstsegm = tsccand._tstsegm;
  }
  return *this;
}

void
DTSectCollThCand::clear()  { 
  _tstsegm=0; 
}

int
DTSectCollThCand::CoarseSync() const{
  int stat= _tstsegm->ChamberId().station();
  if (stat>3){
    std::cout << "DTSectCollThCand::CoarseSync: station number outside valid range: " 
	      << stat << " 0 returned" << std::endl;
    return 0;
  }
    return config()->CoarseSync(stat);
}

  void 
  DTSectCollThCand::print() const {
    std::cout << "Sector Collector Theta Candidate: " << std::endl;
    _tstsegm->print();
    std::cout << "SC step: " << _tstsegm->step()+CoarseSync();
    std::cout << std::endl;

  }

