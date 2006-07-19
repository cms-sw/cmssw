//-------------------------------------------------
//
//   Class: DTSectCollCand.cpp
//
//   Description: A Sextor Collector Candidate
//
//
//   Author List:
//   S.Marcellini D.Bonacorsi
//   Modifications: 
//
//
//--------------------------------------------------

//#include "Utilities/Configuration/interface/Architecture.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTSectorCollector/interface/DTSectCollCand.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/DTUtilities/interface/DTConfig.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>

//----------------
// Constructors --
//---------------- // SM double TSM 
DTSectCollCand::DTSectCollCand(DTSC* tsc, const DTChambPhSegm* tsmsegm, int ifs) 
  : _tsc(tsc), _tsmsegm(tsmsegm) {
  _dataword.one();              // reset dataword to 0x1ff
  if(ifs==1)_dataword.unset(14); // set bit 14 (0=first, 1=second tracks)
}


DTSectCollCand::DTSectCollCand() {
 
}

//--------------
// Destructor --
//--------------
DTSectCollCand::~DTSectCollCand(){
}


//--------------
// Operations --
//--------------

DTSectCollCand& 
DTSectCollCand::operator=(const DTSectCollCand& tsccand) {
  if(this != &tsccand){
    _tsc = tsccand._tsc;
    _tsmsegm = tsccand._tsmsegm;
  }
  return *this;
}

void
DTSectCollCand::clear()  { 
  _tsmsegm=0; 
  _dataword.one();
//   std::cout << " clear dataword : " << _dataword.print() << std::endl;
}

// SM double TSM: remove datawordbk and replace it with dataword
void
DTSectCollCand::setBitsSectColl() {

  clearBitsSectColl();
    
  if(abs(_tsmsegm->DeltaPsiR())>1024 ){ // Check phiB within 10 bits range
    std::cout << "DTSectCollCand::setBitsSectColl phiB outside valid range: " << _tsmsegm->DeltaPsiR();
    std::cout << " deltaPsiR set to 512" <<  std::endl;
  }
  else {
    // SM double TSM
    // assign preview in dataword (common to any other assignment)
    _dataword.assign(0,10,abs(_tsmsegm->DeltaPsiR()));
    // 
    int a2  = 12;
    int a1  = 11;
    int a0  = 10;
   
    if( _tsmsegm->code()==6 ) { _dataword.unset(a2); _dataword.unset(a1); _dataword.unset(a0); }  // 1-000
    if( _tsmsegm->code()==5 ) { _dataword.unset(a2); _dataword.unset(a1); }                         // 1-001
    if( _tsmsegm->code()==4 ) { _dataword.unset(a2); _dataword.unset(a0); }                         // 1-010
    if( _tsmsegm->code()==3 ) { _dataword.unset(a1); }                                                // 1-101
    if( _tsmsegm->code()==2 ) { _dataword.unset(a1); _dataword.unset(a0); }                         // 1-100
    // if( _tsmsegm->code()==1 ) no unset needed => 111
    if( _tsmsegm->code()==0 ) { _dataword.unset(a0); }                                                // 1-110

  }
  
}


  void 
  DTSectCollCand::print() const {
    std::cout << "Sector Collector Candidate: " << std::endl; 
    if(_dataword.element(14)==0) {std::cout << "First track type" << std::endl;}
    else {std::cout << "Second track type" << std::endl;}
       std::cout << " code=" << _tsmsegm->pvCode();

        std::cout << " dataword=";
        _dataword.print();
    // SM double TSM remove datawordbk section

    std::cout << std::endl;
  }

