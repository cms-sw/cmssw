//-------------------------------------------------
//
//   Class: DTTSCand.cpp
//
//   Description: A Trigger Server Candidate
//
//
//   Author List:
//   C. Grandi
//   Modifications: 
//   S. Marcellini, D. Bonacorsi
//   04/01/2007 : C. Battilana local config update
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTTriggerServerPhi/interface/DTTSCand.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTSPhi.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>

//----------------

// Constructors --
//----------------

DTTSCand::DTTSCand(DTTSS* tss, const DTTracoTrigData* tctrig, 
			   int ifs, int pos) 
  : _tss(tss), _tctrig(tctrig), _tcPos(pos), _isCarry(0) {

   _dataword.one();              // reset dataword to 0x1ff

   // SM sector collector Set bit 14 instead of 8, for 1st/2nd track to allow extra space 
   //      if(ifs==1)_dataword.unset(8); // set bit 8 (0=first, 1=second tracks)
      if(ifs==1)_dataword.unset(14); // set bit 14 (0=first, 1=second tracks)

}

DTTSCand::DTTSCand(const DTTSCand& tscand) 
  : _tss(tscand._tss), _tctrig(tscand._tctrig), _tcPos(tscand._tcPos), 
  _isCarry(tscand._isCarry) {

 
}


DTTSCand::DTTSCand() {}

//--------------
// Destructor --
//--------------
DTTSCand::~DTTSCand(){
}


//--------------
// Operations --
//--------------

DTTSCand& 
DTTSCand::operator=(const DTTSCand& tscand) {
  if(this != &tscand){
    _tss = tscand._tss;
    _tctrig = tscand._tctrig;
    _tcPos = tscand._tcPos;
    _isCarry = tscand._isCarry;
  }
  return *this;
}

void
DTTSCand::clear()  { 
  _tctrig=0; 
  _dataword.one();

  _isCarry=0;

}

void
DTTSCand::setBitsTss() {
  // first/second track already set. Set other 3 bits
  int itk=_dataword.element(14); // first tracks 0, second tracks 1
 
   clearBits();
      if(_tctrig->pvK()>32|| _tctrig->pvK()<0){ // Check K within 5 bits range
     std::cout << "DTTSCand::setBitsTss() pvK outside valid range: " << _tctrig->pvK();
     std::cout << " deltaPsiR set to 31" << std::endl;
   }
   else { 
     // assign preview in dataword (common to any other assignment)
        _dataword.assign(0,5,_tctrig->pvK());
     //  _dataword.assign(0,5,0);
     
     int posH=-1;
     int posI=-1;
     int posC=-1;
     switch(config()->TssMasking(itk)) {
     case 123: // H/L, In/Out, Corr/NotC
       posH = 7;
       posI = 6;
       posC = 5;
       break;
     case 132: // H/L, Corr/NotC, In/Out
       posH = 7;
       posI = 5;
       posC = 6;
       break;
     case 213: // In/Out, H/L, Corr/NotC 
       posH = 6;
       posI = 7;
       posC = 5;
       break;
     case 231: // In/Out, Corr/NotC, H/L 
       posH = 5;
       posI = 7;
       posC = 6;
       break;
     case 312: // Corr/NotC, H/L, In/Out
       posH = 6;
       posI = 5;
       posC = 7;
       break;
     case 321: // Corr/NotC, In/Out, H/L
       posH = 5;
       posI = 6;
       posC = 7;
       break;
     default:
       std::cout << "DTTSCand::DTTSCand(): masking not correct: ";
       std::cout << config()->TssMasking(itk);
       std::cout << " All bits set to 1" << std::endl;
     }
     // Masking:
     bool enaH = config()->TssHtrigEna(itk);
     bool enaI = config()->TssInOutEna(itk);
     bool enaC = config()->TssCorrEna(itk) ;
     if(isCarry()) {
       // Special setting for carry
       enaH = config()->TssHtrigEnaCarry();
       enaI = config()->TssInOutEnaCarry();
       enaC = config()->TssCorrEnaCarry() ;
     }
     // Bits set to 0 give higher priority:
     if(isHtrig()&&enaH&&posH>0)_dataword.unset(posH);
     if(isInner()&&enaI&&posI>0)_dataword.unset(posI);
     if(isCorr() &&enaC&&posC>0)_dataword.unset(posC);
   }
          
}


void
DTTSCand::setBitsBkmod() {
  // first/second track already set. Set other 4 bits (1 for null, 3 for a2, a1, a0)
  clearBitsBkmod();
  //  std::cout << " clearbits in TSM bk mode " << _dataword.print() << std::endl;

  int a2  = 6;
  int a1  = 5;
  int a0  = 4;
  //
  // std::cout << " _tctrig->qdec(): " << _tctrig->qdec() << std::endl;
  if( _tctrig->qdec()==6 ) { _dataword.unset(a2); _dataword.unset(a1); _dataword.unset(a0); }  // 1-000
  if( _tctrig->qdec()==5 ) { _dataword.unset(a2); _dataword.unset(a1); }                         // 1-001
  if( _tctrig->qdec()==4 ) { _dataword.unset(a2); _dataword.unset(a0); }                         // 1-010
  if( _tctrig->qdec()==3 ) { _dataword.unset(a1); }                                                // 1-101
  if( _tctrig->qdec()==2 ) { _dataword.unset(a1); _dataword.unset(a0); }                         // 1-100

  if( _tctrig->qdec()==0 ) { _dataword.unset(a0); }                                                // 1-110

  //    std::cout << " set Bits TSM back up " << _dataword.print() << std::endl;    
}


void
DTTSCand::setBitsTsm() {
  // first/second track already set. Set other 3 bits
   int itk=_dataword.element(14); // first tracks 0, second tracks 1

  clearBits();

  if(_tctrig->pvK()>31|| _tctrig->pvK()<0){ // Check K within 5 bits range
    std::cout << "DTTSCand::setBitsTsm pvK outside valid range: " << _tctrig->pvK();
    std::cout << " deltaPsiR set to 31" << std::endl;
  }
  else {
    // SM double TSM
    // assign preview in dataword (common to any other assignment)
     _dataword.assign(0,5,_tctrig->pvK());
     //  _dataword.assign(0,5,0);
    // 
    
    int posH=-1;
    int posI=-1;
    int posC=-1;
    switch(config()->TsmMasking(itk)) {
    case 123: // H/L, In/Out, Corr/NotC
      posH = 7;
      posI = 6;
      posC = 5;
      break;
    case 132: // H/L, Corr/NotC, In/Out
      posH = 7;
      posI = 5;
      posC = 6;
      break;
    case 213: // In/Out, H/L, Corr/NotC 
      posH = 6;
      posI = 7;
      posC = 5;
      break;
    case 231: // In/Out, Corr/NotC, H/L 
      posH = 5;
      posI = 7;
      posC = 6;
      break;
    case 312: // Corr/NotC, H/L, In/Out
      posH = 6;
      posI = 5;
      posC = 7;
      break;
    case 321: // Corr/NotC, In/Out, H/L
      posH = 5;
      posI = 6;
      posC = 7;
      break;
    default:
      std::cout << "DTTSCand::DTTSCand(): masking not correct: ";
      std::cout << config()->TssMasking(itk);
      std::cout << " All bits set to 1" << std::endl;
      
    // Masking:
      bool enaH = config()->TsmHtrigEna(itk);
      bool enaI = config()->TsmInOutEna(itk);
      bool enaC = config()->TsmCorrEna(itk) ;
      if(isCarry()) {
	// Special setting for carry
	enaH = config()->TsmHtrigEnaCarry();
	enaI = config()->TsmInOutEnaCarry();
	enaC = config()->TsmCorrEnaCarry() ;
      }
      // Bits set to 0 give higher priority:
      if(isHtrig()&&enaH&&posH>0)_dataword.unset(posH);
      if(isInner()&&enaI&&posI>0)_dataword.unset(posI);
      if(isCorr() &&enaC&&posC>0)_dataword.unset(posC);
            
    }
  }


}
void 
DTTSCand::print() const {
  std::cout << " First=" << isFirst();
  std::cout << " HTRIG=" << isHtrig();
  std::cout << " Inner=" << isInner();
  std::cout << " Corr="  << isCorr();
  std::cout << " Kpv="   << tracoTr()->pvK();
  std::cout << " dataword=";
  _dataword.print();
  std::cout << std::endl;
}

