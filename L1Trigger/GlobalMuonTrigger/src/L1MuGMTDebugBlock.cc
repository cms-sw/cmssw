//-------------------------------------------------
//
//   Class: L1MuGMTDebugBlock
//
//
//   $Date: 2013/04/22 17:03:01 $
//   $Revision: 1.7 $
//
//   Author :
//   H. Sakulin                HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTDebugBlock.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

//----------------
// Constructors --
//----------------

L1MuGMTDebugBlock::L1MuGMTDebugBlock(int minbx, int maxbx) :
  _minbx(minbx), _maxbx(maxbx), _bx(_minbx),
  _prophi(maxbx-minbx+1, std::vector<float>(32,0)),
  _proeta(maxbx-minbx+1, std::vector<float>(32,0)),
  _phisel(maxbx-minbx+1, std::vector<unsigned>(32,0)),
  _etasel(maxbx-minbx+1, std::vector<unsigned>(32,0)),
  _isMIPISO(maxbx-minbx+1, std::vector<unsigned>(32,0)),
  _pairMatrices(maxbx-minbx+1, std::vector<L1MuGMTMatrix<bool> >(NumMatrices, L1MuGMTMatrix<bool> (4,4))),
  _mqMatrices(maxbx-minbx+1, std::vector<L1MuGMTMatrix<int> >(NumMatrices, L1MuGMTMatrix<int> (4,4))),
  _cancelbits(maxbx-minbx+1, std::vector<unsigned>(4)),
  _brlmuons(maxbx-minbx+1, std::vector<L1MuGMTExtendedCand>(4)),
  _fwdmuons(maxbx-minbx+1, std::vector<L1MuGMTExtendedCand>(4))
  // will not work w/o copy constructor  
{
  if (maxbx < minbx) edm::LogWarning("BxRangeMismatch") << "*** error in L1MuGMTDebugBlock::L1MuGMTDebugBlock(): minbx > maxbx" << endl; 
  reset(); 
}


//--------------
// Destructor --
//--------------

L1MuGMTDebugBlock::~L1MuGMTDebugBlock() {
  for (int bx=0; bx<=(_maxbx-_minbx); bx++) {
    _prophi[bx].clear();
    _proeta[bx].clear();
    _phisel[bx].clear();
    _etasel[bx].clear();
    _isMIPISO[bx].clear();
    _brlmuons[bx].clear();
    _fwdmuons[bx].clear();
  }
  _prophi.clear();
  _proeta.clear();
  _phisel.clear();
  _etasel.clear();
  _isMIPISO.clear();
  _pairMatrices.clear();
  _mqMatrices.clear();
  _brlmuons.clear();
  _fwdmuons.clear();
}

void L1MuGMTDebugBlock::SetCancelBits (int idx, const std::vector<bool>& mine, const vector<bool>& others) {
  unsigned bits = 0;
  unsigned mask = 1;
  
  for (int i=0;i<4;i++) {
    if (mine[i]) bits |= mask;
    mask = mask << 1;
  }
  for (int i=0;i<4;i++) {
    if (others[i]) bits |= mask;
    mask = mask << 1;
  }
  _cancelbits[_bx - _minbx][idx] = bits;
  
}

//--------------
// Operations --
//--------------

void L1MuGMTDebugBlock::reset () {       
  _bx = _minbx;
  for (int bx=0; bx<_maxbx-_minbx+1; bx++) {
    for (int i=0;i<32;i++) {
      _prophi[bx][i]=_proeta[bx][i]=99.;
      _phisel[bx][i]=_etasel[bx][i]=0;
      _isMIPISO[bx][i]=0;
    }
    for (int i=0; i<NumMatrices; i++) {
      _pairMatrices[bx][i].init(0);
      _mqMatrices[bx][i].init(0);
    }
    for (int i=0; i<4; i++) {
      _brlmuons[bx][i].reset();
      _fwdmuons[bx][i].reset();
    }
  }
}











