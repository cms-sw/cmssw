//-------------------------------------------------
//
//   Class: DTTracoCand
//
//   Description: Implementation of L1MuDTTracoChip
//                candidate
//
//
//   Author List:
//   C. Grandi
//   Modifications:
//   SV BTIC parameter from config
//   SV bti Trig pointer stored insted of trigdata
//   22/VI/04 SV: last trigger code update
//   04/XI/04 SV: bug fixed for wrong MB1 superlayer offset!
//   III/05   SV: NEWGEO update
//----------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTTraco/interface/DTTracoCand.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/DTBti/interface/DTBtiTrigData.h"
#include "L1Trigger/DTTraco/interface/DTTracoChip.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTraco.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>

//----------------
// Constructors --
//----------------
DTTracoCand::DTTracoCand(DTTracoChip *tc, const DTBtiTrigData *btitr, int pos, int step)
    : _traco(tc), _btitr(btitr), _step(step), _position(pos), _usable(1) {
  if (pos < 1 || pos > 4 * DTConfigTraco::NBTITC) {
    std::cout << "DTTracoCand::DTTracoCand: wrong position: " << pos;
    std::cout << ", dummy coordinates set!" << std::endl;
    _tcX = 9999;
    _tcK = 9999;
    return;
  }
  // abs value of K in local TRACO frame (angle conversion):
  // for sorting the angle closest to normal of chamber
  // Ktr = Kbti - BTIC - KRAD
  _tcK = abs(btitr->K() - tc->KRad() - tc->BTIC());

  // X in local TRACO frame (position conversion): Xtr = Xbti + BTIC*(i+4 or
  // o-4)
  int lstep = tc->BTIC();
  _tcX = btitr->X() + lstep * ((pos <= DTConfigTraco::NBTITC) * (pos - 1 + DTConfigTraco::NBTITC) +  // inner
                               (pos > DTConfigTraco::NBTITC) * (pos - 1 - DTConfigTraco::NBTITC));   // outer

  // NEWGEO add phi sl offset to inner positions
  if (btitr->btiSL() == 1)
    _tcX += tc->IBTIOFF();

  /* DEBUG
    btitr->print();
    std::cout << "K in local " << tc->number() << " TRACO " << K() << std::endl;
    std::cout << "X in local " << tc->number() << " TRACO " << X() << " offset "
    << tc->IBTIOFF() << std::endl; print();
  */

  /*
    //OBSOLETE
    //ATTENTION!! This isn't the "real" MB-superlayer shift
    //because wires have been renamed/shifted in : DTTrigGeom::cellMapping(int
    sl, int lay, int tube)
    //this is a "patch" : to BE FIXED with NEW GEOMETRY!

    //MB1: half cell shift
    if(btitr->btiSL()==1 && tc->station()==1)
      _tcX += (int)(0.5*lstep);
    //MB2
    //  if(btitr->btiSL()==1   && tc->station()==2)
    //    _tcX += (int)(-lstep);

    //std::cout << "X in local TRACO frame = " << _tcX << std::endl;
    //print();
  */
}

DTTracoCand::DTTracoCand(const DTTracoCand &tccand)
    : _traco(tccand._traco),
      _btitr(tccand._btitr),
      _step(tccand._step),
      _position(tccand._position),
      _usable(tccand._usable),
      _tcX(tccand._tcX),
      _tcK(tccand._tcK) {}

//--------------
// Destructor --
//--------------
DTTracoCand::~DTTracoCand() {}

//--------------
// Operations --
//--------------

DTTracoCand &DTTracoCand::operator=(const DTTracoCand &tccand) {
  if (this != &tccand) {
    _traco = tccand._traco;
    _btitr = tccand._btitr;
    _position = tccand._position;
    _step = tccand._step;
    _usable = tccand._usable;
    _tcX = tccand._tcX;
    _tcK = tccand._tcK;
  }
  return *this;
}

void DTTracoCand::print() const {
  //  int sl = _btitr->btiSL();
  std::cout << " step " << _step;
  std::cout << " Position " << _position;
  std::cout << " Code = " << _btitr->code();
  std::cout << " SL = " << _btitr->btiSL();
  std::cout << " N = " << _btitr->btiNumber();
  std::cout << " X = " << _btitr->X();
  std::cout << " K = " << _btitr->K();
  std::cout << " Kr = " << _traco->KRad();
  std::cout << " |K-Kr| = " << _tcK << std::endl;
}
