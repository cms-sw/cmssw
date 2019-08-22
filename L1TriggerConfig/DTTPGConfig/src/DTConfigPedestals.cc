//-------------------------------------------------
//
//   Class: DTConfigPedestals
//
//   Description: Take the time pedestals for trigger emulation
//
//
//   Author List:
//   			C.Battilana, M.Meneghelli
//
//   Modifications:
//
//-----------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigPedestals.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//----------------
// Constructors --
//----------------

DTConfigPedestals::DTConfigPedestals() : my_debug(false), my_useT0(false), my_tpgParams(nullptr), my_t0i(nullptr) {
  //my_tpgParams = 0;       // CB check that it could be done  outside this
  //my_t0i = 0;
}

//--------------
// Destructor --
//--------------

DTConfigPedestals::~DTConfigPedestals() {}

//--------------
// Operations --
//--------------

void DTConfigPedestals::setES(DTTPGParameters const *tpgParams, DTT0 const *t0Params) {
  my_tpgParams = tpgParams;

  if (useT0())
    my_t0i = t0Params;
}

float DTConfigPedestals::getOffset(const DTWireId &wire) const {
  int nc = 0;
  float ph = 0.;

  //float coarse = my_tpgParams.totalTime(wire.chamberId(),DTTimeUnits::ns); // CB ask for this to be fixed
  my_tpgParams->get(wire.chamberId(), nc, ph, DTTimeUnits::ns);
  float pedestal = 25. * nc + ph;

  float t0mean = 0.;
  float t0rms = 0.;

  if (useT0()) {
    my_t0i->get(wire, t0mean, t0rms, DTTimeUnits::ns);
    pedestal += t0mean;
  }

  if (debug()) {
    std::cout << "DTConfigPedestals::getOffset :" << std::endl;
    std::cout << "\t# of counts (BX): " << nc << " fine corr (ns) : " << ph << std::endl;
    std::cout << "\tt0i subtraction : ";
    if (useT0()) {
      std::cout << "enabled. t0i for wire " << wire << " : " << t0mean << std::endl;
    } else {
      std::cout << "disabled" << std::endl;
    }
  }

  return pedestal;
}

void DTConfigPedestals::print() const {
  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*                             DT ConfigPedestals                             *" << std::endl;
  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*                                                                            *" << std::endl;
  std::cout << "Debug flag : " << debug() << std::endl;
  std::cout << "Use t0i flag : " << useT0() << std::endl;

  for (int wh = -2; wh <= 2; ++wh) {
    for (int sec = 1; sec <= 14; ++sec) {
      for (int st = 1; st <= 4; ++st) {
        if (sec > 12 && st != 4)
          continue;

        int ncount = 0;
        float fine = 0.;
        DTChamberId chId = DTChamberId(wh, st, sec);
        my_tpgParams->get(chId, ncount, fine, DTTimeUnits::ns);

        std::cout << chId << "\t# counts (BX) : " << ncount << "\tfine adj : " << fine << std::endl;
      }
    }
  }

  std::cout << "******************************************************************************" << std::endl;
}
