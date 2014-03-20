///
/// \class l1t::YellowFirmwareImp1
///
/// Description: Implementation 1 of firmware for the fictitious Yellow trigger.
///
/// Implementation:
///    Demonstrates how to implement firmware.
///
/// \author: Michael Mulhearn - UC Davis
///

// This example implemenents firmware version 1 and 2.

#include "YellowFirmwareImp.h"

using namespace std;
using namespace l1t;

YellowFirmwareImp1::YellowFirmwareImp1(const YellowParams & dbPars) : db(dbPars) {}

YellowFirmwareImp1::~YellowFirmwareImp1(){};

void YellowFirmwareImp1::processEvent(const YellowDigiCollection & input, YellowOutputCollection & out){
  YellowDigiCollection::const_iterator incell;
  YellowOutput outcell;

  for (incell = input.begin(); incell != input.end(); ++incell){

    if (db.firmwareVersion() == 1) {  
      // firmware version 1:  et(out) = A * et + B
      outcell.setEt( db.paramA() * incell->et() + db.paramB() );
    } else { 
      // firmware version 2:  et(out) = A * et + B * yvar
      outcell.setEt( db.paramA() * incell->et() + db.paramB() * incell->yvar() );
    }
    // both version yvar(out) = yvar
    outcell.setYvar(incell->yvar());

    out.push_back(outcell);

  }
    
  
}
