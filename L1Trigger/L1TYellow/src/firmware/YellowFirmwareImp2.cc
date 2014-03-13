///
/// \class l1t::YellowFirmwareImp2
///
/// Description: Implementation 2 of firmware for the fictitious Yellow trigger.
///
/// Implementation:
///    Demonstrates how to implement firmware.
///
/// \author: Michael Mulhearn - UC Davis
///

// This example implemenents firmware version 3.

#include "YellowFirmwareImp.h"

using namespace std;
using namespace l1t;

YellowFirmwareImp2::YellowFirmwareImp2(const YellowParams & dbPars) : db(dbPars) {}

YellowFirmwareImp2::~YellowFirmwareImp2(){};

void YellowFirmwareImp2::processEvent(const YellowDigiCollection & input, YellowOutputCollection & out){
  YellowDigiCollection::const_iterator incell;
  YellowOutput outcell;
  
  for (incell = input.begin(); incell != input.end(); ++incell){
    // firmware version 3:  et(out) = A * et + C * yvar
    outcell.setEt( db.paramA() * incell->et() + db.paramC() * incell->yvar() );    
    // yvar(out) = yvar
    outcell.setYvar(incell->yvar());

    out.push_back(outcell);
  }

}
