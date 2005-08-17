
//
// F.Ratnikov (UMd), Jul. 19, 2005
//

#include "CalibCalorimetry/HcalAlgos/interface/HCALClasses.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbServiceFrontier.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"


namespace {
  HcalDb::CellId frontierCell (const cms::HcalDetId& fCell) {
    return HcalDb::CellId (fCell.ietaAbs(), fCell.iphi(), fCell.depth(), fCell.zside());
  }
  int timestamp = 1;

  // temporaly
  const float binMin [33] = {-1,  0,  1,  2,  3,  4,  5,  6,  7,  8,
			      9, 10, 11, 12, 13, 14, 16, 18, 20, 22,
			     24, 26, 28, 31, 34, 37, 40, 44, 48, 52,
			     57, 62, 67};
  int range (int adc) {
    return (adc >> 5) & 3;
  }
}

HcalDbServiceFrontier::HcalDbServiceFrontier () {}
HcalDbServiceFrontier::~HcalDbServiceFrontier () {}

const char* HcalDbServiceFrontier::name () const {return "HcalDbServiceFontier";}

// basic conversion function for single range (0<=count<32)
double HcalDbServiceFrontier::adcShape (int fCount) const {
  return 0.5 * (binMin[fCount] + binMin[fCount+1]);
}
// bin size for the QIE conversion
double HcalDbServiceFrontier::adcShapeBin (int fCount) const {
  return binMin[fCount+1] - binMin[fCount];
}

  // pedestal  
double HcalDbServiceFrontier::pedestal (const cms::HcalDetId& fCell, int fCapId) const {
  double result = 0;
  const HcalDb::Pedestals* pedestals = HcalDb::getPedestals(timestamp);
  const HcalDb::Ped* pedestal = pedestals->getById (frontierCell (fCell));
  if (pedestal) result = pedestal->ped[0]; // capid not used for pedestal ?
  else {
    std::cerr << "HcalDbServiceFrontier::pedestal-> Can not find pedestal" 
	      << " for cell " << frontierCell (fCell) << ". Use default = 0" << std::endl;
  }
  return result;
}

  // pedestal width
double HcalDbServiceFrontier::pedestalError (const cms::HcalDetId& fCell, int fCapId) const {
  return 0.; // none
}

  // gain
double HcalDbServiceFrontier::gain (const cms::HcalDetId& fCell, int fCapId) const {
  double result = 1.;
  const HcalDb::Gains* gains = HcalDb::getGains(timestamp);
  const HcalDb::Gain* gain = gains->getById (frontierCell (fCell));
  if (gain) result = gain->gain;
  else {
    std::cerr << "HcalDbServiceFrontier::gain-> Can not find gain" 
	      << " for cell " << frontierCell (fCell) << ". Use default = 1" << std::endl;
  }
  return result;
}

  // gain width
double HcalDbServiceFrontier::gainError (const cms::HcalDetId& fCell, int fCapId) const {
  return 0.; // none
}

// offset for the (cell,capId,range)
double HcalDbServiceFrontier::offset (const cms::HcalDetId& fCell, int fCapId, int fRange) const {
  double result = 0;
  const HcalDb::Ranges* ranges = HcalDb::getRanges(timestamp);
  const HcalDb::Range* range = ranges->getById (frontierCell (fCell));
 if (range) result = range->getOffset (fCapId, fRange);
 else {
   std::cerr << "HcalDbServiceFrontier::offset-> Can not find range" 
	     << " for cell " << frontierCell (fCell) << ". Use default = 0" << std::endl;
 }
 return result;
}

// slope for the (cell,capId,range)
double HcalDbServiceFrontier::slope (const cms::HcalDetId& fCell, int fCapId, int fRange) const {
  double result = 1.;
  const HcalDb::Ranges* ranges = HcalDb::getRanges(timestamp);
  const HcalDb::Range* range = ranges->getById (frontierCell (fCell));
  if (range) result = range->getOffset (fCapId, fRange);
  else {
    std::cerr << "HcalDbServiceFrontier::slope-> Can not find range" 
	      << " for cell " << frontierCell (fCell) << ". Use default = 1" << std::endl;
  }
  return result;
}


HcalDbService* HcalDbServiceFrontier::clone () const {
  return (HcalDbService*) new HcalDbServiceFrontier ();
}

EVENTSETUP_DATA_REG(HcalDbServiceFrontier);
