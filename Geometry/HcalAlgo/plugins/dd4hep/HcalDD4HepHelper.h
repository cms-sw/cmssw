#ifndef HcalAlgo_HcalDD4HepHelper_h
#define HcalAlgo_HcalDD4HepHelper_h
#include "DD4hep/DD4hepUnits.h"

namespace HcalDD4HepHelper {
  const double convert2mm(double length) { return (length / dd4hep::mm); }
}  // namespace HcalDD4HepHelper
#endif
