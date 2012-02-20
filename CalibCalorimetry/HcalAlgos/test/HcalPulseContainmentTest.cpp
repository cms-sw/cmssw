#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseContainmentManager.h"
#include <cassert>
#include <iostream>
int main() {
  float fixedphase_ns = 6.0;
  float max_fracerror = 0.02;
  std::auto_ptr<HcalPulseContainmentManager> manager;
  manager = std::auto_ptr<HcalPulseContainmentManager>( new HcalPulseContainmentManager(max_fracerror));

  HcalDetId hb1(HcalBarrel, 1, 1, 1);
  HcalDetId he1(HcalEndcap, 17, 1, 1);
  double fc = 10.;
  // test re-finding the correction
  double corr1 = manager->correction(hb1, 4, fixedphase_ns, fc);
  double corr2 = manager->correction(hb1, 4, fixedphase_ns, fc);
  assert(corr1 == corr2);
  // fewer toAdd means bigger correction
  double corr3 = manager->correction(hb1, 2, fixedphase_ns, fc);
  assert(corr3 > corr1);
  // HB and HE have the same shape here
  double corr4 = manager->correction(he1, 4, fixedphase_ns, fc);
  assert(corr4 == corr1);
  std::cout << corr1 << " " <<corr2 << " " <<corr3 << " " <<corr4 << " " << std::endl;
  return 0;
}

