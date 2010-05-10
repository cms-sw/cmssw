#include "RecoTracker/MeasurementDet/interface/TkStripMeasurementDet.h"
#include "RecoTracker/MeasurementDet/interface/TkPixelMeasurementDet.h"
#include "RecoTracker/MeasurementDet/interface/TkGluedMeasurementDet.h"

#include<iostream>


int main() {

  std::cout << "size of TkStripMeasurementDet " << sizeof(TkStripMeasurementDet) << std::endl;
  std::cout << "size of TkPixelMeasurementDet "<< sizeof(TkPixelMeasurementDet) << std::endl;
  std::cout << "size of TkGluedMeasurementDet "<< sizeof(TkGluedMeasurementDet) << std::endl;

  return 0;


}
