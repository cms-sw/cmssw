#include "RecoTracker/MeasurementDet/plugins/TkStripMeasurementDet.h"
#include "RecoTracker/MeasurementDet/plugins/TkPixelMeasurementDet.h"
#include "RecoTracker/MeasurementDet/plugins/TkGluedMeasurementDet.h"

#include<iostream>


int main() {

  std::cout << "size of TkStripMeasurementDet " << sizeof(TkStripMeasurementDet) << std::endl;
  std::cout << "size of TkPixelMeasurementDet "<< sizeof(TkPixelMeasurementDet) << std::endl;
  std::cout << "size of TkGluedMeasurementDet "<< sizeof(TkGluedMeasurementDet) << std::endl;

  return 0;


}
