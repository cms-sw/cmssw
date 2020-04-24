#include "CondFormats/HcalObjects/interface/HcalQIEData.h"

void HcalQIEData::setupShape() {
  //qie8
  const float binMin [32] = {-1,  0,  1,  2,  3,  4,  5,  6,  7,  8,
			     9, 10, 11, 12, 13, 14, 16, 18, 20, 22,
			     24, 26, 28, 31, 34, 37, 40, 44, 48, 52,
			     57, 62};
  mShape[0].setLowEdges(32,binMin);


  //qie10
  const float binMin2 [64] = {-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, // 16 bins with width 1x 
                             15.5, 17.5, 19.5, 21.5, 23.5, 25.5, 27.5, 29.5, 31.5, 33.5, 35.5, 37.5, 39.5, 41.5, 43.5, 45.5, 47.5, 49.5, 51.5, 53.5, // 20 bins with width 2x 
                             55.5, 59.5, 63.5, 67.5, 71.5, 75.5, 79.5, 83.5, 87.5, 91.5, 95.5, 99.5, 103.5, 107.5, 111.5, 115.5, 119.5, 123.5, 127.5, 131.5, 135.5, // 21 bins with width 4x 
                             139.5, 147.5, 155.5, 163.5, 171.5, 179.5, 187.5}; // 7 bins with width 8x
  mShape[1].setLowEdges(64,binMin2);
}
