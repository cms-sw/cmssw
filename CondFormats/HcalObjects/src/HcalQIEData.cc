#include "CondFormats/HcalObjects/interface/HcalQIEData.h"

void HcalQIEData::setupShape() {
  //qie8
  const float binMin [32] = {-1,  0,  1,  2,  3,  4,  5,  6,  7,  8,
			     9, 10, 11, 12, 13, 14, 16, 18, 20, 22,
			     24, 26, 28, 31, 34, 37, 40, 44, 48, 52,
			     57, 62};
  mShape[0].setLowEdges(32,binMin);


  //qie10
  const float binMin2 [64] = {-1,  0,  1,  2,  3,    4,  5,  6,  7,  8,    9, 10, 11, 12, 13,   14,  // 16*1
                             15, 17, 19, 21, 23,   25, 27, 29, 31, 33,   35, 37, 39, 41, 43,   45, 47, 49, 51, 53,//20*2
                             55, 59, 63, 67, 71,   75, 79, 83, 87, 91,   95, 99, 103,107,111, 115,119,123,127,131,  135,//21*4 
                             139, 147, 155, 163, 171, 179, 187};// 7*8
  mShape[1].setLowEdges(64,binMin2);
}
