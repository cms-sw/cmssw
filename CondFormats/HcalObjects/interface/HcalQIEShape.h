#ifndef HcalQIEShape_h
#define HcalQIEShape_h

/** 
\class HcalQIEData
\author Fedor Ratnikov (UMd)
POOL object to store QIE basic shape
$Author: ratnikov
$Date: 2005/10/18 23:34:56 $
$Revision: 1.1 $
*/

#include <vector>
#include <algorithm>

// 128 QIE channels
class HcalQIEShape {
 public:
  HcalQIEShape();
  ~HcalQIEShape();
  float lowEdge (unsigned fAdc) const;
  float highEdge (unsigned fAdc) const;
  bool setLowEdge (float fValue, unsigned fAdc);
  bool setLowEdges (const float fValue [32]);
 protected:
  std::vector<float> mValues;
};

#endif
