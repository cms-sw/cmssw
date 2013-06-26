#ifndef HcalQIEShape_h
#define HcalQIEShape_h

/** 
\class HcalQIEData
\author Fedor Ratnikov (UMd)
POOL object to store QIE basic shape
$Author: ratnikov
$Date: 2013/03/26 11:31:11 $
$Revision: 1.5 $
*/

#include <vector>
#include <algorithm>

// N QIE channels
class HcalQIEShape {
 public:
  HcalQIEShape();
  ~HcalQIEShape();
  float lowEdge (unsigned fAdc) const;
  float highEdge (unsigned fAdc) const;
  float center (unsigned fAdc) const;
  bool setLowEdges (unsigned int nVals, const float *fValue);
  unsigned range (unsigned fAdc) const {return (fAdc >> 5) & 0x3;}
  unsigned local (unsigned fAdc) const {
    unsigned tmp = nbins_ == 32 ? (fAdc & 0x1f) : (fAdc & 0x3f) ;
    return   tmp;
  }

 protected:
 private:
  void expand ();
  bool setLowEdge (float fValue, unsigned fAdc);
  std::vector<float> mValues;
  unsigned int nbins_;
};

#endif
