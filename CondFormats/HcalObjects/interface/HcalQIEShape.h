#ifndef HcalQIEShape_h
#define HcalQIEShape_h

/** 
\class HcalQIEData
\author Fedor Ratnikov (UMd)
POOL object to store QIE basic shape
$Author: ratnikov
$Date: 2012/11/02 14:13:13 $
$Revision: 1.4 $
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
  unsigned range (unsigned fAdc) const {
    //6 bit mantissa in QIE10, 5 in QIE8
    return (nbins_ == 32) ? (fAdc >> 5) & 0x3 : (fAdc >> 6) & 0x3;
  }
  unsigned local (unsigned fAdc) const {
    unsigned tmp = nbins_ == 32 ? (fAdc & 0x1f) : (fAdc & 0x3f) ;
    return   tmp;
  }
  unsigned nbins() const { return nbins_; }

 protected:
 private:
  void expand ();
  bool setLowEdge (float fValue, unsigned fAdc);
  std::vector<float> mValues;
  unsigned int nbins_;
};

#endif
