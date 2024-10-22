#ifndef CastorQIEShape_h
#define CastorQIEShape_h

/** 
\class CastorQIEData
\author Panos Katsas (UoA)
POOL object to store QIE basic shape
*/

#include <vector>
#include <algorithm>

// 128 QIE channels
class CastorQIEShape {
public:
  CastorQIEShape();
  ~CastorQIEShape();
  float lowEdge(unsigned fAdc) const;
  float highEdge(unsigned fAdc) const;
  float center(unsigned fAdc) const;
  bool setLowEdges(const float fValue[32]);
  unsigned range(unsigned fAdc) const { return (fAdc >> 5) & 0x3; }
  unsigned local(unsigned fAdc) const { return fAdc & 0x1f; }

protected:
private:
  void expand();
  bool setLowEdge(float fValue, unsigned fAdc);
  float mValues[129];
};

#endif
