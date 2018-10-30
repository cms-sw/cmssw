#ifndef Navigation_MTDEtaRange_H
#define Navigation_MTDEtaRange_H

/** \class MTDEtaRange
 *
 *  a class to define eta range used in Muon Navigation
 *
 *
 * \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 */

class MTDEtaRange {

  public:

    MTDEtaRange();
    MTDEtaRange(float max, float min);
    MTDEtaRange(const MTDEtaRange&);
    ~MTDEtaRange() {}
    MTDEtaRange& operator=(const MTDEtaRange&);
    inline float min() const { return theMin; }
    inline float max() const { return theMax; }
    bool isInside(float eta, float error=0.) const;
    bool isInside(const MTDEtaRange& range) const;
    bool isCompatible(const MTDEtaRange& range) const;
    MTDEtaRange add(const MTDEtaRange&) const;
    MTDEtaRange minRange(const MTDEtaRange&) const;
    MTDEtaRange subtract(const MTDEtaRange&) const;
  private:

    float theMin;
    float theMax;
};
#include <iostream>
inline std::ostream& operator<<(std::ostream& os, const MTDEtaRange& range)
{
  os << "(" << range.min() << " : " << range.max() << ")" ;
  return os;
}

#endif 
