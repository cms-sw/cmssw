#ifndef MUON_ETA_RANGE_H
#define MUON_ETA_RANGE_H

//Ported form ORCA 

class MuonEtaRange {

  public:

    MuonEtaRange();
    MuonEtaRange(float max, float min);
    MuonEtaRange(const MuonEtaRange&);
    ~MuonEtaRange() {}
    MuonEtaRange& operator=(const MuonEtaRange&);
    inline float min() const { return theMin; }
    inline float max() const { return theMax; }
    bool isInside(float eta, float error=0.) const;
    bool isInside(const MuonEtaRange& range) const;
    bool isCompatible(const MuonEtaRange& range) const;
    MuonEtaRange add(const MuonEtaRange&) const;
    MuonEtaRange minRange(const MuonEtaRange&) const;
    MuonEtaRange subtract(const MuonEtaRange&) const;
  private:

    float theMin;
    float theMax;
};
#include <iostream>
inline std::ostream& operator<<(std::ostream& os, const MuonEtaRange& range)
{
  os << "(" << range.min() << " : " << range.max() << ")" ;
  return os;
}

#endif 
