#ifndef RecoTBCalo_EcalTBHodoscopeReconstructor_EcalTBHodoscopeRecInfoAlgo_HH
#define RecoTBCalo_EcalTBHodoscopeReconstructor_EcalTBHodoscopeRecInfoAlgo_HH

#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRawInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRecInfo.h"
#include "Geometry/EcalTestBeam/interface/EcalTBHodoscopeGeometry.h"

#include <vector>
#include <cmath>

class EcalTBHodoscopeRecInfoAlgo {
public:
  EcalTBHodoscopeRecInfoAlgo();

  explicit EcalTBHodoscopeRecInfoAlgo(int fitMethod,
                                      const std::vector<double>& planeShift,
                                      const std::vector<double>& zPosition);

  EcalTBHodoscopeRecInfo reconstruct(const EcalTBHodoscopeRawInfo& hodoscopeRawInfo) const;

private:
  //! Class to hold track information
  class BeamTrack {
  public:
    float x;
    float xS;
    float xQ;

    bool operator<(BeamTrack& b2) { return (fabs(xS) < fabs(b2.xS)); }

    BeamTrack(float x0, float xs, float xq) : x(x0), xS(xs), xQ(xq) {}

    ~BeamTrack() {}

  private:
    BeamTrack() {}
  };

  //Methods taken from h4ana. They can change in a future version

  void clusterPos(float& x, float& xQuality, const int& ipl, const int& xclus, const int& wclus) const;

  void fitHodo(float& x,
               float& xQuality,
               const int& ipl,
               const int& nclus,
               const std::vector<int>& xclus,
               const std::vector<int>& wclus) const;

  void fitLine(float& x,
               float& xSlope,
               float& xQuality,
               const int& ipl1,
               const int& nclus1,
               const std::vector<int>& xclus1,
               const std::vector<int>& wclus1,
               const int& ipl2,
               const int& nclus2,
               const std::vector<int>& xclus2,
               const std::vector<int>& wclus2) const;

  int fitMethod_;

  std::vector<double> planeShift_;
  std::vector<double> zPosition_;

  //for the moment mantaining it here
  EcalTBHodoscopeGeometry myGeometry_;
};

#endif
