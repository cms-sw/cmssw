// Based on TAMU comparator digi fitter
//     L1Trigger/CSCTriggerPrimitives/src/CSCComparatorDigiFitter.h
//     L1Trigger/CSCTriggerPrimitives/src/CSCComparatorDigiFitter.cc

#ifndef L1TMuonEndCap_EMTFCSCComparatorDigiFitter_h_experimental
#define L1TMuonEndCap_EMTFCSCComparatorDigiFitter_h_experimental

#include "DataFormats/CSCDigi/interface/CSCComparatorDigi.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"

#include <Math/Functions.h>
#include <Math/SVector.h>
#include <Math/SMatrix.h>

#include <stdexcept>
#include <vector>
#include <string>


namespace experimental {

class EMTFCSCComparatorDigiFitter {
public:
  // Constructor
  EMTFCSCComparatorDigiFitter();

  // Destructor
  ~EMTFCSCComparatorDigiFitter();

  typedef CSCComparatorDigi CompDigi;

  struct FitResult {
    FitResult() : position(0.), slope(0.), chi2(100.), ndof(4) {}
    float position;  // local position at layer 3
    float slope;     // slope
    float chi2;      // chi2
    int ndof;        // degress of freedom
  };

  // For doing least square fit
  typedef ROOT::Math::SMatrix<double,2> SMatrix22;
  typedef ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > SMatrixSym2;
  typedef ROOT::Math::SVector<double,2> SVector2;

  // Fit comp digis
  FitResult fit(const std::vector<std::vector<CompDigi> >& compDigisAllLayers, const std::vector<int>& stagger, int keyStrip) const;

  // Least square fit with local x & y coordinates
  FitResult fitlsq(const std::vector<float>& x, const std::vector<float>& y) const;

  // A custom exception class used in making combinations
  class StopIteration : public std::exception {
  public:
    explicit StopIteration(const std::string& what_arg) {}
  };

  // Make combinations
  std::vector<std::vector<int> > make_combinations(const std::vector<std::vector<CompDigi> >& compDigisAllLayers) const;

private:
  static constexpr unsigned int min_nhits  = 3;
  static constexpr unsigned int max_ncombs = 10;
  static constexpr float        max_dx     = 2.;
};

}  // namespace experimental

#endif
