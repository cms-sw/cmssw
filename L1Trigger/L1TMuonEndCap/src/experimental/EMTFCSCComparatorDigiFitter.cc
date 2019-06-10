#include "L1Trigger/L1TMuonEndCap/interface/experimental/EMTFCSCComparatorDigiFitter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <cassert>
#include <iostream>


namespace experimental {

EMTFCSCComparatorDigiFitter::EMTFCSCComparatorDigiFitter() {}

EMTFCSCComparatorDigiFitter::~EMTFCSCComparatorDigiFitter() {}

EMTFCSCComparatorDigiFitter::FitResult
EMTFCSCComparatorDigiFitter::fit(const std::vector<std::vector<CompDigi> >& compDigisAllLayers, const std::vector<int>& stagger, int keyStrip) const
{
  FitResult res;

  // Make combinations
  std::vector<std::vector<int> > combinations = make_combinations(compDigisAllLayers);

  // Shuffle
  std::random_shuffle(combinations.begin(), combinations.end());

  // Only fit up to 10 combinations
  if (combinations.size() > max_ncombs) {
    combinations.erase(combinations.begin() + max_ncombs, combinations.end());
  }

  std::vector<float> x;
  std::vector<float> y;

  // Loop over combinations
  std::vector<std::vector<int> >::const_iterator it  = combinations.begin();
  std::vector<std::vector<int> >::const_iterator end = combinations.end();

  for (; it != end; ++it) {
    // Prepare local x & y coordinates
    x.clear();
    y.clear();

    const std::vector<int>& combination = *it;
    for (unsigned i=0; i<combination.size(); ++i) {
      int ii = combination.at(i);
      if (compDigisAllLayers.at(i).size() > 0) { // protect against empty layer
        const CompDigi& compDigi = compDigisAllLayers.at(i).at(ii);
        x.push_back(compDigi.getHalfStrip() - keyStrip + stagger.at(i) - stagger.at(CSCConstants::KEY_CLCT_LAYER-1));
        y.push_back(i+1);
      }
    }

    // Fit
    const FitResult& tmp_res = fitlsq(x, y);
    if (res.chi2 > tmp_res.chi2) {  // minimize on chi2
      res = tmp_res;
    }

    if (res.chi2 == 0.0)  // already perfect
      break;
  }
  return res;
}

EMTFCSCComparatorDigiFitter::FitResult EMTFCSCComparatorDigiFitter::fitlsq(const std::vector<float>& x, const std::vector<float>& y) const
{
  FitResult res;

  if (x.size() < min_nhits) {  // not enough hits
    return res;
  }

  assert(x.size() == y.size());

  // Copied from RecoLocalMuon/GEMSegment/plugins/MuonSegFit.cc

  static SMatrix22 M;
  static SVector2 B;
  static SVector2 p;

  M *= 0;  // reset to 0
  B *= 0;  // reset to 0

  for (unsigned i=0; i<x.size(); ++i) {
    float e   = 1.0;
    float ee  = e*e;
    float x_i = x[i];
    float y_i = y[i];

    M(0,0) += ee;
    M(0,1) += ee * y_i;
    M(1,0) += ee * y_i;
    M(1,1) += ee * y_i * y_i;
    B(0)   += ee * x_i;
    B(1)   += ee * x_i * y_i;
    //std::cout << ".... " << x_i << " " << y_i << std::endl;
  }

  bool ok = M.Invert();
  if (!ok) {
    edm::LogWarning("EMTFCSCComparatorDigiFitter") <<  "fitlsq(): failed to invert matrix M.";
    return res;
  }

  p = M * B;

  float intercept    = p(0);
  float slope        = p(1);
  int   ndof         = x.size() - 2;
  float chi2         = 0.;
  bool  please_refit = false;

  for (unsigned i=0; i<x.size(); ++i) {
    float e   = 1.0;
    float ee  = e*e;
    float x_i = x[i];
    float y_i = y[i];

    float dx = (intercept + slope * y_i) - x_i;
    chi2 += ee * dx * dx;

    if (std::abs(dx) > max_dx)
      please_refit = true;  // detect outlier
  }
  //std::cout << "...... " << intercept << " " << slope << " " << chi2 << " " << ndof << std::endl;

  // Refit if necessary
  if (please_refit) {
    std::vector<float> x_refit;
    std::vector<float> y_refit;

    for (unsigned i=0; i<x.size(); ++i) {
      float x_i = x[i];
      float y_i = y[i];

      float dx = (intercept + slope * y_i) - x_i;
      if (std::abs(dx) > max_dx)  continue;  // detect outlier

      x_refit.push_back(x_i);
      y_refit.push_back(y_i);
    }

    if (x_refit.size() < min_nhits)  // not enough hits
      please_refit = false;

    assert(x_refit.size() == y_refit.size());

    if (please_refit) {
      M *= 0;  // reset to 0
      B *= 0;  // reset to 0

      for (unsigned i=0; i<x_refit.size(); ++i) {
        float e   = 1.0;
        float ee  = e*e;
        float x_i = x_refit[i];
        float y_i = y_refit[i];

        M(0,0) += ee;
        M(0,1) += ee * y_i;
        M(1,0) += ee * y_i;
        M(1,1) += ee * y_i * y_i;
        B(0)   += ee * x_i;
        B(1)   += ee * x_i * y_i;
        //std::cout << ".... " << x_i << " " << y_i << std::endl;
      }

      ok = M.Invert();
      if (!ok) {
        edm::LogWarning("EMTFCSCComparatorDigiFitter") <<  "fitlsq(): failed to invert matrix M.";
        return res;
      }

      p = M * B;

      intercept    = p(0);
      slope        = p(1);
      ndof         = x_refit.size() - 2;
      chi2         = 0.;

      for (unsigned i=0; i<x_refit.size(); ++i) {
        float e   = 1.0;
        float ee  = e*e;
        float x_i = x_refit[i];
        float y_i = y_refit[i];

        float dx = (intercept + slope * y_i) - x_i;
        chi2 += ee * dx * dx;
      }
    }
    //std::cout << "...... " << intercept << " " << slope << " " << chi2 << " " << ndof << std::endl;
  }  // end refit if necessary

  // Calculate deltaPhi a la ME0, chi2/ndof
  res.position = intercept + (slope * CSCConstants::KEY_CLCT_LAYER);
  res.slope = slope;
  res.chi2 = chi2;
  res.ndof = ndof;
  return res;
}

std::vector<std::vector<int> >
EMTFCSCComparatorDigiFitter::make_combinations(const std::vector<std::vector<CompDigi> >& compDigisAllLayers) const
{
  std::vector<std::vector<int> > combinations;  // combinations to output

  std::vector<int> combination(CSCConstants::NUM_LAYERS, 0);  // initial combination

  assert(compDigisAllLayers.size() == combination.size());

  try {
    while (true) {
      // A combination consists of 6 indices, one at every layer.
      // For each iteration, check each index starting from the last layer.
      // If the index can be incremented, increment it and reset the indices
      // in all the following layers.
      // Repeat until the index at the first layer cannot be incremented.
      for (int i=CSCConstants::NUM_LAYERS-1; i>=0; --i) {
        int j = combination.at(i);
        int len = compDigisAllLayers.at(i).size();
        if (len == 0)  len = 1;  // protect against empty layer

        if (j != len-1) {
          combinations.push_back(combination);
          combination.at(i) += 1;
          for (int ii=CSCConstants::NUM_LAYERS-1; ii>=i+1; --ii) {
            combination.at(ii) = 0;
          }
          break;
        } else if (i == 0) {
          combinations.push_back(combination);
          throw StopIteration("");
        }
      }
    }
  } catch (const StopIteration &e) {
    // Debug
    //for (unsigned i = 0; i < compDigisAllLayers.size(); ++i) {
    //  std::cout << "Layer " << i << ": ";
    //  for (unsigned j = 0; j < compDigisAllLayers.at(i).size(); ++j) {
    //    std::cout << compDigisAllLayers.at(i).at(j).getHalfStrip() << " ";
    //  }
    //  std::cout << std::endl;
    //}
    //for (unsigned i = 0; i < combinations.size(); ++i) {
    //  std::cout << "Comb " << i << ": ";
    //  for (unsigned j = 0; j < combinations.at(i).size(); ++j) {
    //    std::cout << combinations.at(i).at(j) << " ";
    //  }
    //}
    //std::cout << std::endl;
  }
  return combinations;
}

}  // namespace experimental
