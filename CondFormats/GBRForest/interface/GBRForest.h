#ifndef EGAMMAOBJECTS_GBRForest
#define EGAMMAOBJECTS_GBRForest

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// GBRForest                                                            //
//                                                                      //
// A fast minimal implementation of Gradient-Boosted Regression Trees   //
// which has been especially optimized for size on disk and in memory.  //
//                                                                      //
// Designed to be built from TMVA-trained trees, but could also be      //
// generalized to otherwise-trained trees, classification,              //
//  or other boosting methods in the future                             //
//                                                                      //
//  Josh Bendavid - MIT                                                 //
//////////////////////////////////////////////////////////////////////////

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/GBRForest/interface/GBRTree.h"

#include <cmath>
#include <vector>

class GBRForest {
public:
  GBRForest() {}

  double GetResponse(const float* vector) const;
  double GetGradBoostClassifier(const float* vector) const;
  double GetAdaBoostClassifier(const float* vector) const { return GetResponse(vector); }

  //for backwards-compatibility
  double GetClassifier(const float* vector) const { return GetGradBoostClassifier(vector); }

  void SetInitialResponse(double response) { fInitialResponse = response; }

  std::vector<GBRTree>& Trees() { return fTrees; }
  const std::vector<GBRTree>& Trees() const { return fTrees; }

protected:
  double fInitialResponse = 0.0;
  std::vector<GBRTree> fTrees;

  COND_SERIALIZABLE;
};

//_______________________________________________________________________
inline double GBRForest::GetResponse(const float* vector) const {
  double response = fInitialResponse;
  for (auto const& tree : fTrees) {
    response += tree.GetResponse(vector);
  }
  return response;
}

//_______________________________________________________________________
inline double GBRForest::GetGradBoostClassifier(const float* vector) const {
  double response = GetResponse(vector);
  return 2.0 / (1.0 + std::exp(-2.0 * response)) - 1;  //MVA output between -1 and 1
}

#endif
