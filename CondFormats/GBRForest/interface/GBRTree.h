
#ifndef EGAMMAOBJECTS_GBRTree
#define EGAMMAOBJECTS_GBRTree

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

// The decision tree is implemented here as a set of two arrays, one for
// intermediate nodes, containing the variable index and cut value, as well
// as the indices of the 'left' and 'right' daughter nodes.  Positive indices
// indicate further intermediate nodes, whereas negative indices indicate
// terminal nodes, which are stored simply as a vector of regression responses

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>

class GBRTree {
public:
  GBRTree() {}
  explicit GBRTree(int nIntermediate, int nTerminal);

  double GetResponse(const float *vector) const;

  std::vector<float> &Responses() { return fResponses; }
  const std::vector<float> &Responses() const { return fResponses; }

  std::vector<unsigned char> &CutIndices() { return fCutIndices; }
  const std::vector<unsigned char> &CutIndices() const { return fCutIndices; }

  std::vector<float> &CutVals() { return fCutVals; }
  const std::vector<float> &CutVals() const { return fCutVals; }

  std::vector<int> &LeftIndices() { return fLeftIndices; }
  const std::vector<int> &LeftIndices() const { return fLeftIndices; }

  std::vector<int> &RightIndices() { return fRightIndices; }
  const std::vector<int> &RightIndices() const { return fRightIndices; }

protected:
  std::vector<unsigned char> fCutIndices;
  std::vector<float> fCutVals;
  std::vector<int> fLeftIndices;
  std::vector<int> fRightIndices;
  std::vector<float> fResponses;

  COND_SERIALIZABLE;
};

//_______________________________________________________________________
inline double GBRTree::GetResponse(const float *vector) const {
  int index = 0;
  do {
    auto r = fRightIndices[index];
    auto l = fLeftIndices[index];
    index = vector[fCutIndices[index]] > fCutVals[index] ? r : l;
  } while (index > 0);
  return fResponses[-index];
}

#endif
