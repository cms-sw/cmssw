
#ifndef EGAMMAOBJECTS_GBRTree2D
#define EGAMMAOBJECTS_GBRTree2D

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

class GBRTree2D {
public:
  GBRTree2D() {}

  void GetResponse(const float *vector, double &x, double &y) const;
  int TerminalIndex(const float *vector) const;

  std::vector<float> &ResponsesX() { return fResponsesX; }
  const std::vector<float> &ResponsesX() const { return fResponsesX; }

  std::vector<float> &ResponsesY() { return fResponsesY; }
  const std::vector<float> &ResponsesY() const { return fResponsesY; }

  std::vector<unsigned short> &CutIndices() { return fCutIndices; }
  const std::vector<unsigned short> &CutIndices() const { return fCutIndices; }

  std::vector<float> &CutVals() { return fCutVals; }
  const std::vector<float> &CutVals() const { return fCutVals; }

  std::vector<int> &LeftIndices() { return fLeftIndices; }
  const std::vector<int> &LeftIndices() const { return fLeftIndices; }

  std::vector<int> &RightIndices() { return fRightIndices; }
  const std::vector<int> &RightIndices() const { return fRightIndices; }

protected:
  std::vector<unsigned short> fCutIndices;
  std::vector<float> fCutVals;
  std::vector<int> fLeftIndices;
  std::vector<int> fRightIndices;
  std::vector<float> fResponsesX;
  std::vector<float> fResponsesY;

  COND_SERIALIZABLE;
};

//_______________________________________________________________________
inline void GBRTree2D::GetResponse(const float *vector, double &x, double &y) const {
  int index = 0;

  unsigned short cutindex = fCutIndices[0];
  float cutval = fCutVals[0];

  while (true) {
    if (vector[cutindex] > cutval) {
      index = fRightIndices[index];
    } else {
      index = fLeftIndices[index];
    }

    if (index > 0) {
      cutindex = fCutIndices[index];
      cutval = fCutVals[index];
    } else {
      x = fResponsesX[-index];
      y = fResponsesY[-index];
      return;
    }
  }
}

//_______________________________________________________________________
inline int GBRTree2D::TerminalIndex(const float *vector) const {
  int index = 0;

  unsigned short cutindex = fCutIndices[0];
  float cutval = fCutVals[0];

  while (true) {
    if (vector[cutindex] > cutval) {
      index = fRightIndices[index];
    } else {
      index = fLeftIndices[index];
    }

    if (index > 0) {
      cutindex = fCutIndices[index];
      cutval = fCutVals[index];
    } else {
      return (-index);
    }
  }
}

#endif
