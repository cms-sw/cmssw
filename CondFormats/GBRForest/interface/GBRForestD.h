
#ifndef EGAMMAOBJECTS_GBRForestD
#define EGAMMAOBJECTS_GBRForestD

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// GBRForestD                                                           //
//                                                                      //
// A fast minimal implementation of Gradient-Boosted Regression Trees   //
// which has been especially optimized for size on disk and in memory.  //
//                                                                      //
// Designed to be built from the output of GBRLikelihood,               //
// but could also be  generalized to otherwise-trained trees            //
// classification, or other boosting methods in the future              //
//                                                                      //
//  Josh Bendavid - CERN                                                //
//////////////////////////////////////////////////////////////////////////

#include "CondFormats/Serialization/interface/Serializable.h"

#include "GBRTreeD.h"

#include <vector>

class GBRForestD {
public:
  typedef GBRTreeD TreeT;

  GBRForestD() {}
  template <typename InputForestT>
  GBRForestD(const InputForestT &forest);

  double GetResponse(const float *vector) const;

  double InitialResponse() const { return fInitialResponse; }
  void SetInitialResponse(double response) { fInitialResponse = response; }

  std::vector<GBRTreeD> &Trees() { return fTrees; }
  const std::vector<GBRTreeD> &Trees() const { return fTrees; }

protected:
  double fInitialResponse = 0.0;
  std::vector<GBRTreeD> fTrees;

  COND_SERIALIZABLE;
};

//_______________________________________________________________________
inline double GBRForestD::GetResponse(const float *vector) const {
  double response = fInitialResponse;
  for (std::vector<GBRTreeD>::const_iterator it = fTrees.begin(); it != fTrees.end(); ++it) {
    int termidx = it->TerminalIndex(vector);
    response += it->GetResponse(termidx);
  }
  return response;
}

//_______________________________________________________________________
template <typename InputForestT>
GBRForestD::GBRForestD(const InputForestT &forest) : fInitialResponse(forest.InitialResponse()) {
  //templated constructor to allow construction from Forest classes in GBRLikelihood
  //without creating an explicit dependency

  for (typename std::vector<typename InputForestT::TreeT>::const_iterator treeit = forest.Trees().begin();
       treeit != forest.Trees().end();
       ++treeit) {
    fTrees.push_back(GBRTreeD(*treeit));
  }
}

#endif
