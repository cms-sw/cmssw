
#ifndef EGAMMAOBJECTS_GBRForest2D
#define EGAMMAOBJECTS_GBRForest2D

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// GBRForest2D                                                            //
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

#include <vector>
#include "GBRTree2D.h"
#include <stdio.h>

  class GBRForest2D {

    public:

       GBRForest2D();
       ~GBRForest2D() {}
       
       void GetResponse(const float* vector, double &x, double &y) const;
      
       void SetInitialResponse(double x, double y) { fInitialResponseX = x; fInitialResponseY = y; }
       
       std::vector<GBRTree2D> &Trees() { return fTrees; }
       const std::vector<GBRTree2D> &Trees() const { return fTrees; }
       
    protected:
      double                 fInitialResponseX;
      double                 fInitialResponseY;
      std::vector<GBRTree2D> fTrees;  
      
  };

//_______________________________________________________________________
inline void GBRForest2D::GetResponse(const float* vector, double &x, double &y) const {
  x = fInitialResponseX;
  y = fInitialResponseY;
  double tx, ty;
  for (std::vector<GBRTree2D>::const_iterator it=fTrees.begin(); it!=fTrees.end(); ++it) {
    it->GetResponse(vector,tx,ty);
    x += tx;
    y += ty;
  }
  return;
}
  
#endif
