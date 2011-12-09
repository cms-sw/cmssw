
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


#include <vector>
#include <map>
#include "Rtypes.h"


  namespace TMVA {
    class DecisionTree;
    class DecisionTreeNode;
  }

  class GBRTree {

    public:

       GBRTree();
       GBRTree(const TMVA::DecisionTree *tree);
       //       GBRTree(const GBRTree &other);       

       virtual ~GBRTree();
       
       Double_t GetResponse(const Float_t* vector) const;

       //       virtual void Streamer(TBuffer &b);
       
    protected:      
        UInt_t CountIntermediateNodes(const TMVA::DecisionTreeNode *node);
        UInt_t CountTerminalNodes(const TMVA::DecisionTreeNode *node);
      
        void AddNode(const TMVA::DecisionTreeNode *node);
        
        Int_t    fNIntermediateNodes;
        Int_t    fNTerminalNodes;
      
	std::vector<UChar_t> fCutIndices;//[fNIntermediateNodes]
	std::vector<Float_t> fCutVals;//[fNIntermediateNodes]
	std::vector<Int_t> fLeftIndices;//[fNIntermediateNodes]
	std::vector<Int_t> fRightIndices;//[fNIntermediateNodes]        
	std::vector<Float_t> fResponses;//[fNTerminalNodes]

        
  };

//_______________________________________________________________________
inline Double_t GBRTree::GetResponse(const Float_t* vector) const {
  
  Int_t index = 0;
  
  UChar_t cutindex = fCutIndices[0];
  Float_t cutval = fCutVals[0];
  
  while (true) {
    if (vector[cutindex] > cutval) {
      index = fRightIndices[index];
    }
    else {
      index = fLeftIndices[index];
    }
    
    if (index>0) {
      cutindex = fCutIndices[index];
      cutval = fCutVals[index];
    }
    else {
      return fResponses[-index];
    }
    
  }
  

}
  
#endif
