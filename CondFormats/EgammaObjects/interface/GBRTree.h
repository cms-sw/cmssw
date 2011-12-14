
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


  namespace TMVA {
    class DecisionTree;
    class DecisionTreeNode;
  }

  class GBRTree {

    public:

       GBRTree();
       explicit GBRTree(const TMVA::DecisionTree *tree);
       virtual ~GBRTree();
       
       double GetResponse(const float* vector) const;
       
    protected:      
        unsigned int CountIntermediateNodes(const TMVA::DecisionTreeNode *node);
        unsigned int CountTerminalNodes(const TMVA::DecisionTreeNode *node);
      
        void AddNode(const TMVA::DecisionTreeNode *node);
        
	std::vector<unsigned char> fCutIndices;
	std::vector<float> fCutVals;
	std::vector<int> fLeftIndices;
	std::vector<int> fRightIndices;
	std::vector<float> fResponses;  
        
  };

//_______________________________________________________________________
inline double GBRTree::GetResponse(const float* vector) const {
  
  int index = 0;
  
  unsigned char cutindex = fCutIndices[0];
  float cutval = fCutVals[0];
  
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
