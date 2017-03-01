// Forest.h

#ifndef L1Trigger_L1TMuonEndCap_Forest
#define L1Trigger_L1TMuonEndCap_Forest

#include "L1Trigger/L1TMuonEndCap/interface/Tree.h"
#include "L1Trigger/L1TMuonEndCap/interface/LossFunctions.h"

class L1TForest
{
 public:
  
  // Constructor(s)/Destructor
  L1TForest();
  L1TForest(std::vector<emtf::Event*>& trainingEvents);
  ~L1TForest();
  
  // Get/Set
  void setTrainingEvents(std::vector<emtf::Event*>& trainingEvents);
  std::vector<emtf::Event*> getTrainingEvents();
  
  // Returns the number of trees in the forest.
  unsigned int size();
  
  // Get info on variable importance.
  void rankVariables(std::vector<int>& rank);
  
  // Output the list of split values used for each variable.
  void saveSplitValues(const char* savefilename);
  
  // Helpful operations
  void listEvents(std::vector< std::vector<emtf::Event*> >& e);
  void sortEventVectors(std::vector< std::vector<emtf::Event*> >& e);
  void generate(Int_t numTrainEvents, Int_t numTestEvents, double sigma);
  void loadL1TForestFromXML(const char* directory, unsigned int numTrees); 
  
  // Perform the regression
  void updateRegTargets(emtf::Tree *tree, double learningRate, L1TLossFunction* l);
  void doRegression(Int_t nodeLimit, Int_t treeLimit, double learningRate, L1TLossFunction* l, 
		    const char* savetreesdirectory, bool saveTrees);
  
  // Stochastic Gradient Boosting
  void prepareRandomSubsample(double fraction);
  void doStochasticRegression(Int_t nodeLimit, Int_t treeLimit, double learningRate, 
			      double fraction, L1TLossFunction* l);
  
  // Predict some events
  void updateEvents(emtf::Tree* tree);
  void appendCorrection(std::vector<emtf::Event*>& eventsp, Int_t treenum);
  void predictEvents(std::vector<emtf::Event*>& eventsp, unsigned int trees);
  void appendCorrection(emtf::Event* e, Int_t treenum);
  void predictEvent(emtf::Event* e, unsigned int trees);
  
  emtf::Tree* getTree(unsigned int i);
  
 private:
  
  std::vector< std::vector<emtf::Event*> > events;
  std::vector< std::vector<emtf::Event*> > subSample;
  std::vector<emtf::Tree*> trees;
};

#endif
