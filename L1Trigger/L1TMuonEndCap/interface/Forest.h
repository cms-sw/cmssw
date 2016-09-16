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
  L1TForest(std::vector<Event*>& trainingEvents);
  ~L1TForest();
  
  // Get/Set
  void setTrainingEvents(std::vector<Event*>& trainingEvents);
  std::vector<Event*> getTrainingEvents();
  
  // Returns the number of trees in the forest.
  unsigned int size();
  
  // Get info on variable importance.
  void rankVariables(std::vector<int>& rank);
  
  // Output the list of split values used for each variable.
  void saveSplitValues(const char* savefilename);
  
  // Helpful operations
  void listEvents(std::vector< std::vector<Event*> >& e);
  void sortEventVectors(std::vector< std::vector<Event*> >& e);
  void generate(Int_t numTrainEvents, Int_t numTestEvents, double sigma);
  void loadL1TForestFromXML(const char* directory, unsigned int numTrees); 
  
  // Perform the regression
  void updateRegTargets(Tree *tree, double learningRate, L1TLossFunction* l);
  void doRegression(Int_t nodeLimit, Int_t treeLimit, double learningRate, L1TLossFunction* l, 
		    const char* savetreesdirectory, bool saveTrees);
  
  // Stochastic Gradient Boosting
  void prepareRandomSubsample(double fraction);
  void doStochasticRegression(Int_t nodeLimit, Int_t treeLimit, double learningRate, 
			      double fraction, L1TLossFunction* l);
  
  // Predict some events
  void updateEvents(Tree* tree);
  void appendCorrection(std::vector<Event*>& eventsp, Int_t treenum);
  void predictEvents(std::vector<Event*>& eventsp, unsigned int trees);
  void appendCorrection(Event* e, Int_t treenum);
  void predictEvent(Event* e, unsigned int trees);
  
  Tree* getTree(unsigned int i);
  
 private:
  
  std::vector< std::vector<Event*> > events;
  std::vector< std::vector<Event*> > subSample;
  std::vector<Tree*> trees;
};

#endif
