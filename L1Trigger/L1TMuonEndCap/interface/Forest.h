// Forest.h

#ifndef ADD_FOREST
#define ADD_FOREST

#include "L1Trigger/L1TMuonEndCap/interface/Tree.h"
#include "L1Trigger/L1TMuonEndCap/interface/LossFunctions.h"

class Forest
{
    public:

        // Constructor(s)/Destructor
        Forest();
        Forest(std::vector<Event*>& trainingEvents, std::vector<Event*>& testEvents);
        ~Forest();

        // Get/Set
        void setTrainingEvents(std::vector<Event*>& trainingEvents);
        void setTestEvents(std::vector<Event*>& testingEvents);
        std::vector<Event*> getTrainingEvents();
        std::vector<Event*> getTestEvents();

        // Returns the number of trees in the forest.
        unsigned int size();

        // Get info on variable importance.
        std::vector<Double_t> rankVariables();

        // Helpful operations
        void listEvents(std::vector< std::vector<Event*> >& e);
        void sortEventVectors(std::vector< std::vector<Event*> >& e);
        void generate(Int_t numTrainEvents, Int_t numTestEvents, Double_t sigma);
        void loadForestFromXML(const char* directory, unsigned int numTrees); 

        // Perform the regression
        void updateRegTargets(Tree *tree, Double_t learningRate, LossFunction* l);
        void doRegression(Int_t nodeLimit, Int_t treeLimit, Double_t learningRate, LossFunction* l, 
                          const char* savetreesdirectory, bool saveTrees);

        // Stochastic Gradient Boosting
        void prepareRandomSubsample(Double_t fraction);
        void doStochasticRegression(Int_t nodeLimit, Int_t treeLimit, Double_t learningRate, 
                                    Double_t fraction, LossFunction* l);

        // Predict some events
        void updateEvents(Tree* tree);
        void appendCorrection(std::vector<Event*> eventsp, Int_t treenum);
        void predictEvents(std::vector<Event*> eventsp, unsigned int trees);

    private:

        std::vector< std::vector<Event*> > events;
        std::vector< std::vector<Event*> > subSample;
        std::vector<Event*> testEvents;
        std::vector<Tree*> trees;
};

#endif
