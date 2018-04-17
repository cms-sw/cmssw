// Forest.h

#ifndef L1Trigger_L1TMuonEndCap_emtf_Forest
#define L1Trigger_L1TMuonEndCap_emtf_Forest

#include "Tree.h"
#include "LossFunctions.h"
#include "CondFormats/L1TObjects/interface/L1TMuonEndCapForest.h"

namespace emtf {

class Forest
{
    public:

        // Constructor(s)/Destructor
        Forest();
        Forest(std::vector<Event*>& trainingEvents);
        ~Forest();

        Forest(const Forest &forest);
        Forest& operator=(const Forest &forest);
        Forest(Forest && forest) = default;

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
        void generate(int numTrainEvents, int numTestEvents, double sigma);
        void loadForestFromXML(const char* directory, unsigned int numTrees);
        void loadFromCondPayload(const L1TMuonEndCapForest::DForest& payload);

        // Perform the regression
        void updateRegTargets(Tree *tree, double learningRate, LossFunction* l);
        void doRegression(int nodeLimit, int treeLimit, double learningRate, LossFunction* l,
                          const char* savetreesdirectory, bool saveTrees);

        // Stochastic Gradient Boosting
        void prepareRandomSubsample(double fraction);
        void doStochasticRegression(int nodeLimit, int treeLimit, double learningRate,
                                    double fraction, LossFunction* l);

        // Predict some events
        void updateEvents(Tree* tree);
        void appendCorrection(std::vector<Event*>& eventsp, int treenum);
        void predictEvents(std::vector<Event*>& eventsp, unsigned int trees);
        void appendCorrection(Event* e, int treenum);
        void predictEvent(Event* e, unsigned int trees);

        Tree* getTree(unsigned int i);

    private:

        std::vector< std::vector<Event*> > events;
        std::vector< std::vector<Event*> > subSample;
        std::vector<Tree*> trees;
};

} // end of emtf namespace

#endif
