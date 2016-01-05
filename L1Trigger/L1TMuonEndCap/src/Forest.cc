//////////////////////////////////////////////////////////////////////////
//                            Forest.cxx                                //
// =====================================================================//
// This is the object implementation of a forest of decision trees.     //
// We need this to implement gradient boosting.                         //
// References include                                                   //
//    *Elements of Statistical Learning by Hastie,                      //
//     Tibshirani, and Friedman.                                        //
//    *Greedy Function Approximation: A Gradient Boosting Machine.      //
//     Friedman. The Annals of Statistics, Vol. 29, No. 5. Oct 2001.    //
//    *Inductive Learning of Tree-based Regression Models. Luis Torgo.  //    
//                                                                      //
//////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
// _______________________Includes_______________________________________//
///////////////////////////////////////////////////////////////////////////

#include "L1Trigger/L1TMuonEndCap/interface/Forest.h"
#include "L1Trigger/L1TMuonEndCap/interface/Utilities.h"

#include "TRandom3.h"
#include "TStopwatch.h"
#include "TROOT.h"
#include "TTree.h"
#include "TNtuple.h"
#include "TFile.h"
#include "TH1D.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TChain.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <utility>

//////////////////////////////////////////////////////////////////////////
// _______________________Constructor(s)________________________________//
//////////////////////////////////////////////////////////////////////////

Forest::Forest()
{
    events = std::vector< std::vector<Event*> >(1);
}

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

Forest::Forest(std::vector<Event*>& trainingEvents, std::vector<Event*>& testingEvents)
{
    setTrainingEvents(trainingEvents);
    setTestEvents(testingEvents);
}

/////////////////////////////////////////////////////////////////////////
// _______________________Destructor____________________________________//
//////////////////////////////////////////////////////////////////////////

Forest::~Forest()
{
// When the forest is destroyed it will delete the trees as well as the
// events from the training and testing sets.
// The user may want the events to remain after they destroy the forest
// this should be changed in future upgrades.

    for(unsigned int i=0; i < trees.size(); i++)
    { 
        delete trees[i];
    }

    for(unsigned int j=0; j < events[0].size(); j++)
    {
        delete events[0][j];
    }   

    for(unsigned int j=0; j < testEvents.size(); j++)
    {
        delete testEvents[j];
    }   
}
//////////////////////////////////////////////////////////////////////////
// ______________________Get/Set_Functions______________________________//
//////////////////////////////////////////////////////////////////////////

void Forest::setTrainingEvents(std::vector<Event*>& trainingEvents)
{
// tell the forest which events to use for training

    Event* e = trainingEvents[0];
    //unsigned int numrows = e->data.size();
   
    // Reset the events matrix. 
    events = std::vector< std::vector<Event*> >();

    for(unsigned int i=0; i<e->data.size(); i++) 
    {    
        events.push_back(trainingEvents);
    }    
}

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

void Forest::setTestEvents(std::vector<Event*>& testingEvents)
{   
// tell the forest which events to use for testing
    testEvents = testingEvents; 
}

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

// return a copy of the training events
std::vector<Event*> Forest::getTrainingEvents(){ return events[0]; }

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

// return a copy of the testEvents
std::vector<Event*> Forest::getTestEvents(){ return testEvents; }

//////////////////////////////////////////////////////////////////////////
// ______________________Various_Helpful_Functions______________________//
//////////////////////////////////////////////////////////////////////////

unsigned int Forest::size()
{
// Return the number of trees in the forest.
    return trees.size();
}

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//*** Need to make a data structure that includes the next few functions ***
//*** pertaining to events. These don't really have much to do with the  ***
//*** forest class.                                                      ***
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

void Forest::listEvents(std::vector< std::vector<Event*> >& e)
{
// Simply list the events in each event vector. We have multiple copies
// of the events vector. Each copy is sorted according to a different
// determining variable.
    std::cout << std::endl << "Listing Events... " << std::endl;

    for(unsigned int i=0; i < e.size(); i++)
    {
        std::cout << std::endl << "Variable " << i << " vector contents: " << std::endl;
        for(unsigned int j=0; j<e[i].size(); j++)
        {
            e[i][j]->outputEvent();
        }   
       std::cout << std::endl;
    }
}

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

// We have to initialize Event::sortingIndex outside of a function since
// it is a static member.
Int_t Event::sortingIndex = 1;

bool compareEvents(Event* e1, Event* e2)
{
// Sort the events according to the variable given by the sortingIndex.
    return e1->data[Event::sortingIndex] < e2->data[Event::sortingIndex];
}
//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

bool compareEventsById(Event* e1, Event* e2)
{
// Sort the events by ID. We need this to produce rate plots.
    return e1->id < e2->id;
}
//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

void Forest::sortEventVectors(std::vector< std::vector<Event*> >& e)
{
// When a node chooses the optimum split point and split variable it needs
// the events to be sorted according to the variable it is considering.

    for(unsigned int i=0; i<e.size(); i++)
    {
        Event::sortingIndex = i;
        std::sort(e[i].begin(), e[i].end(), compareEvents);
    }
}

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

std::vector<Double_t> Forest::rankVariables()
{
// This function ranks the determining variables according to their importance
// in determining the fit. Use a low learning rate for better results.
// Separates completely useless variables from useful ones well,
// but isn't the best at separating variables of similar importance. 
// This is calculated using the error reduction on the training set. The function
// should be changed to use the testing set, but this works fine for now.
// I will try to change this in the future.

    // Initialize the vector v, which will store the total error reduction
    // for each variable i in v[i].
    std::vector<Double_t> v(events.size(), 0);

    std::cout << std::endl << "Ranking Variables by Net Error Reduction... " << std::endl;

    for(unsigned int j=0; j < trees.size(); j++)
    {
        trees[j]->rankVariables(v); 
    }

    Double_t max = *std::max_element(v.begin(), v.end());
   
    // Scale the importance. Maximum importance = 100.
    for(unsigned int i=0; i < v.size(); i++)
    {
        v[i] = 100*v[i]/max;
    }

    // Change the storage format so that we can keep the index 
    // and the value associated after sorting.
    std::vector< std::pair<Double_t, Int_t> > w(events.size());

    for(unsigned int i=0; i<v.size(); i++)
    {
        w[i] = std::pair<Double_t, Int_t>(v[i],i);
    }

    // Sort so that we can output in order of importance.
    std::sort(w.begin(),w.end());

    // Output the results.
    for(int i=(v.size()-1); i>=0; i--)
    {
        std::cout << "x" << w[i].second  << ": " << w[i].first  << std::endl; 
    }
    
    std::cout << std::endl << "Done." << std::endl << std::endl;
    return v;

}

//////////////////////////////////////////////////////////////////////////
// ______________________Update_Events_After_Fitting____________________//
//////////////////////////////////////////////////////////////////////////

void Forest::updateRegTargets(Tree* tree, Double_t learningRate, LossFunction* l)
{
// Prepare the global vector of events for the next tree.
// Update the fit for each event and set the new target value
// for the next tree.

    // Get the list of terminal nodes for this tree.
    std::list<Node*>& tn = tree->getTerminalNodes();

    // Loop through the terminal nodes.
    for(std::list<Node*>::iterator it=tn.begin(); it!=tn.end(); it++)
    {   
        // Get the events in the current terminal region.
        std::vector<Event*>& v = (*it)->getEvents()[0];

        // Fit the events depending on the loss function criteria.
        Double_t fit = l->fit(v);

        // Scale the rate at which the algorithm converges.
        fit = learningRate*fit;

        // Store the official fit value in the terminal node.
        (*it)->setFitValue(fit);

        // Loop through each event in the terminal region and update the
        // the target for the next tree.
        for(unsigned int j=0; j<v.size(); j++)
        {
            Event* e = v[j];
            e->predictedValue += fit;
            e->data[0] = l->target(e);
        }

        // Release memory.
        (*it)->getEvents() = std::vector< std::vector<Event*> >();
    }
}

/////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
/////////////////////////////////////////////////////////////////////////

void Forest::updateEvents(Tree* tree)
{
// Prepare the test events for the next tree.

    // Get the list of terminal nodes for this tree.
    std::list<Node*>& tn = tree->getTerminalNodes();

    // Loop through the terminal nodes.
    for(std::list<Node*>::iterator it=tn.begin(); it!=tn.end(); it++)
    {   
        std::vector<Event*>& v = (*it)->getEvents()[0];
        Double_t fit = (*it)->getFitValue();

        // Loop through each event in the terminal region and update the
        // the global event it maps to.
        for(unsigned int j=0; j<v.size(); j++)
        {   
            Event* e = v[j];
            e->predictedValue += fit;
        }   

        // Release memory.
        (*it)->getEvents() = std::vector< std::vector<Event*> >();
    }   
}

//////////////////////////////////////////////////////////////////////////
// ____________________Do/Test_the Regression___________________________//
//////////////////////////////////////////////////////////////////////////

void Forest::doRegression(Int_t nodeLimit, Int_t treeLimit, Double_t learningRate, LossFunction* l, const char* savetreesdirectory, bool saveTrees)
{
// Build the forest using the training sample.

    std::cout << std::endl << "--Building Forest..." << std::endl << std::endl;

    // The trees work with a matrix of events where the rows have the same set of events. Each row however
    // is sorted according to the feature variable given by event->data[row].
    // If we only had one set of events we would have to sort it according to the
    // feature variable every time we want to calculate the best split point for that feature.
    // By keeping sorted copies we avoid the sorting operation during splint point calculation
    // and save computation time. If we do not sort each of the rows the regression will fail.
    std::cout << "Sorting event vectors..." << std::endl;
    sortEventVectors(events);

    // See how long the regression takes.
    TStopwatch timer;
    timer.Start(kTRUE);

    for(unsigned int i=0; i< (unsigned) treeLimit; i++)
    {
        std::cout << "++Building Tree " << i << "... " << std::endl;
        Tree* tree = new Tree(events);
        trees.push_back(tree);    
        tree->buildTree(nodeLimit);

        // Update the targets for the next tree to fit.
        updateRegTargets(tree, learningRate, l);

        // Save trees to xml in some directory.
        std::ostringstream ss; 
        ss << savetreesdirectory << "/" << i << ".xml";
        std::string s = ss.str();
        const char* c = s.c_str();

        if(saveTrees) tree->saveToXML(c);
    }
    std::cout << std::endl;
    std::cout << std::endl << "Done." << std::endl << std::endl;

//    std::cout << std::endl << "Total calculation time: " << timer.RealTime() << std::endl;
}

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

void Forest::predictEvents(std::vector<Event*> eventsp, unsigned int numtrees)
{
// Predict values for eventsp by running them through the forest up to numtrees.

    //std::cout << "Using " << numtrees << " trees from the forest to predict events ... " << std::endl;
    if(numtrees > trees.size())
    {
       // std::cout << std::endl << "!! Input greater than the forest size. Using forest.size() = " << trees.size() << " to predict instead." << std::endl;
        numtrees = trees.size();
    }

    // i iterates through the trees in the forest. Each tree corrects the last prediction.
    for(unsigned int i=0; i < numtrees; i++) 
    {
        //std::cout << "++Tree " << i << "..." << std::endl;
        appendCorrection(eventsp, i);
    }
}

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

void Forest::appendCorrection(std::vector<Event*> eventsp, Int_t treenum)
{
// Update the prediction by appending the next correction.

    Tree* tree = trees[treenum];
    tree->filterEvents(eventsp); 

    // Update the events with their new prediction.
    updateEvents(tree);
}

/////////////////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------------------
/////////////////////////////////////////////////////////////////////////////////////

void Forest::loadForestFromXML(const char* directory, unsigned int numTrees)
{
// Load a forest that has already been created and stored into XML somewhere.

    // Initialize the vector of trees.
    trees = std::vector<Tree*>(numTrees);

    // Load the Forest.
    //std::cout << std::endl << "Loading Forest from XML ... " << std::endl;
    for(unsigned int i=0; i < numTrees; i++) 
    {   
        trees[i] = new Tree(); 

        std::stringstream ss;
        ss << directory << "/" << i << ".xml";
	
		trees[i]->loadFromXML(edm::FileInPath(ss.str().c_str()).fullPath().c_str());
    }   

   // std::cout << "Done." << std::endl << std::endl;
}

//////////////////////////////////////////////////////////////////////////
// ___________________Stochastic_Sampling_&_Regression__________________//
//////////////////////////////////////////////////////////////////////////

void Forest::prepareRandomSubsample(Double_t fraction)
{
// We use this for Stochastic Gradient Boosting. Basically you
// take a subsample of the training events and build a tree using
// those. Then use the tree built from the subsample to update
// the predictions for all the events.

    subSample = std::vector< std::vector<Event*> >(events.size()) ;
    size_t subSampleSize = fraction*events[0].size();

    // Randomize the first subSampleSize events in events[0].
    shuffle(events[0].begin(), events[0].end(), subSampleSize);

    // Get a copy of the random subset we just made.
    std::vector<Event*> v(events[0].begin(), events[0].begin()+subSampleSize); 

    // Initialize and sort the subSample collection.
    for(unsigned int i=0; i<subSample.size(); i++)
    {
        subSample[i] = v;
    }
    
    sortEventVectors(subSample);
}

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

void Forest::doStochasticRegression(Int_t nodeLimit, Int_t treeLimit, Double_t learningRate, Double_t fraction, LossFunction* l)
{
// If the fraction of events to use is one then this algorithm is slower than doRegression due to the fact
// that we have to sort the events every time we extract a subsample. Without random sampling we simply 
// use all of the events and keep them sorted.

// Anyways, this algorithm uses a portion of the events to train each tree. All of the events are updated
// afterwards with the results from the subsample built tree.

    // Prepare some things.
    sortEventVectors(events);
    trees = std::vector<Tree*>(treeLimit);

    // See how long the regression takes.
    TStopwatch timer;
    timer.Start(kTRUE); 

    // Output the current settings.
    std::cout << std::endl << "Running stochastic regression ... " << std::endl;
    std::cout << "# Nodes: " << nodeLimit << std::endl;
    std::cout << "Learning Rate: " << learningRate << std::endl;
    std::cout << "Bagging Fraction: " << fraction << std::endl;
    std::cout << std::endl;
    

    for(unsigned int i=0; i< (unsigned) treeLimit; i++)
    {
        // Build the tree using a random subsample.
        prepareRandomSubsample(fraction);
        trees[i] = new Tree(subSample);    
        trees[i]->buildTree(nodeLimit);

        // Fit all of the events based upon the tree we built using
        // the subsample of events.
        trees[i]->filterEvents(events[0]);

        // Update the targets for the next tree to fit.
        updateRegTargets(trees[i], learningRate, l);

        // Save trees to xml in some directory.
        std::ostringstream ss; 
        ss << "trees/" << i << ".xml";
        std::string s = ss.str();
        const char* c = s.c_str();

        trees[i]->saveToXML(c);
    }

    std::cout << std::endl << "Done." << std::endl << std::endl;

    std::cout << std::endl << "Total calculation time: " << timer.RealTime() << std::endl;
}
//////////////////////////////////////////////////////////////////////////
// ______________________Generate_Events________________________________//
//////////////////////////////////////////////////////////////////////////

void Forest::generate(Int_t n, Int_t m, Double_t sigma)
{
// Generate events to use for the building and testing of the forest.
// We keep as many copies of the events as there are variables.
// And we store these copies in the events vector of vectors.
// events[0] is a vector sorted by var 0, events[1] by var 1, etc.
// All of the vectors have the same events, but each vector is just
// sorted by a different variable.
 
    // Store these in case we need them
    // for plotting or troubleshooting.
    std::ofstream trainData;
    trainData.open("training.data");

    std::ofstream testData;
    testData.open("testing.data");

    // Prepare our containers.
    TRandom3 r(0);
    std::vector<Event*> v(n);

    events = std::vector< std::vector<Event*> >(3, std::vector<Event*>(n));
    testEvents = std::vector<Event*>(m);

    std::cout << std::endl << "Generating " << n << " events..." << std::endl;

    // Generate the data set we will use to build the forest. 
    for(unsigned int i=0; i< (unsigned) n; i++)
    {  
        // data[0] is the target, which is determined
        // by the other variables data[1], data[2] ... 
        std::vector<Double_t> x(3);
        x[1] = r.Rndm();
        x[2] = r.Rndm();

        // Store the variable which is determined by the others.
        // Our target for BDT prediction.
        x[0] = x[1]*x[2];


        // Add noise to the determining variables.
        x[1] += r.Gaus(0,sigma);
        x[2] += r.Gaus(0,sigma);

        // Create the event.
        Event* e = new Event();
        v[i]=e;

        // Store the event.
        e->predictedValue = 0;
        e->trueValue = x[0];
        e->data = x;  
        e->id = i;
    }

    // Set up the events matrix and the events vector.
    for(unsigned int i=0; i < events.size(); i++)
    {
        events[i] = v;
    }

    // Generate a separate data set for testing.
    for(unsigned int i=0; i< (unsigned) m; i++)
    {  
        // data[0] is the target, which is determined
        // by the other variables data[1], data[2] .... 
        std::vector<Double_t> x(3);
        x[1] = r.Rndm();
        x[2] = r.Rndm();
        x[0] = x[1]*x[2];

        x[1] += r.Gaus(0,sigma);
        x[2] += r.Gaus(0,sigma);

        // Create the event.
        Event* e = new Event();
        Event* f = new Event();

        testEvents[i] = e;

        // Store the event.
        e->predictedValue = 0;
        e->trueValue = x[0];
        e->data = x;  
        e->id = i;

        f->predictedValue = 0;
        f->trueValue = x[0];
        f->data = x;  
        f->id = i;
        
    }

    // Sort the events by the target variable.
    Event::sortingIndex=0;

    for(unsigned int i=0; i< (unsigned) n; i++)
    {
        // Argh, write to files if ye want, matie.
    }

    trainData.close();
    testData.close();
}
