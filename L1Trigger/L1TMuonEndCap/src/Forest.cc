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

#include "TStopwatch.h"

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

Forest::Forest(std::vector<Event*>& trainingEvents)
{
    setTrainingEvents(trainingEvents);
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
}
//////////////////////////////////////////////////////////////////////////
// ______________________Get/Set_Functions______________________________//
//////////////////////////////////////////////////////////////////////////

void Forest::setTrainingEvents(std::vector<Event*>& trainingEvents)
{
// tell the forest which events to use for training

    Event* e = trainingEvents[0];
    // Unused variable
    // unsigned int numrows = e->data.size();
   
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

// return a copy of the training events
std::vector<Event*> Forest::getTrainingEvents(){ return events[0]; }

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

// return the ith tree
Tree* Forest::getTree(unsigned int i)
{ 
    if(/*i>=0 && */i<trees.size()) return trees[i]; 
    else
    {
      //std::cout << i << "is an invalid input for getTree. Out of range." << std::endl;
        return 0;
    }
}

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

void Forest::rankVariables(std::vector<int>& rank)
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
    std::vector<double> v(events.size(), 0);

    //std::cout << std::endl << "Ranking Variables by Net Error Reduction... " << std::endl;

    for(unsigned int j=0; j < trees.size(); j++)
    {
        trees[j]->rankVariables(v); 
    }

    double max = *std::max_element(v.begin(), v.end());
   
    // Scale the importance. Maximum importance = 100.
    for(unsigned int i=0; i < v.size(); i++)
    {
        v[i] = 100*v[i]/max;
    }

    // Change the storage format so that we can keep the index 
    // and the value associated after sorting.
    std::vector< std::pair<double, Int_t> > w(events.size());

    for(unsigned int i=0; i<v.size(); i++)
    {
        w[i] = std::pair<double, Int_t>(v[i],i);
    }

    // Sort so that we can output in order of importance.
    std::sort(w.begin(),w.end());

    // Output the results.
    for(int i=(v.size()-1); i>=0; i--)
    {
        rank.push_back(w[i].second);
       // std::cout << "x" << w[i].second  << ": " << w[i].first  << std::endl; 
    }
    
    //std::cout << std::endl << "Done." << std::endl << std::endl;
}

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

void Forest::saveSplitValues(const char* savefilename)
{
// This function gathers all of the split values from the forest and puts them into lists.

    std::ofstream splitvaluefile;
    splitvaluefile.open(savefilename);

    // Initialize the matrix v, which will store the list of split values
    // for each variable i in v[i].
    std::vector<std::vector<double>> v(events.size(), std::vector<double>());

    //std::cout << std::endl << "Gathering split values... " << std::endl;

    // Gather the split values from each tree in the forest.
    for(unsigned int j=0; j<trees.size(); j++)
    {
        trees[j]->getSplitValues(v); 
    }

    // Sort the lists of split values and remove the duplicates.
    for(unsigned int i=0; i<v.size(); i++)
    {
        std::sort(v[i].begin(),v[i].end());
        v[i].erase( unique( v[i].begin(), v[i].end() ), v[i].end() );
    }

    // Output the results after removing duplicates.
    // The 0th variable is special and is not used for splitting, so we start at 1.
    for(unsigned int i=1; i<v.size(); i++)
    {
      TString splitValues;
      for(unsigned int j=0; j<v[i].size(); j++)
      {
        std::stringstream ss;
        ss.precision(14);
        ss << std::scientific << v[i][j];
        splitValues+=","; 
        splitValues+=ss.str().c_str();
      }

      splitValues=splitValues(1,splitValues.Length());
      splitvaluefile << splitValues << std::endl << std::endl;;
    }
}
//////////////////////////////////////////////////////////////////////////
// ______________________Update_Events_After_Fitting____________________//
//////////////////////////////////////////////////////////////////////////

void Forest::updateRegTargets(Tree* tree, double learningRate, LossFunction* l)
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
        double fit = l->fit(v);

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
        double fit = (*it)->getFitValue();

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

void Forest::doRegression(Int_t nodeLimit, Int_t treeLimit, double learningRate, LossFunction* l, const char* savetreesdirectory, bool saveTrees)
{
// Build the forest using the training sample.

    //std::cout << std::endl << "--Building Forest..." << std::endl << std::endl;

    // The trees work with a matrix of events where the rows have the same set of events. Each row however
    // is sorted according to the feature variable given by event->data[row].
    // If we only had one set of events we would have to sort it according to the
    // feature variable every time we want to calculate the best split point for that feature.
    // By keeping sorted copies we avoid the sorting operation during splint point calculation
    // and save computation time. If we do not sort each of the rows the regression will fail.
    //std::cout << "Sorting event vectors..." << std::endl;
    sortEventVectors(events);

    // See how long the regression takes.
    TStopwatch timer;
    timer.Start(kTRUE);

    for(unsigned int i=0; i< (unsigned) treeLimit; i++)
    {
       // std::cout << "++Building Tree " << i << "... " << std::endl;
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
    //std::cout << std::endl;
    //std::cout << std::endl << "Done." << std::endl << std::endl;

//    std::cout << std::endl << "Total calculation time: " << timer.RealTime() << std::endl;
}

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

void Forest::predictEvents(std::vector<Event*>& eventsp, unsigned int numtrees)
{
// Predict values for eventsp by running them through the forest up to numtrees.

    //std::cout << "Using " << numtrees << " trees from the forest to predict events ... " << std::endl;
    if(numtrees > trees.size())
    {
      //std::cout << std::endl << "!! Input greater than the forest size. Using forest.size() = " << trees.size() << " to predict instead." << std::endl;
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

void Forest::appendCorrection(std::vector<Event*>& eventsp, Int_t treenum)
{
// Update the prediction by appending the next correction.

    Tree* tree = trees[treenum];
    tree->filterEvents(eventsp); 

    // Update the events with their new prediction.
    updateEvents(tree);
}

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

void Forest::predictEvent(Event* e, unsigned int numtrees)
{
// Predict values for eventsp by running them through the forest up to numtrees.

    //std::cout << "Using " << numtrees << " trees from the forest to predict events ... " << std::endl;
    if(numtrees > trees.size())
    {
      //std::cout << std::endl << "!! Input greater than the forest size. Using forest.size() = " << trees.size() << " to predict instead." << std::endl;
        numtrees = trees.size();
    }

    // i iterates through the trees in the forest. Each tree corrects the last prediction.
    for(unsigned int i=0; i < numtrees; i++) 
    {
        //std::cout << "++Tree " << i << "..." << std::endl;
        appendCorrection(e, i);
    }
}

//////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------
//////////////////////////////////////////////////////////////////////////

void Forest::appendCorrection(Event* e, Int_t treenum)
{
// Update the prediction by appending the next correction.

    Tree* tree = trees[treenum];
    Node* terminalNode = tree->filterEvent(e); 

    // Update the event with its new prediction.
    double fit = terminalNode->getFitValue();
    e->predictedValue += fit;
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

        //trees[i]->loadFromXML(ss.str().c_str());
		trees[i]->loadFromXML(edm::FileInPath(ss.str().c_str()).fullPath().c_str());
    }   

   // std::cout << "Done." << std::endl << std::endl;
}

//////////////////////////////////////////////////////////////////////////
// ___________________Stochastic_Sampling_&_Regression__________________//
//////////////////////////////////////////////////////////////////////////

void Forest::prepareRandomSubsample(double fraction)
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

void Forest::doStochasticRegression(Int_t nodeLimit, Int_t treeLimit, double learningRate, double fraction, LossFunction* l)
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
   // std::cout << std::endl << "Running stochastic regression ... " << std::endl;
    //std::cout << "# Nodes: " << nodeLimit << std::endl;
    //std::cout << "Learning Rate: " << learningRate << std::endl;
    //std::cout << "Bagging Fraction: " << fraction << std::endl;
    //std::cout << std::endl;
    

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

    //std::cout << std::endl << "Done." << std::endl << std::endl;

    //std::cout << std::endl << "Total calculation time: " << timer.RealTime() << std::endl;
}
