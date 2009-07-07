//$Id: SprGoFDecisionTreeApp.cc,v 1.3 2007/10/07 21:03:02 narsky Exp $

/*
  This executable evaluates consistency between two multivariate samples
  using a goodness-of-fit method described by Friedman in the proceedings
  of Phystat 2003.

  Two samples labeled as category 0 and category 1, of sizes N0 and N1,
  respectively, are mixed together. At first step, the executable builds
  a decision tree to separate these two samples from each other using
  the Gini index as the optimization criterion. The associated value of the
  Gini index is used as an estimate of the consistency between the two 
  samples.

  To obtain the GOF value, one needs to obtain a distribution of Gini 
  indexes for toy samples. The toy samples are obtained by random relabeling
  of points from the two original samples in such a way that one always has
  N0 points from category 0 and N1 points from category 1. For each toy 
  experiment, a decision tree with the same parameters is used to separate
  category 0 from category 1 and compute the associated Gini index. The GOF
  is estimated as a fraction of toy experiments in which the negative Gini 
  index is greater than the one computed for the two original samples
  (or, equivalently, the conventional positive Gini index is less than the
  one for the two original samples).

  In this approach, an ideal consistency between two samples would be
  expressed by a 50% GOF value. Small values of GOF indicate that the two
  samples are inconsistent. Values of GOF close to 100% generally indicate
  a good agreement between the two samples but may be worth further
  investigating: somehow the good agreement of the two original samples cannot
  be reproduced by random relabeling of input points. This may indicate
  a problem with how the method works for this particular dataset.

  To use the executable, the user has to choose two parameters:
    -n --- number of toy experiments
    -l --- minimal number of events in the decision tree leaf
  The minimal leaf size has to be chosen by finding a good trade-off between
  bias and variance. One can think of it in terms of selecting a proper
  bin size for a multidimensional histogram. If the bin size is chosen small,
  the histogram will capture the data structure but suffer from large
  statistical fluctuations in the bins (low bias, high variance). If the 
  bin size is chosen large, statistical fluctuations will be suppressed but
  the histogram may not be able to capture the data structure (high bias,
  low variance). In principle, the optimal leaf size can be chosen by an
  independent experiment - if you generate an independent sample of 
  category 0 and an independent sample of category 1 and run this method
  for several values of the leaf size, choose the one that gives the poorest
  GOF value; that way you will maximize the sensitivity of your method.
  In the absence of an independent sample, guess.

  If the method returns a poor GOF value, you can train a decision tree with
  the same leaf size and see for yourself what part of the input space causes
  the problem.

  Examples of usage:
  -----------------

  unix> SprGoFDecisionTreeApp -n 100 -l 1000 uniform_on_uniform2_d.pat

  Evaluates consistency of two 2D uniform distributions. You get:

24 experiments out of 100 have better GoF values than the data.
GoF=0.76

  unix> SprGoFDecisionTreeApp -n 100 -l 1000 gauss_uniform2_d_train.pat 

  Evaluates consistency of a 2D Gaussian and a 2D uniform distribution. 
  You get:

100 experiments out of 100 have better GoF values than the data.
GoF=0

  Let us look at this sample with a decision tree:

  unix> SprDecisionTreeApp -c 5 -n 1000 -f save.spr gauss_uniform2_d_train.pat

Read data from file gauss_uniform2_d_train.pat for variables "x0" "x1"
Total number of points read: 20000
Points in class 0: 10000 1: 10000
Optimization criterion set to Gini index  -1+p^2+q^2
Decision tree initialized mith minimal number of events per node 1000
Included 9 nodes in category 1 with overall FOM=-0.725763    W1=8736 W0=2731    N1=8736 N0=2731

  unix> less save.spr

Trained Decision Tree: 9      nodes.    Overall FOM=-0.725763  W0=2731       W1=8736       N0=2731       N1=8736       
-------------------------------------------------------
Node      0    Size 2       FOM=-0.385045  W0=172        W1=1422       N0=172        N1=1422
Variable                             x0    Limits         -1.04095        0.993995
Variable                             x1    Limits        -0.631275         0.30833
.......

  The first node of the decision tree points to the region in space with 
  a large excess of events from category 1 over events from category 0.
*/

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDecisionTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRWFactory.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassGiniIndex.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerPermutator.hh"

#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
#include <utility>
#include <cassert>

using namespace std;


void help(const char* prog) 
{
  cout << "Usage:  " << prog 
       << " training_data_file" << endl;
  cout << "\t Options: " << endl;
  cout << "\t-h --- help                                        " << endl;
  cout << "\t-a input ascii file mode (see SprSimpleReader.hh)  " << endl;
  cout << "\t-s random seed for permutations (default=0)        " << endl;
  cout << "\t-n number of cycles for GoF evaluation             " << endl;
  cout << "\t-l minimal number of entries per tree leaf (def=1) " << endl;
  cout << "\t-v verbose level (0=silent default,1,2)            " << endl;
  cout << "\t-w scale all signal weights by this factor         " << endl;
  cout << "\t-V include only these input variables              " << endl;
  cout << "\t-z exclude input variables from the list           " << endl;
  cout << "\t\t Variables must be listed in quotes and separated by commas." 
       << endl;
}


int main(int argc, char ** argv)
{
  // check command line
  if( argc < 2 ) {
    help(argv[0]);
    return 1;
  }

  // init
  string tupleFile;
  int readMode = 0;
  unsigned cycles = 0;
  unsigned nmin = 1;
  int verbose = 0;
  string includeList, excludeList;
  int seed = 0;
  bool scaleWeights = false;
  double sW = 1.;

  // decode command line
  int c;
  extern char* optarg;
  //  extern int optind;
  while( (c = getopt(argc,argv,"ha:s:n:l:v:w:V:z:")) != EOF ) {
    switch( c )
      {
      case 'h' :
	help(argv[0]);
	return 1;
      case 'a' :
	readMode = (optarg==0 ? 0 : atoi(optarg));
	break;
      case 's' :
        seed = (optarg==0 ? 0 : atoi(optarg));
        break;
      case 'n' :
	cycles = (optarg==0 ? 1 : atoi(optarg));
	break;
      case 'l' :
	nmin = (optarg==0 ? 1 : atoi(optarg));
	break;
      case 'v' :
        verbose = (optarg==0 ? 0 : atoi(optarg));
        break;
      case 'w' :
	if( optarg != 0 ) {
	  scaleWeights = true;
	  sW = atof(optarg);
	}
	break;
      case 'V' :
	includeList = optarg;
	break;
      case 'z' :
	excludeList = optarg;
	break;
      }
  }

  // There has to be 1 argument after all options.
  string trFile = argv[argc-1];
  if( trFile.empty() ) {
    cerr << "No training file is specified." << endl;
    return 1;
  }

  // make reader
  SprRWFactory::DataType inputType 
    = ( readMode==0 ? SprRWFactory::Root : SprRWFactory::Ascii );
  auto_ptr<SprAbsReader> reader(SprRWFactory::makeReader(inputType,readMode));

  // include variables
  set<string> includeSet;
  if( !includeList.empty() ) {
    vector<vector<string> > includeVars;
    SprStringParser::parseToStrings(includeList.c_str(),includeVars);
    assert( !includeVars.empty() );
    for( unsigned int i=0;i<includeVars[0].size();i++ ) 
      includeSet.insert(includeVars[0][i]);
    if( !reader->chooseVars(includeSet) ) {
      cerr << "Unable to include variables in training set." << endl;
      return 2;
    }
    else {
      cout << "Following variables have been included in optimization: ";
      for( set<string>::const_iterator 
	     i=includeSet.begin();i!=includeSet.end();i++ )
	cout << "\"" << *i << "\"" << " ";
      cout << endl;
    }
  }

  // exclude variables
  set<string> excludeSet;
  if( !excludeList.empty() ) {
    vector<vector<string> > excludeVars;
    SprStringParser::parseToStrings(excludeList.c_str(),excludeVars);
    assert( !excludeVars.empty() );
    for( unsigned int i=0;i<excludeVars[0].size();i++ ) 
      excludeSet.insert(excludeVars[0][i]);
    if( !reader->chooseAllBut(excludeSet) ) {
      cerr << "Unable to exclude variables from training set." << endl;
      return 2;
    }
    else {
      cout << "Following variables have been excluded from optimization: ";
      for( set<string>::const_iterator 
	     i=excludeSet.begin();i!=excludeSet.end();i++ )
	cout << "\"" << *i << "\"" << " ";
      cout << endl;
    }
  }

  // read training data from file
  auto_ptr<SprAbsFilter> filter(reader->read(trFile.c_str()));
  if( filter.get() == 0 ) {
    cerr << "Unable to read data from file " << trFile.c_str() << endl;
    return 2;
  }
  vector<string> vars;
  filter->vars(vars);
  cout << "Read data from file " << trFile.c_str() 
       << " for variables";
  for( unsigned int i=0;i<vars.size();i++ ) 
    cout << " \"" << vars[i].c_str() << "\"";
  cout << endl;
  cout << "Total number of points read: " << filter->size() << endl;
  const unsigned n0 = filter->ptsInClass(0);
  const unsigned n1 = filter->ptsInClass(1);
  cout << "Points in class 0: " << n0 << " 1: " << n1 << endl;

  // scale weights
  vector<double> origWeights;
  if( scaleWeights ) {
    filter->weights(origWeights);
    cout << "Signal weights are multiplied by " << sW << endl;
    filter->scaleWeights(1,sW);
  }

  // make optimization criterion
  SprTwoClassGiniIndex crit;

  // make decision tree
  bool doMerge = false;
  SprDecisionTree tree(filter.get(),&crit,nmin,doMerge,true);

  // train the tree on the original sample and save the original fom
  if( !tree.train(verbose) ) {
    cerr << "Unable to train decision tree." << endl;
    return 3;
  }
  double origFom = tree.fom();

  // save original class labels
  vector<pair<SprPoint*,int> > origLabels(filter->size());
  for( unsigned int i=0;i<filter->size();i++ ) {
    SprPoint* p = (*filter.get())[i];
    origLabels[i] = pair<SprPoint*,int>(p,p->class_);
  }

  // train decision tree on permuted replicas of the data
  cout << "Will perform " << cycles 
       << " toy experiments for GoF calculation." << endl;
  vector<double> fom;
  SprIntegerPermutator permu(filter->size(),seed);
  assert( (n0+n1) == filter->size() );
  for( unsigned int ic=0;ic<cycles;ic++ ) {
    // print out
    if( (ic%10) == 0 ) 
      cout << "Performing toy experiment " << ic << endl;

    // permute labels
    vector<unsigned> labels;
    permu.sequence(labels);
    for( unsigned int i=0;i<n0;i++ ) {
      unsigned ip = labels[i];
      (*filter.get())[ip]->class_ = 0;
    }
    for( unsigned int i=n0;i<n0+n1;i++ ) {
      unsigned ip = labels[i];
      (*filter.get())[ip]->class_ = 1;
    }

    // reset weights
    if( scaleWeights ) {
      filter->setPermanentWeights(origWeights);
      filter->scaleWeights(1,sW);
    }

    // reset tree
    tree.reset();

    // train the tree and save FOM
    if( !tree.train(verbose) ) continue;
    fom.push_back(tree.fom());
  }
  if( fom.empty() ) {
    cerr << "Failed to compute FOMs for any experiments." << endl;
    return 4;
  }

  // restore original class labels and weights
  // Not necessary here - just to remember that this needs to be done
  //  if data will be used in the future.
  for( unsigned int i=0;i<origLabels.size();i++ )
    origLabels[i].first->class_ = origLabels[i].second;
  if( scaleWeights ) filter->setPermanentWeights(origWeights);

  // compute fraction of experiments with worse FOM's
  stable_sort(fom.begin(),fom.end());
  vector<double>::iterator iter = find_if(fom.begin(),fom.end(),
					  bind2nd(greater<double>(),origFom));
  int below = iter - fom.begin();
  int above = fom.size() - below;
  cout << below << " experiments out of " << fom.size() 
       << " have better GoF values than the data." << endl;
  cout << "GoF=" << double(above)/double(fom.size()) << endl;

  // exit
  return 0;
}
