//$Id: SprVariableImportanceApp.cc,v 1.3 2007/10/29 22:10:40 narsky Exp $
//
// An executable to estimate the relative importance of variables.
// See notes in README (Variable Selection).
//

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRWFactory.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprMultiClassReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprCoordinateMapper.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedMultiClassLearner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierEvaluator.hh"

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <set>
#include <vector>
#include <memory>
#include <string>
#include <cassert>
#include <algorithm>
#include <utility>
#include <functional>

using namespace std;


// sorts in descending order
struct SVICmpSDDSecond
  : public binary_function<pair<string,pair<double,double> >,
			   pair<string,pair<double,double> >,bool> {
  bool operator()(const pair<string,pair<double,double> >& l, 
		  const pair<string,pair<double,double> >& r)
    const {
    return (l.second.first > r.second.first);
  }
};


void help(const char* prog) 
{
  cout << "Usage:  " << prog << " classifier_config_file"
       << " input_data_file" << endl;
  cout << "\t Options: " << endl;
  cout << "\t-h --- help                                        " << endl;
  cout << "\t-y list of input classes (see SprAbsFilter.hh)     " << endl;
  cout << "\t-a input ascii file mode (see SprSimpleReader.hh)  " << endl;
  cout << "\t-v verbose level (0=silent default,1,2)            " << endl;
  cout << "\t-m use multiclass learner                          " << endl;
  cout << "\t-n number of class permutations per variable (def=1)" << endl;
  cout << "\t-w scale all signal weights by this factor         " << endl;
  cout << "\t-V include only these input variables              " << endl;
  cout << "\t-z exclude input variables from the list           " << endl;
  cout << "\t-M map variable lists from trained classifiers onto" << endl;
  cout << "\t\t variables available in input data."               << endl;
  cout << "\t\t Variables must be listed in quotes and separated by commas." 
       << endl;
}


int main(int argc, char ** argv)
{
  // check command line
  if( argc < 3 ) {
    help(argv[0]);
    return 1;
  }

  // init
  int readMode = 0;
  int verbose = 0;
  bool scaleWeights = false;
  double sW = 1.;
  string includeList, excludeList;
  string inputClassesString;
  bool mapTrainedVars = false;
  int nPerm = 1;
  bool useMCLearner = false;

  // decode command line
  int c;
  extern char* optarg;
  extern int optind;
  while( (c = getopt(argc,argv,"hy:a:mn:v:w:V:z:M")) != EOF ) {
    switch( c )
      {
      case 'h' :
	help(argv[0]);
	return 1;
      case 'y' :
	inputClassesString = optarg;
	break;
      case 'a' :
	readMode = (optarg==0 ? 0 : atoi(optarg));
	break;
      case 'm' :
	useMCLearner = true;
	break;
      case 'n' :
	nPerm = (optarg==0 ? 1 : atoi(optarg));
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
      case 'M' :
	mapTrainedVars = true;
	break;
      }
  }

  // Must have 3 arguments on the command line
  string configFile     = argv[argc-2];
  string dataFile       = argv[argc-1];
  if( configFile.empty() ) {
    cerr << "No classifier configuration file is specified." << endl;
    return 1;
  }
  if( dataFile.empty() ) {
    cerr << "No input data file is specified." << endl;
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
    for( int i=0;i<includeVars[0].size();i++ ) 
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
    for( int i=0;i<excludeVars[0].size();i++ ) 
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

  // read input data from file
  auto_ptr<SprAbsFilter> filter(reader->read(dataFile.c_str()));
  if( filter.get() == 0 ) {
    cerr << "Unable to read data from file " << dataFile.c_str() << endl;
    return 2;
  }
  vector<string> vars;
  filter->vars(vars);
  cout << "Read data from file " << dataFile.c_str() << " for variables";
  for( int i=0;i<vars.size();i++ ) 
    cout << " \"" << vars[i].c_str() << "\"";
  cout << endl;
  cout << "Total number of points read: " << filter->size() << endl;

  // filter training data by class
  vector<SprClass> inputClasses;
  if( !filter->filterByClass(inputClassesString.c_str()) ) {
    cerr << "Cannot choose input classes for string " 
	 << inputClassesString << endl;
    return 2;
  }
  filter->classes(inputClasses);
  assert( inputClasses.size() > 1 );
  cout << "Training data filtered by class." << endl;
  for( int i=0;i<inputClasses.size();i++ ) {
    cout << "Points in class " << inputClasses[i] << ":   " 
	 << filter->ptsInClass(inputClasses[i]) << endl;
  }

  // scale weights
  if( scaleWeights ) {
    cout << "Signal weights are multiplied by " << sW << endl;
    filter->scaleWeights(inputClasses[1],sW);
  }

  // read classifier configuration
  auto_ptr<SprAbsTrainedClassifier> trained;
  auto_ptr<SprTrainedMultiClassLearner> mcTrained;
  if( useMCLearner ) {
    SprMultiClassReader multiReader;
    if( !multiReader.read(configFile.c_str()) ) {
      cerr << "Failed to read saved multi class learner from file "
           << configFile.c_str() << endl;
      return 3;
    }
    mcTrained.reset(multiReader.makeTrained());
    cout << "Read classifier " << mcTrained->name().c_str()
	 << " with dimensionality " << mcTrained->dim() << endl;
  }
  else {
    trained.reset(SprClassifierReader::readTrained(configFile.c_str(),
						   verbose));
    if( trained.get() == 0 ) {
      cerr << "Unable to read classifier configuration from file "
	   << configFile.c_str() << endl;
      return 3;
    }
    cout << "Read classifier " << trained->name().c_str()
	 << " with dimensionality " << trained->dim() << endl;
  }

  // get a list of trained variables
  vector<string> trainedVars;
  if( trained.get() != 0 )
    trained->vars(trainedVars);
  else
    mcTrained->vars(trainedVars);
  if( verbose > 0 ) {
    cout << "Variables:      " << endl;
    for( int j=0;j<trainedVars.size();j++ ) 
      cout << trainedVars[j].c_str() << " ";
    cout << endl;
  }

  // map trained-classifier variables onto data variables
  auto_ptr<SprCoordinateMapper> mapper;
  if( mapTrainedVars || 
      (trained.get()!=0 && trained->name()=="Combiner") ) {
    mapper.reset(SprCoordinateMapper::createMapper(trainedVars,vars));
    if( mapper.get() == 0 ) {
      cerr << "Unable to map trained classifier vars onto data vars." << endl;
      return 4;
    }
  }

  // call evaluator
  vector<SprClassifierEvaluator::NameAndValue> lossIncrease;
  if( !SprClassifierEvaluator::variableImportance(filter.get(),
						  trained.get(),
						  mcTrained.get(),
						  mapper.get(),
						  nPerm,
						  lossIncrease) ) {
    cerr << "Unable to estimate variable importance." << endl;
    return 5;
  }

  //
  // process computed loss
  //
  stable_sort(lossIncrease.begin(),lossIncrease.end(),SVICmpSDDSecond());
  cout << "==========================================================" << endl;
  char t [200];
  sprintf(t,"%35s      %15s","Variable","Change in loss");
  cout << t << endl;
  for( int d=0;d<lossIncrease.size();d++ ) {
    char s [200];
    sprintf(s,"%35s      %15.10f +- %15.10f",
	    lossIncrease[d].first.c_str(),
	    lossIncrease[d].second.first,lossIncrease[d].second.second);
    cout << s << endl;
  }
  cout << "==========================================================" << endl;

  // exit
  return 0;
}
