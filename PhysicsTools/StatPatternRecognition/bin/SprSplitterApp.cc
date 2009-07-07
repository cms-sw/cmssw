//$Id: SprSplitterApp.cc,v 1.1 2007/12/01 01:29:41 narsky Exp $
/*
  This executable splits input data into training and test data,
  optionally converting them into a different format (e.g., Ascii 
  instead of Root).
*/

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsWriter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDataFeeder.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRWFactory.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"

#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <memory>

using namespace std;


void help(const char* prog) 
{
  cout << "Usage:  " << prog 
       << " input_data_file output_training_data_file output_test_data_file" 
       << endl;
  cout << "\t Options: " << endl;
  cout << "\t-h --- help                                        " << endl;
  cout << "\t-a input ascii file mode (see SprSimpleReader.hh)  " << endl;
  cout << "\t-A save output data in ascii instead of Root       " << endl;
  cout << "\t-y list of input classes (see SprAbsFilter.hh)     " << endl;
  cout << "\t-v verbose level (0=silent default,1,2)            " << endl;
  cout << "\t-V include only these input variables              " << endl;
  cout << "\t-z exclude input variables from the list           " << endl;
  cout << "\t\t Variables must be listed in quotes and separated by commas." 
       << endl;
  cout << "\t-K keep the specified fraction in input data       " << endl;
  cout << "\t\t If no fraction specified, 0.5 is assumed.       " << endl;
  cout << "\t-S random seed used for splitting.                 " << endl;
  cout << "\t\t If none, puts K into training and (1-K) into test data." 
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
  int readMode = 0;
  SprRWFactory::DataType writeMode = SprRWFactory::Root;
  int verbose = 0;
  string outFile;
  string includeList, excludeList;
  string inputClassesString;
  double splitFactor = 0.5;
  int seed = 0;
  bool splitRandomize = false;

  // decode command line
  int c;
  extern char* optarg;
  //  extern int optind;
  while( (c = getopt(argc,argv,"ha:Ay:v:V:z:K:S:")) != EOF ) {
    switch( c )
      {
      case 'h' :
	help(argv[0]);
	return 1;
       case 'a' :
	readMode = (optarg==0 ? 0 : atoi(optarg));
	break;
      case 'A' :
	writeMode = SprRWFactory::Ascii;
	break;
      case 'y' :
	inputClassesString = optarg;
	break;
      case 'v' :
	verbose = (optarg==0 ? 0 : atoi(optarg));
	break;
      case 'V' :
	includeList = optarg;
	break;
      case 'z' :
	excludeList = optarg;
	break;
      case 'K' :
	splitFactor = (optarg==0 ? 0.5 : atof(optarg));
	break;
      case 'S' :
	splitRandomize = true;
	seed = (optarg==0 ? 0 : atoi(optarg));
	break;
      }
  }

  // arguments
  string inputFile = argv[argc-3];
  if( inputFile.empty() ) {
    cerr << "No input file is specified." << endl;
    return 1;
  }
  string trainFile = argv[argc-2];
  if( trainFile.empty() ) {
    cerr << "No training file is specified." << endl;
    return 1;
  }
  string testFile = argv[argc-1];
  if( testFile.empty() ) {
    cerr << "No test file is specified." << endl;
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
      cerr << "Unable to include variables in input set." << endl;
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
      cerr << "Unable to exclude variables from input set." << endl;
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
  auto_ptr<SprAbsFilter> filter(reader->read(inputFile.c_str()));
  if( filter.get() == 0 ) {
    cerr << "Unable to read data from file " << inputFile.c_str() << endl;
    return 2;
  }
  vector<string> vars;
  filter->vars(vars);
  cout << "Read data from file " << inputFile.c_str() << " for variables";
  for( unsigned int i=0;i<vars.size();i++ ) 
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
  cout << "Input data filtered by class." << endl;
  for( unsigned int i=0;i<inputClasses.size();i++ ) {
    cout << "Points in class " << inputClasses[i] << ":   " 
	 << filter->ptsInClass(inputClasses[i]) << endl;
  }

  // split data
  cout << "Splitting input data with factor " << splitFactor << endl;
  vector<double> weights;
  SprData* splitted = filter->split(splitFactor,weights,splitRandomize,seed);
  if( splitted == 0 ) {
    cerr << "Unable to split input data." << endl;
    return 2;
  }
  bool ownData = true;
  auto_ptr<SprAbsFilter> valFilter(new SprEmptyFilter(splitted,
						      weights,ownData));
  cout << "Data re-filtered:" << endl;
  cout << "Training data:" << endl;
  for( unsigned int i=0;i<inputClasses.size();i++ ) {
    cout << "Points in class " << inputClasses[i] << ":   " 
	 << filter->ptsInClass(inputClasses[i]) << endl;
  }
  cout << "Test data:" << endl;
  for( unsigned int i=0;i<inputClasses.size();i++ ) {
    cout << "Points in class " << inputClasses[i] << ":   " 
	 << valFilter->ptsInClass(inputClasses[i]) << endl;
  }

  // make a writer
  auto_ptr<SprAbsWriter> trainTuple(SprRWFactory::makeWriter(writeMode,
							     "training"));
  if( !trainTuple->init(trainFile.c_str()) ) {
    cerr << "Unable to open output file " << trainFile.c_str() << endl;
    return 3;
  }
  auto_ptr<SprAbsWriter> testTuple(SprRWFactory::makeWriter(writeMode,
							    "test"));
  if( !testTuple->init(testFile.c_str()) ) {
    cerr << "Unable to open output file " << testFile.c_str() << endl;
    return 3;
  }

  // feed
  SprDataFeeder feeder(filter.get(),trainTuple.get());
  if( !feeder.feed(1000) ) {
    cerr << "Cannot feed data into file " << trainFile.c_str() << endl;
    return 4;
  }
  SprDataFeeder valFeeder(valFilter.get(),testTuple.get());
  if( !valFeeder.feed(1000) ) {
    cerr << "Cannot feed data into file " << testFile.c_str() << endl;
    return 4;
  }

  // exit
  return 0;
}
