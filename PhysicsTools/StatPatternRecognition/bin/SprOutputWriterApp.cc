//$Id: SprOutputWriterApp.cc,v 1.1 2007/02/05 21:49:46 narsky Exp $


#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprSimpleReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDataFeeder.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprMyWriter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedFisher.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedLogitR.hh"

#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <set>
#include <vector>
#include <memory>
#include <string>
#include <cassert>

using namespace std;


void help(const char* prog) 
{
  cout << "Usage:  " << prog << " list_of_classifier_config_files"
       << " input_data_file output_tuple_file" << endl;
  cout << "\t (List of files must be in quotes, separated by commas.)" << endl;
  cout << "\t Options: " << endl;
  cout << "\t-h --- help                                        " << endl;
  cout << "\t-y list of input classes (see SprAbsFilter.hh)     " << endl;
  cout << "\t-a input ascii file mode (see SprSimpleReader.hh)  " << endl;
  cout << "\t-v verbose level (0=silent default,1,2)            " << endl;
  cout << "\t-w scale all signal weights by this factor         " << endl;
  cout << "\t-t output tuple name (default=data)                " << endl;
  cout << "\t-C output classifier names (in quotes, separated by commas)" 
       << endl;
  cout << "\t-p feeder print-out frequency (default=1000 events)" << endl;
  cout << "\t-s use output in range (-infty,+infty) instead of [0,1]" << endl;
  cout << "\t-V include only these input variables              " << endl;
  cout << "\t-z exclude input variables from the list           " << endl;
  cout << "\t-Z exclude input variables from the list, "
       << "but put them in the output file " << endl;
  cout << "\t\t Variables must be listed in quotes and separated by commas." 
       << endl;
}


void cleanup(vector<SprAbsTrainedClassifier*>& trained) {
  for( int i=0;i<trained.size();i++ ) delete trained[i];
}


int main(int argc, char ** argv)
{
  // check command line
  if( argc < 4 ) {
    help(argv[0]);
    return 1;
  }

  // init
  int readMode = 1;
  int verbose = 0;
  bool scaleWeights = false;
  double sW = 1.;
  bool useStandard = false;
  string tupleName;
  string classifierNameList;
  string includeList, excludeList;
  string inputClassesString;
  int nPrintOut = 1000;
  string stringVarsDoNotFeed;
  
  // decode command line
  int c;
  extern char* optarg;
  extern int optind;
  while( (c = getopt(argc,argv,"hy:a:v:w:t:C:p:sV:z:Z:")) != EOF ) {
    switch( c )
      {
      case 'h' :
	help(argv[0]);
	return 1;
      case 'y' :
	inputClassesString = optarg;
	break;
      case 'a' :
	readMode = (optarg==0 ? 1 : atoi(optarg));
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
      case 't' :
	tupleName = optarg;
	break;
      case 'C' :
	classifierNameList = optarg;
	break;
      case 'p' :
	nPrintOut = (optarg==0 ? 1000 : atoi(optarg));
	break;
      case 's' :
	useStandard = true;
	break;
      case 'V' :
	includeList = optarg;
	break;
      case 'z' :
	excludeList = optarg;
	break;
      case 'Z' :
	stringVarsDoNotFeed = optarg;
	break;
      }
  }

  // Must have 3 arguments on the command line
  string configFileList = argv[argc-3];
  string dataFile       = argv[argc-2];
  string tupleFile      = argv[argc-1];
  if( configFileList.empty() ) {
    cerr << "No classifier configuration files are specified." << endl;
    return 1;
  }
  if( dataFile.empty() ) {
    cerr << "No input data file is specified." << endl;
    return 1;
  }
  if( tupleFile.empty() ) {
    cerr << "No output tuple file is specified." << endl;
    return 1;
  }

  // get classifier names and config files
  vector<vector<string> > classifierNames, configFiles;
  SprStringParser::parseToStrings(classifierNameList.c_str(),classifierNames);
  SprStringParser::parseToStrings(configFileList.c_str(),configFiles);
  if( configFiles.empty() || configFiles[0].empty() ) {
    cerr << "Unable to parse config file list: " 
	 << configFileList.c_str() << endl;
    return 1;
  }
  int nTrained = configFiles[0].size();
  bool useClassifierNames 
    = (!classifierNames.empty() && !classifierNames[0].empty());
  if( useClassifierNames && (classifierNames[0].size()!=nTrained) ) {
    cerr << "Sizes of classifier name list and config file list do not match!"
	 << endl;
    return 1;
  }

  // make reader
  SprSimpleReader reader(readMode);

  // include variables
  set<string> includeSet;
  if( !includeList.empty() ) {
    vector<vector<string> > includeVars;
    SprStringParser::parseToStrings(includeList.c_str(),includeVars);
    assert( !includeVars.empty() );
    for( int i=0;i<includeVars[0].size();i++ ) 
      includeSet.insert(includeVars[0][i]);
    if( !reader.chooseVars(includeSet) ) {
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
    if( !reader.chooseAllBut(excludeSet) ) {
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
  auto_ptr<SprAbsFilter> filter(reader.read(dataFile.c_str()));
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
  vector<SprAbsTrainedClassifier*> trained(nTrained);
  for( int i=0;i<nTrained;i++ ) {
    trained[i] 
      = SprClassifierReader::readTrained(configFiles[0][i].c_str(),verbose);
    if( trained[i] == 0 ) {
      cerr << "Unable to read classifier configuration from file "
	   << configFiles[0][i].c_str() << endl;
      cleanup(trained);
      return 3;
    }

    // switch classifier output range
    if( useStandard ) {
      if(      trained[i]->name() == "AdaBoost" ) {
	SprTrainedAdaBoost* specific 
	  = static_cast<SprTrainedAdaBoost*>(trained[i]);
	specific->useStandard();
      }
      else if( trained[i]->name() == "Fisher" ) {
	SprTrainedFisher* specific 
	  = static_cast<SprTrainedFisher*>(trained[i]);
	specific->useStandard();
      }
      else if( trained[i]->name() == "LogitR" ) {
	SprTrainedLogitR* specific 
	  = static_cast<SprTrainedLogitR*>(trained[i]);
	specific->useStandard();
      }
    }
  }

  // make tuple
  if( tupleName.empty() ) tupleName = "data";
  SprMyWriter tuple(tupleName.c_str());
  if( !tuple.init(tupleFile.c_str()) ) {
    cerr << "Unable to open output file " << tupleFile.c_str() << endl;
    cleanup(trained);
    return 5;
  }

  // determine if certain variables are to be excluded from usage,
  // but included in the output storage file (-Z option)
  string printVarsDoNotFeed;
  vector<vector<string> > varsDoNotFeed;
  SprStringParser::parseToStrings(stringVarsDoNotFeed.c_str(),varsDoNotFeed);
  vector<unsigned> mapper;
  for( int d=0;d<vars.size();d++ ) {
    if( varsDoNotFeed.empty() ||
        (find(varsDoNotFeed[0].begin(),varsDoNotFeed[0].end(),vars[d])
	 ==varsDoNotFeed[0].end()) ) {
      mapper.push_back(d);
    }
    else {
      printVarsDoNotFeed += ( printVarsDoNotFeed.empty() ? "" : ", " );
      printVarsDoNotFeed += vars[d];
    }
  }
  if( !printVarsDoNotFeed.empty() ) {
    cout << "The following variables are not used in the algorithm, " 
         << "but will be included in the output file: " 
         << printVarsDoNotFeed.c_str() << endl;
  }

  // feed data into tuple
  SprDataFeeder feeder(filter.get(),&tuple,mapper);
  for( int i=0;i<nTrained;i++ ) {
    string useName;
    if( useClassifierNames ) 
      useName = classifierNames[0][i];
    else
      useName = trained[i]->name();
    feeder.addClassifier(trained[i],useName.c_str());
  }
  if( !feeder.feed(nPrintOut) ) {
    cerr << "Cannot feed data into file " << tupleFile.c_str() << endl;
    cleanup(trained);
    return 6;
  }

  // exit
  cleanup(trained);
  return 0;
}
