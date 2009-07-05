//$Id: SprOutputWriterApp.cc,v 1.6 2007/12/01 01:29:41 narsky Exp $


#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsWriter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDataFeeder.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRWFactory.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprCoordinateMapper.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedFisher.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedLogitR.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsVarTransformer.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprVarTransformerReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformerFilter.hh"

#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <set>
#include <vector>
#include <memory>
#include <string>
#include <cassert>
#include <algorithm>

using namespace std;


void help(const char* prog) 
{
  cout << "Usage:  " << prog << " list_of_classifier_config_files"
       << " input_data_file output_tuple_file" << endl;
  cout << "\t (List of files must be in quotes, separated by commas.)" << endl;
  cout << "\t Options: " << endl;
  cout << "\t-h --- help                                        " << endl;
  cout << "\t-y list of input classes (see SprAbsFilter.hh)     " << endl;
  cout << "\t-Q apply variable transformation saved in file     " << endl;
  cout << "\t-a input ascii file mode (see SprSimpleReader.hh)  " << endl;
  cout << "\t-A save output data in ascii instead of Root       " << endl;
  cout << "\t-K use 1-fraction of input data                    " << endl;
  cout << "\t\t This option is for consistency with other execs." << endl;
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
  cout << "\t-M map variable lists from trained classifiers onto" << endl;
  cout << "\t\t variables available in input data."               << endl;
  cout << "\t\t Variables must be listed in quotes and separated by commas." 
       << endl;
}


void cleanup(vector<SprAbsTrainedClassifier*>& trained) {
  for( unsigned int i=0;i<trained.size();i++ ) delete trained[i];
}


int main(int argc, char ** argv)
{
  // check command line
  if( argc < 4 ) {
    help(argv[0]);
    return 1;
  }

  // init
  int readMode = 0;
  SprRWFactory::DataType writeMode = SprRWFactory::Root;
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
  bool mapTrainedVars = false;
  bool split = false;
  double splitFactor = 0;
  string transformerFile;

 
  // decode command line
  int c;
  extern char* optarg;
  extern int optind;
  while( (c = getopt(argc,argv,"hy:Q:a:AK:v:w:t:C:p:sV:z:Z:M")) != EOF ) {
    switch( c )
      {
      case 'h' :
	help(argv[0]);
	return 1;
      case 'y' :
	inputClassesString = optarg;
	break;
      case 'Q' :
        transformerFile = optarg;
        break;
      case 'a' :
	readMode = (optarg==0 ? 0 : atoi(optarg));
	break;
      case 'A' :
	writeMode = SprRWFactory::Ascii;
	break;
      case 'K' :
        split = true;
        splitFactor = (optarg==0 ? 0 : atof(optarg));
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
      case 'M' :
	mapTrainedVars = true;
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
  unsigned int nTrained = configFiles[0].size();
  bool useClassifierNames 
    = (!classifierNames.empty() && !classifierNames[0].empty());
  if( useClassifierNames && (classifierNames[0].size()!=nTrained) ) {
    cerr << "Sizes of classifier name list and config file list do not match!"
	 << endl;
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

  // read input data from file
  auto_ptr<SprAbsFilter> filter(reader->read(dataFile.c_str()));
  if( filter.get() == 0 ) {
    cerr << "Unable to read data from file " << dataFile.c_str() << endl;
    return 2;
  }
  vector<string> vars;
  filter->vars(vars);
  cout << "Read data from file " << dataFile.c_str() << " for variables";
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
  cout << "Training data filtered by class." << endl;
  for( unsigned int i=0;i<inputClasses.size();i++ ) {
    cout << "Points in class " << inputClasses[i] << ":   " 
	 << filter->ptsInClass(inputClasses[i]) << endl;
  }

  // scale weights
  if( scaleWeights ) {
    cout << "Signal weights are multiplied by " << sW << endl;
    filter->scaleWeights(inputClasses[1],sW);
  }

  // apply transformation of variables to training and test data
  auto_ptr<SprAbsFilter> garbage_train;
  if( !transformerFile.empty() ) {
    SprVarTransformerReader transReader;
    const SprAbsVarTransformer* t = transReader.read(transformerFile.c_str());
    if( t == 0 ) {
      cerr << "Unable to read VarTransformer from file "
           << transformerFile.c_str() << endl;
      return 2;
    }
    SprTransformerFilter* t_train = new SprTransformerFilter(filter.get());
    bool replaceOriginalData = true;
    if( !t_train->transform(t,replaceOriginalData) ) {
      cerr << "Unable to apply VarTransformer to training data." << endl;
      return 2;
    }
    cout << "Variable transformation from file "
         << transformerFile.c_str() << " has been applied to data." << endl;
    garbage_train.reset(filter.release());
    filter.reset(t_train);
    filter->vars(vars);
  }

  // split data if desired
  auto_ptr<SprAbsFilter> valFilter;
  if( split ) {
    cout << "Splitting data with factor " << splitFactor << endl;
    vector<double> weights;
    SprData* splitted = filter->split(splitFactor,weights,false);
    if( splitted == 0 ) {
      cerr << "Unable to split data." << endl;
      return 2;
    }
    bool ownData = true;
    valFilter.reset(new SprEmptyFilter(splitted,weights,ownData));
    cout << "Data re-filtered:" << endl;
    for( unsigned int i=0;i<inputClasses.size();i++ ) {
      cout << "Points in class " << inputClasses[i] << ":   "
           << valFilter->ptsInClass(inputClasses[i]) << endl;
    }
  }
  else {
    valFilter.reset(filter.release());
  }

  // read classifier configuration
  vector<SprAbsTrainedClassifier*> trained(nTrained);
  vector<SprCoordinateMapper*> specificMappers(nTrained);
  for( unsigned int i=0;i<nTrained;i++ ) {

    // read classifier
    trained[i] 
      = SprClassifierReader::readTrained(configFiles[0][i].c_str(),verbose);
    if( trained[i] == 0 ) {
      cerr << "Unable to read classifier configuration from file "
	   << configFiles[0][i].c_str() << endl;
      cleanup(trained);
      return 3;
    }
    cout << "Read classifier " << trained[i]->name().c_str()
	 << " with dimensionality " << trained[i]->dim() << endl;

    // get a list of trained variables
    vector<string> trainedVars;
    trained[i]->vars(trainedVars);
    if( verbose > 0 ) {
      cout << "Variables:      " << endl;
      for( unsigned int j=0;j<trainedVars.size();j++ ) 
	cout << trainedVars[j].c_str() << " ";
      cout << endl;
    }

    // map trained-classifier variables onto data variables
    if( mapTrainedVars || trained[i]->name()=="Combiner" ) {
      specificMappers[i] 
	= SprCoordinateMapper::createMapper(trainedVars,vars);
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
  auto_ptr<SprAbsWriter> 
    tuple(SprRWFactory::makeWriter(writeMode,tupleName.c_str()));
  if( !tuple->init(tupleFile.c_str()) ) {
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
  for( unsigned int d=0;d<vars.size();d++ ) {
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
  SprDataFeeder feeder(valFilter.get(),tuple.get(),mapper);
  for( unsigned int i=0;i<nTrained;i++ ) {
    string useName;
    if( useClassifierNames ) 
      useName = classifierNames[0][i];
    else
      useName = trained[i]->name();
    feeder.addClassifier(trained[i],useName.c_str(),specificMappers[i]);
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
