//$Id: SprCombinerApp.cc,v 1.3 2007/11/12 06:19:11 narsky Exp $


#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprCombiner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsWriter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRWFactory.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerBootstrap.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAverageLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformation.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBagger.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStdBackprop.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsVarTransformer.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprVarTransformerReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformerFilter.hh"


#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <string>
#include <cassert>
#include <map>
#include <utility>

using namespace std;


void help(const char* prog) 
{
  cout << "Usage:  " << prog
       << " list_of_input_config_subclassifier_files"
       << " input_config_file_for_global_classifier" 
       << " input_data_file" << endl;
  cout << "\t Options: " << endl;
  cout << "\t-h --- help                                        " << endl;
  cout << "\t-y list of input classes (see SprAbsFilter.hh)     " << endl;
  cout << "\t-Q apply variable transformation saved in file     " << endl;
  cout << "\t-a input ascii file mode (see SprSimpleReader.hh)  " << endl;
  cout << "\t-v verbose level (0=silent default,1,2)            " << endl;
  cout << "\t-w scale all signal weights by this factor         " << endl;
  cout << "\t-f save trained classifier configuration to file   " << endl;
  cout << "\t-K keep this fraction in training set and          " << endl;
  cout << "\t\t put the rest into validation set                " << endl;
  cout << "\t-D randomize training set split-up                 " << endl;
  cout << "\t-t read validation/test data from a file           " << endl;
  cout << "\t\t (must be in same format as input data!!!        " << endl;
  cout << "\t-d frequency of print-outs for validation data     " << endl;
}


void prepareExit(vector<SprAbsTwoClassCriterion*>& criteria,
                 vector<SprIntegerBootstrap*>& bstraps,
		 vector<SprAbsClassifier*>& classifiers)
{
  for( unsigned int i=0;i<criteria.size();i++ ) delete criteria[i];
  for( unsigned int i=0;i<classifiers.size();i++ ) delete classifiers[i];
  for( unsigned int i=0;i<bstraps.size();i++ ) delete bstraps[i];
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
  int verbose = 0;
  bool scaleWeights = false;
  double sW = 1.;
  // bool useStandard = false;
  string inputClassesString;
  string valFile;
  unsigned valPrint = 0;
  string outFile;
  bool split = false;
  double splitFactor = 0;
  bool splitRandomize = false;
  string transformerFile;

  // decode command line
  int c;
  extern char* optarg;
  extern int optind;
  while( (c = getopt(argc,argv,"hy:a:v:w:f:K:Dt:d:")) != EOF ) {
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
      case 'v' :
	verbose = (optarg==0 ? 0 : atoi(optarg));
	break;
      case 'w' :
	if( optarg != 0 ) {
	  scaleWeights = true;
	  sW = atof(optarg);
	}
	break;
      case 'f' :
	outFile = optarg;
        break;
      case 'K' :
	split = true;
	splitFactor = (optarg==0 ? 0 : atof(optarg));
	break;
      case 'D' :
	splitRandomize = true;
	break;
      case 't' :
        valFile = optarg;
        break;
      case 'd' :
        valPrint = (optarg==0 ? 0 : atoi(optarg));
        break;
      }
  }

  // Must have 3 arguments on the command line
  string trainFile = argv[argc-1];
  if( trainFile.empty() ) {
    cerr << "No input data file is specified." << endl;
    return 1;
  }
  cout << "Will read input data from file " << trainFile.c_str() << endl;
  string configFile = argv[argc-2];
  if( configFile.empty() ) {
    cerr << "No config file for the global classifier specified." << endl;
    return 1;
  }
  cout << "Will read global classifier config from file "
       << configFile.c_str() << endl;
  string subConfigList = argv[argc-3];
  if( subConfigList.empty() ) {
    cerr << "No config file list found for sub-classifiers." << endl;
    return 1;
  }
  cout << "Will read sub-classifier configs from files " 
       << subConfigList.c_str() << endl;

  // check options
  if( subConfigList.empty() || configFile.empty() ) {
    cerr << "User must specify combiner configuration." << endl;
    return 1;
  }

  // get classifier names and config files
  vector<vector<string> > subConfigFiles;
  SprStringParser::parseToStrings(subConfigList.c_str(),subConfigFiles);
  bool useSubConfig 
    = ( !subConfigFiles.empty() && !subConfigFiles[0].empty() );
  if( !useSubConfig ) {
    cerr << "Unable to process list of sub-classifier config files." << endl;
    return 1;
  }
  int nTrained = subConfigFiles[0].size();

  // make reader
  SprRWFactory::DataType inputType 
    = ( readMode==0 ? SprRWFactory::Root : SprRWFactory::Ascii );
  auto_ptr<SprAbsReader> reader(SprRWFactory::makeReader(inputType,readMode));

  // read input data from file
  auto_ptr<SprAbsFilter> filter(reader->read(trainFile.c_str()));
  if( filter.get() == 0 ) {
    cerr << "Unable to read data from file " << trainFile.c_str() << endl;
    return 2;
  }
  vector<string> vars;
  filter->vars(vars);
  cout << "Read data from file " << trainFile.c_str() << " for variables";
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

  // read test data
  auto_ptr<SprAbsFilter> valFilter;
  if( split && !valFile.empty() ) {
    cerr << "Unable to split training data and use validation data " 
	 << "from a separate file." << endl;
    return 2;
  }
  if( split ) {
    cout << "Splitting training data with factor " << splitFactor << endl;
    if( splitRandomize )
      cout << "Will use randomized splitting." << endl;
    vector<double> weights;
    SprData* splitted = filter->split(splitFactor,weights,splitRandomize);
    if( splitted == 0 ) {
      cerr << "Unable to split training data." << endl;
      return 2;
    }
    bool ownData = true;
    valFilter.reset(new SprEmptyFilter(splitted,weights,ownData));
    cout << "Training data re-filtered:" << endl;
    for( unsigned int i=0;i<inputClasses.size();i++ ) {
      cout << "Points in class " << inputClasses[i] << ":   " 
	   << filter->ptsInClass(inputClasses[i]) << endl;
    }
  }
  if( !valFile.empty() ) {
    // make test reader
    auto_ptr<SprAbsReader> 
      valReader(SprRWFactory::makeReader(inputType,readMode));
    
    // read test data from file
    valFilter.reset(valReader->read(valFile.c_str()));
    if( valFilter.get() == 0 ) {
      cerr << "Unable to read data from file " << valFile.c_str() << endl;
      return 2;
    }
    vector<string> valVars;
    valFilter->vars(valVars);
    cout << "Read data from file " << valFile.c_str() << " for variables";
    for( unsigned int i=0;i<valVars.size();i++ ) 
      cout << " \"" << valVars[i].c_str() << "\"";
    cout << endl;
    cout << "Total number of points read: " << valFilter->size() << endl;
    
    // filter training data by class
    if( !valFilter->filterByClass(inputClassesString.c_str()) ) {
      cerr << "Cannot choose input classes for string " 
	   << inputClassesString << endl;
      return 2;
    }
    valFilter->classes(inputClasses);
    assert( inputClasses.size() > 1 );
    cout << "Validation data filtered by class." << endl;
    for( unsigned int i=0;i<inputClasses.size();i++ ) {
      cout << "Points in class " << inputClasses[i] << ":   " 
	   << valFilter->ptsInClass(inputClasses[i]) << endl;
    }
    
    // scale weights
    if( scaleWeights ) {
      cout << "Signal weights are multiplied by " << sW << endl;
      valFilter->scaleWeights(inputClasses[1],sW);
    }
  }

  // apply transformation of variables to training and test data
  auto_ptr<SprAbsFilter> garbage_train, garbage_valid;
  if( !transformerFile.empty() ) {
    SprVarTransformerReader transReader;
    const SprAbsVarTransformer* t = transReader.read(transformerFile.c_str());
    if( t == 0 ) {
      cerr << "Unable to read VarTransformer from file "
           << transformerFile.c_str() << endl;
      return 2;
    }
    SprTransformerFilter* t_train = new SprTransformerFilter(filter.get());
    SprTransformerFilter* t_valid = 0;
    if( valFilter.get() != 0 )
      t_valid = new SprTransformerFilter(valFilter.get());
    bool replaceOriginalData = true;
    if( !t_train->transform(t,replaceOriginalData) ) {
      cerr << "Unable to apply VarTransformer to training data." << endl;
      return 2;
    }
    if( t_valid!=0 && !t_valid->transform(t,replaceOriginalData) ) {
      cerr << "Unable to apply VarTransformer to validation data." << endl;
      return 2;
    }
    cout << "Variable transformation from file "
         << transformerFile.c_str() << " has been applied to "
         << "training and validation data." << endl;
    garbage_train.reset(filter.release());
    garbage_valid.reset(valFilter.release());
    filter.reset(t_train);
    valFilter.reset(t_valid);
  }

  //
  // make combiner
  //
  SprCombiner combiner(filter.get());

  //
  // read classifier configuration
  //
  for( int ic=0;ic<nTrained;ic++ ) {

    // open file
    string fname = subConfigFiles[0][ic];
    ifstream file(fname.c_str());
    if( !file ) {
      cerr << "Unable to open file " << fname.c_str() << endl;
      return 3;
    }
    cout << "Reading classifier configuration from file " 
	 << fname.c_str() << endl;

    // get path to sub-classifier file
    string line;
    unsigned nLine = 1;
    if( !getline(file,line) ) {
      cerr << "Cannot read line " << nLine 
	   << " from file " << fname.c_str() << endl;
      return 3;
    }
    string pathToConfig, dummy;
    istringstream istpath(line);
    istpath >> dummy >> pathToConfig;
    if( pathToConfig.empty() ) {
      cerr << "Path to classifier not specified in file "
	   << fname.c_str() << endl;
    }

    // read designated classifier name
    nLine++;
    if( !getline(file,line) ) {
      cerr << "Cannot read line " << nLine 
	   << " from file " << fname.c_str() << endl;
      return 3;
    }
    string subName;
    istringstream istname(line);
    istname >> dummy >> subName;
    if( subName.empty() ) {
      cout << "Name for classifier " << ic << " not specified." 
	   << " Will use the default." << endl;
    }

    // read default value
    nLine++;
    if( !getline(file,line) ) {
      cerr << "Cannot read line " << nLine 
	   << " from file " << fname.c_str() << endl;
      return 3;
    }
    double defaultValue = 0;
    istringstream istdefault(line);
    istdefault >> dummy >> defaultValue;
    cout << "Will use default response " << defaultValue 
	 << " for classifier " << ic << endl;

    // read number of constraints
    nLine++;
    if( !getline(file,line) ) {
      cerr << "Cannot read line " << nLine 
	   << " from file " << fname.c_str() << endl;
      return 3;
    }
    unsigned nConstraints = 0;
    istringstream istconst(line);
    istconst >> dummy >> nConstraints;
    cout << "Will use " << nConstraints << " constraints "
	 << "for classifier " << ic << endl;

    // read constraints
    map<string,SprCut> constraints;
    for( unsigned int j=0;j<nConstraints;j++ ) {
      nLine++;
      if( !getline(file,line) ) {
	cerr << "Cannot read line " << nLine 
	     << " from file " << fname.c_str() << endl;
	return 3;
      }
      istringstream ist(line);
      string varName;
      unsigned nCut = 0;
      ist >> varName >> nCut;
      if( varName.empty() ) {
	cerr << "Unable to read variable name on line " << nLine
	     << " in file " << fname.c_str() << endl;
      }
      SprCut cut;
      double xa(0), xb(0);
      for( unsigned k=0;k<nCut;k++ ) {
	ist >> xa >> xb;
	cut.push_back(SprInterval(xa,xb));
      }
      cout << "Applying constraint on variable " << varName.c_str()
	   << " for classifier " << ic << " : ";
      for( unsigned int k=0;k<cut.size();k++ ) 
	cout << cut[k].first << " " << cut[k].second << "   | ";
      cout << endl;
      constraints.insert(pair<const string,SprCut>(varName,cut));
    }

    // read classifier
    SprAbsTrainedClassifier* trained
      = SprClassifierReader::readTrained(pathToConfig.c_str(),verbose);
    if( trained == 0 ) {
      cerr << "Unable to read classifier configuration from file "
	   << pathToConfig.c_str() << endl;
      return 3;
    }
    cout << "Read classifier " << trained->name().c_str()
	 << " with dimensionality " << trained->dim() << endl;

    // get a list of trained variables
    vector<string> trainedVars;
    trained->vars(trainedVars);
    cout << "Variables:      " << endl;
    for( unsigned int j=0;j<trainedVars.size();j++ ) 
      cout << trainedVars[j].c_str() << " ";
    cout << endl;

    // add classifier to the combiner
    bool ownTrained = true;
    if( !combiner.addTrained(trained,subName.c_str(),constraints,
			     defaultValue,ownTrained) ) {
      cerr << "Unable to add trained classifier " << ic 
	   << " to combiner." << endl;
      return 3;
    }
  }

  // close trained classifier list
  if( !combiner.closeClassifierList() ) {
    cerr << "Unable to close the trained classifier list for the combiner." 
	 << endl;
    return 4;
  }
  SprEmptyFilter* features = combiner.features();

  //
  // read trainable classifier config
  //
  ifstream file(configFile.c_str());
  if( !file ) {
    cerr << "Unable to open file " << configFile.c_str() << endl;
    return 5;
  }
  cout << "Reading classifier configuration from file " 
       << configFile.c_str() << endl;
  unsigned nLine = 0;
  bool discreteTree = false;
  bool mixedNodesTree = false;
  bool fastSort = false;
  bool readOneEntry = true;
  vector<SprAbsTwoClassCriterion*> crits;
  vector<SprIntegerBootstrap*> bstraps;
  vector<SprAbsClassifier*> destroyC;
  vector<SprCCPair> useC;
  if( !SprClassifierReader::readTrainableConfig(file,nLine,features,
						discreteTree,mixedNodesTree,
						fastSort,crits,
						bstraps,destroyC,
						useC,readOneEntry) ) {
    cerr << "Unable to read trainable classifier config from file " 
	 << configFile.c_str() << endl;
    prepareExit(crits,bstraps,destroyC);
    return 5;
  }
  SprAbsClassifier* trainable = useC[0].first;
  cout << "Setting trainable classifier for combiner to " 
       << trainable->name() << endl;
  combiner.setTrainable(trainable);

  // make per-event loss
  auto_ptr<SprAverageLoss> loss;
  if( valFilter.get()!=0 && valPrint>0 ) {
    string trainableName = trainable->name();
    if( trainableName=="AdaBoost" || trainableName=="Bagger"
	|| trainableName=="ArcE4" || trainableName=="StdBackprop" ) {
      cout << "For simplicity only quadratic loss can be displayed." << endl;
      if( trainableName=="AdaBoost" ) {
	loss.reset(new SprAverageLoss(&SprLoss::quadratic,
				      &SprTransformation::logit));
      }
      else {
	loss.reset(new SprAverageLoss(&SprLoss::quadratic));
      }
      if( trainableName=="AdaBoost" ) {
	if( !static_cast<SprAdaBoost*>(trainable)
	    ->setValidation(features,valPrint,loss.get()) ) {
	  cerr << "Unable to set validation loss." << endl;
	  return 6;
	}
      }
      else if( trainableName=="Bagger" || trainableName=="ArcE4" ) {
	if( !static_cast<SprBagger*>(trainable)
	    ->setValidation(features,valPrint,0,loss.get()) ) {
	  cerr << "Unable to set validation loss." << endl;
	  return 6;
	}
      }
      else if( trainableName=="StdBackprop" ) {
	if( !static_cast<SprStdBackprop*>(trainable)
	    ->setValidation(features,valPrint,loss.get()) ) {
	  cerr << "Unable to set validation loss." << endl;
	  return 6;
	}
      }
    }
  }

  // train
  if( !combiner.train(verbose) ) {
    cerr << "Combiner finished with error." << endl;
    prepareExit(crits,bstraps,destroyC);
    return 7;
  }

  // save trained combiner
  if( !outFile.empty() ) {
    if( !combiner.store(outFile.c_str()) ) {
      cerr << "Cannot store Combiner to file " << outFile.c_str() << endl;
      prepareExit(crits,bstraps,destroyC);
      return 8;
    }
  }

  // exit
  prepareExit(crits,bstraps,destroyC);
  return 0;
}
