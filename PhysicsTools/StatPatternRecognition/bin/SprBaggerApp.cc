//$Id: SprBaggerApp.cc,v 1.5 2007/11/12 06:19:11 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBagger.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprArcE4.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedBagger.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsWriter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRWFactory.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerBootstrap.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAverageLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformation.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsVarTransformer.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprVarTransformerReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformerFilter.hh"

#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <string>
#include <memory>

using namespace std;



void help(const char* prog) 
{
  cout << "Usage:  " << prog 
       << " training_data_file"
       << " file_of_classifier_parameters(see booster.config for syntax)" 
       << endl;
  cout << "\t Options: " << endl;
  cout << "\t-h --- help                                        " << endl;
  cout << "\t-a input ascii file mode (see SprSimpleReader.hh)  " << endl;
  cout << "\t-A save output data in ascii instead of Root       " << endl;
  cout << "\t-n number of Bagger training cycles                " << endl;
  cout << "\t-y list of input classes (see SprAbsFilter.hh)     " << endl;
  cout << "\t-Q apply variable transformation saved in file     " << endl;
  cout << "\t-b use a version of Breiman's arc-x4 algorithm     " << endl;
  cout << "\t-g per-event loss for (cross-)validation           " << endl;
  cout << "\t\t 1 - quadratic loss (y-f(x))^2                   " << endl;
  cout << "\t\t 2 - exponential loss exp(-y*f(x))               " << endl;
  cout << "\t-m replace data values below this cutoff with medians" << endl;
  cout << "\t-v verbose level (0=silent default,1,2)            " << endl;
  cout << "\t-f store trained AdaBoost to file                  " << endl;
  cout << "\t-r resume training for Bagger stored in file       " << endl;
  cout << "\t-K keep this fraction in training set and          " << endl;
  cout << "\t\t put the rest into validation set                " << endl;
  cout << "\t-D randomize training set split-up                 " << endl;
  cout << "\t-G generate seed from time of day for bootstrap    " << endl;
  cout << "\t\t (this option is required for parallelization)   " << endl;
  cout << "\t-t read validation/test data from a file           " << endl;
  cout << "\t\t (must be in same format as input data!!!        " << endl;
  cout << "\t-d frequency of print-outs for validation data     " << endl;
  cout << "\t-w scale all signal weights by this factor         " << endl;
  cout << "\t-V include only these input variables              " << endl;
  cout << "\t-z exclude input variables from the list           " << endl;
  cout << "\t\t Variables must be listed in quotes and separated by commas." 
       << endl;
}


void prepareExit(vector<SprAbsTwoClassCriterion*>& criteria,
		 vector<SprAbsClassifier*>& classifiers,
		 vector<SprIntegerBootstrap*>& bstraps) 
{
  for( unsigned int  i=0;i<criteria.size();i++ ) delete criteria[i];
  for( unsigned int  i=0;i<classifiers.size();i++ ) delete classifiers[i];
  for( unsigned int  i=0;i<bstraps.size();i++ ) delete bstraps[i];
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
  SprRWFactory::DataType writeMode = SprRWFactory::Root;
  unsigned cycles = 0;
  int verbose = 0;
  string outFile;
  string resumeFile;
  string valFile;
  unsigned valPrint = 0;
  bool scaleWeights = false;
  double sW = 1.;
  bool setLowCutoff = false;
  double lowCutoff = 0;
  string includeList, excludeList;
  int iLoss = 1;
  string inputClassesString;
  bool useArcE4 = false;
  bool split = false;
  double splitFactor = 0;
  bool splitRandomize = false;
  bool initBootstrapFromTimeOfDay = false;
  string transformerFile;

  // decode command line
  int c;
  extern char* optarg;
  //  extern int optind;
  while((c = getopt(argc,argv,"ha:An:y:Q:bg:m:v:f:r:K:DGt:d:w:V:z:")) != EOF ) {
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
      case 'n' :
	cycles = (optarg==0 ? 1 : atoi(optarg));
	break;
      case 'y' :
	inputClassesString = optarg;
	break;
      case 'Q' :
        transformerFile = optarg;
        break;
      case 'b' :
        useArcE4 = true;
        break;
      case 'g' :
        iLoss = (optarg==0 ? 1 : atoi(optarg));
        break;
      case 'm' :
	if( optarg != 0 ) {
	  setLowCutoff = true;
	  lowCutoff = atof(optarg);
	}
	break;
      case 'v' :
	verbose = (optarg==0 ? 0 : atoi(optarg));
	break;
      case 'f' :
	outFile = optarg;
	break;
      case 'r' :
	resumeFile = optarg;
	break;
      case 'K' :
	split = true;
	splitFactor = (optarg==0 ? 0 : atof(optarg));
	break;
      case 'D' :
	splitRandomize = true;
	break;
      case 'G' :
	initBootstrapFromTimeOfDay = true;
	break;
      case 't' :
	valFile = optarg;
	break;
      case 'd' :
	valPrint = (optarg==0 ? 0 : atoi(optarg));
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

  // Must have 2 arguments after all options.
  string trFile = argv[argc-2];
  string configFile = argv[argc-1]; 
  if( trFile.empty() ) {
    cerr << "No training file is specified." << endl;
    return 1;
  }
  if( configFile.empty() ) {
    cerr << "No classifier configuration file specified." << endl;
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
    for( unsigned int  i=0;i<includeVars[0].size();i++ ) 
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
    for( unsigned int  i=0;i<excludeVars[0].size();i++ ) 
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
  cout << "Read data from file " << trFile.c_str() << " for variables";
  for( unsigned int  i=0;i<vars.size();i++ ) 
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
  for( unsigned int  i=0;i<inputClasses.size();i++ ) {
    cout << "Points in class " << inputClasses[i] << ":   " 
	 << filter->ptsInClass(inputClasses[i]) << endl;
  }

  // scale weights
  if( scaleWeights ) {
    cout << "Signal weights are multiplied by " << sW << endl;
    filter->scaleWeights(inputClasses[1],sW);
  }

  // apply low cutoff
  if( setLowCutoff ) {
    if( !filter->replaceMissing(SprUtils::lowerBound(lowCutoff),1) ) {
      cerr << "Unable to replace missing values in training data." << endl;
      return 2;
    }
    else
      cout << "Values below " << lowCutoff << " in training data"
	   << " have been replaced with medians." << endl;
  }

  // read validation data from file
  auto_ptr<SprAbsFilter> valFilter;
  if( split && !valFile.empty() ) {
    cerr << "Unable to split training data and use validation data " 
	 << "from a separate file." << endl;
    return 2;
  }
  if( split && valPrint!=0 ) {
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
    for( unsigned int  i=0;i<inputClasses.size();i++ ) {
      cout << "Points in class " << inputClasses[i] << ":   " 
	   << filter->ptsInClass(inputClasses[i]) << endl;
    }
  }
  if( !valFile.empty() && valPrint!=0 ) {
    auto_ptr<SprAbsReader> 
      valReader(SprRWFactory::makeReader(inputType,readMode));
    if( !includeSet.empty() ) {
      if( !valReader->chooseVars(includeSet) ) {
	cerr << "Unable to include variables in validation set." << endl;
	return 2;
      }
    }
    if( !excludeSet.empty() ) {
      if( !valReader->chooseAllBut(excludeSet) ) {
	cerr << "Unable to exclude variables from validation set." << endl;
	return 2;
      }
    }
    valFilter.reset(valReader->read(valFile.c_str()));
    if( valFilter.get() == 0 ) {
      cerr << "Unable to read data from file " << valFile.c_str() << endl;
      return 2;
    }
    vector<string> valVars;
    valFilter->vars(valVars);
    cout << "Read validation data from file " << valFile.c_str() 
	 << " for variables";
    for( unsigned int  i=0;i<valVars.size();i++ ) 
      cout << " \"" << valVars[i].c_str() << "\"";
    cout << endl;
    cout << "Total number of points read: " << valFilter->size() << endl;
    cout << "Points in class 0: " << valFilter->ptsInClass(inputClasses[0])
	 << " 1: " << valFilter->ptsInClass(inputClasses[1]) << endl;
  }

  // filter validation data by class
  if( valFilter.get() != 0 ) {
    if( !valFilter->filterByClass(inputClassesString.c_str()) ) {
      cerr << "Cannot choose input classes for string " 
	   << inputClassesString << endl;
      return 2;
    }
    valFilter->classes(inputClasses);
    cout << "Validation data filtered by class." << endl;
    for( unsigned int  i=0;i<inputClasses.size();i++ ) {
      cout << "Points in class " << inputClasses[i] << ":   " 
	   << valFilter->ptsInClass(inputClasses[i]) << endl;
    }
  }

  // scale weights
  if( scaleWeights && valFilter.get()!=0 )
    valFilter->scaleWeights(inputClasses[1],sW);

  // apply low cutoff
  if( setLowCutoff && valFilter.get()!=0 ) {
    if( !valFilter->replaceMissing(SprUtils::lowerBound(lowCutoff),1) ) {
      cerr << "Unable to replace missing values in validation data." << endl;
      return 2;
    }
    else
      cout << "Values below " << lowCutoff << " in validation data"
	   << " have been replaced with medians." << endl;
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

  // make per-event loss
  auto_ptr<SprAverageLoss> loss;
  switch( iLoss )
    {
    case 1 :
      loss.reset(new SprAverageLoss(&SprLoss::quadratic));
      cout << "Per-event loss set to "
           << "Quadratic loss (y-f(x))^2 " << endl;
      break;
    case 2 :
      loss.reset(new SprAverageLoss(&SprLoss::purity_ratio));
      cout << "Per-event loss set to "
           << "Exponential loss exp(-y*f(x)) " << endl;
      break;
    default :
      cout << "No per-event loss is chosen. Will use the default." << endl;
      break;
    }

  // open file with classifier configs
  ifstream file(configFile.c_str());
  if( !file ) {
    cerr << "Unable to open file " << configFile.c_str() << endl;
    return 3;
  }

  // prepare vectors of objects
  vector<SprAbsTwoClassCriterion*> criteria;
  vector<SprAbsClassifier*> destroyC;// classifiers to be deleted
  vector<SprIntegerBootstrap*> bstraps;
  vector<SprCCPair> useC;// classifiers and cuts to be used

  // read classifier params
  unsigned nLine = 0;
  bool discreteTree = false;
  bool mixedNodesTree = false;
  bool readOneEntry = false;
  bool fastSort = true;
  if( !SprClassifierReader::readTrainableConfig(file,nLine,filter.get(),
						discreteTree,mixedNodesTree,
						fastSort,criteria,
						bstraps,destroyC,useC,
						readOneEntry) ) {
    cerr << "Unable to read weak classifier configurations from file " 
	 << configFile.c_str() << endl;
    prepareExit(criteria,destroyC,bstraps);
    return 4;
  }
  cout << "Finished reading " << useC.size() << " classifiers from file "
       << configFile.c_str() << endl;

  // make Bagger
  auto_ptr<SprBagger> bagger;
  bool discrete = false;
  if( useArcE4 )
    bagger.reset(new SprArcE4(filter.get(),cycles,discrete));
  else
    bagger.reset(new SprBagger(filter.get(),cycles,discrete));

  // set seed for bootstrap if necessary
  if( initBootstrapFromTimeOfDay && !bagger->initBootstrapFromTimeOfDay() ) {
    cerr << "Unable to generate seed from time of day for Bagger." << endl;
    return 4;
  }

  // set validation
  if( valFilter.get()!=0 && !valFilter->empty() )
    bagger->setValidation(valFilter.get(),valPrint,0,loss.get());

  // read saved Bagger from file
  if( !resumeFile.empty() ) {
    if( !SprClassifierReader::readTrainable(resumeFile.c_str(),
					    bagger.get(),verbose) ) {
      cerr << "Failed to read saved Bagger from file " 
	   << resumeFile.c_str() << endl;
      prepareExit(criteria,destroyC,bstraps);
      return 5;
    }
    cout << "Read saved Bagger from file " << resumeFile.c_str()
	 << " with " << bagger->nTrained() << " trained classifiers." << endl;
  }

  // add trainable classifiers
  for( unsigned int  i=0;i<useC.size();i++ ) {
    if( !bagger->addTrainable(useC[i].first) ) {
      cerr << "Unable to add classifier " << i << " of type " 
	   << useC[i].first->name() << " to Bagger." << endl;
      prepareExit(criteria,destroyC,bstraps);
      return 6;
    }
  }

  // train
  if( !bagger->train(verbose) )
    cerr << "Bagger terminated with error." << endl;
  if( bagger->nTrained() == 0 ) {
    cerr << "Unable to train Bagger." << endl;
    prepareExit(criteria,destroyC,bstraps);
    return 7;
  }
  else {
    cout << "Bagger finished training with " << bagger->nTrained() 
	 << " classifiers." << endl;
  }

  // save trained Bagger
  if( !outFile.empty() ) {
    if( !bagger->store(outFile.c_str()) ) {
      cerr << "Cannot store Bagger in file " << outFile.c_str() << endl;
      prepareExit(criteria,destroyC,bstraps);
      return 8;
    }
  }

  // exit
  prepareExit(criteria,destroyC,bstraps);
  return 0;
}
