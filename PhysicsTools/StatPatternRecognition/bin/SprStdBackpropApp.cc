//$Id: SprStdBackpropApp.cc,v 1.4 2007/11/12 06:19:11 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStdBackprop.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedStdBackprop.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsWriter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDataFeeder.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRWFactory.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprCrossValidator.hh"
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
#include <vector>
#include <set>
#include <string>
#include <memory>
#include <iomanip>

using namespace std;


void help(const char* prog) 
{
  cout << "Usage:  " << prog 
       << " training_data_file " << endl;
  cout << "\t Options: " << endl;
  cout << "\t-h --- help                                        " << endl;
  cout << "\t-o output Tuple file                               " << endl;
  cout << "\t-a input ascii file mode (see SprSimpleReader.hh)  " << endl;
  cout << "\t-A save output data in ascii instead of Root       " << endl;
  cout << "\t-M AdaBoost mode                                   " << endl;
  cout << "\t\t 1 = Discrete AdaBoost (default)                 " << endl;
  cout << "\t\t 2 = Real AdaBoost                               " << endl;
  cout << "\t\t 3 = Epsilon AdaBoost                            " << endl;
  cout << "\t-E epsilon for Epsilon and Real AdaBoosts (def=0.01)" << endl;
  cout << "\t-n number of AdaBoost training cycles (1 for single NN)" << endl;
  cout << "\t-l number of Neural Net training cycles            " << endl;
  cout << "\t-N neural net configuration, e.g., '6:3:1' (see SprStdBackprop.hh)" << endl;
  cout << "\t-L learning rate of the network (default=0.1)      " << endl;
  cout << "\t-I learning rate for network initialization (def=0.1)" << endl;
  cout << "\t-i number of input points to use for initialization (def=all)"
       << endl;
  cout << "\t-y list of input classes (see SprAbsFilter.hh)     " << endl;
  cout << "\t-Q apply variable transformation saved in file     " << endl;
  cout << "\t-g per-event loss for (cross-)validation           " << endl;
  cout << "\t\t 1 - quadratic loss (y-f(x))^2                   " << endl;
  cout << "\t\t 2 - exponential loss exp(-y*f(x))               " << endl;
  cout << "\t-m replace data values below this cutoff with medians" << endl;
  cout << "\t-s use standard AdaBoost (see SprTrainedAdaBoost.hh)"<< endl;
  cout << "\t-e skip initial event reweighting when resuming    " << endl;
  cout << "\t-u store data with modified weights to file        " << endl;
  cout << "\t-v verbose level (0=silent default,1,2)            " << endl;
  cout << "\t-f store trained AdaBoost to file                  " << endl;
  cout << "\t-r resume training for AdaBoost stored in file     " << endl;
  cout << "\t-R resume training for a single neural net stored in file" 
       << endl;
  cout << "\t-S resume training from SNNS configuration stored in file" 
       << endl;
  cout << "\t-K keep this fraction in training set and          " << endl;
  cout << "\t\t put the rest into validation set                " << endl;
  cout << "\t-D randomize training set split-up                 " << endl;
  cout << "\t-t read validation/test data from a file           " << endl;
  cout << "\t\t (must be in same format as input data!!!        " << endl;
  cout << "\t-d frequency of print-outs for validation data     " << endl;
  cout << "\t-w scale all signal weights by this factor         " << endl;
  cout << "\t-V include only these input variables              " << endl;
  cout << "\t-z exclude input variables from the list           " << endl;
  cout << "\t-Z exclude input variables from the list, "
       << "but put them in the output file " << endl;
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
  SprRWFactory::DataType writeMode = SprRWFactory::Root;
  unsigned adaCycles = 0;
  unsigned nnCycles = 0;
  double eta = 0.1;
  int iLoss = 1;
  int verbose = 0;
  string outFile;
  string valFile;
  unsigned valPrint = 0;
  bool scaleWeights = false;
  double sW = 1.;
  bool useStandardAB = false;
  int iAdaBoostMode = 1;
  double epsilon = 0.01;
  bool skipInitialEventReweighting = false;
  string weightedDataOut;
  bool setLowCutoff = false;
  double lowCutoff = 0;
  string includeList, excludeList;
  string inputClassesString;
  string stringVarsDoNotFeed;
  string resumeFile, resumeSNNSFile, resumeNNFile;
  string netConfig;
  double initEta = 0.1;
  unsigned initPoints = 0;
  bool split = false;
  double splitFactor = 0;
  bool splitRandomize = false;
  string transformerFile;

  // decode command line
  int c;
  extern char* optarg;
  //  extern int optind;
  while((c = getopt(argc,argv,"ho:a:AM:E:n:l:N:L:I:i:y:Q:g:m:seu:v:f:r:R:S:K:Dt:d:w:V:z:Z:")) != EOF ) {
    switch( c )
      {
      case 'h' :
	help(argv[0]);
	return 1;
      case 'M' :
	iAdaBoostMode = (optarg==0 ? 1 : atoi(optarg));
	break;
      case 'E' :
	epsilon = (optarg==0 ? 0.01 : atof(optarg));
	break;
      case 'o' :
	tupleFile = optarg;
	break;
      case 'a' :
	readMode = (optarg==0 ? 0 : atoi(optarg));
	break;
      case 'A' :
	writeMode = SprRWFactory::Ascii;
	break;
      case 'n' :
	adaCycles = (optarg==0 ? 1 : atoi(optarg));
	break;
      case 'l' :
	nnCycles = (optarg==0 ? 1 : atoi(optarg));
	break;
      case 'N' :
	netConfig = optarg;
	break;
      case 'L' :
	eta = (optarg==0 ? 0.1 : atof(optarg));
	break;
      case 'I' :
	initEta = (optarg==0 ? 0.1 : atof(optarg));
	break;
      case 'i' :
	initPoints = (optarg==0 ? 0 : atoi(optarg));
	break;
      case 'y' :
	inputClassesString = optarg;
	break;
      case 'Q' :
        transformerFile = optarg;
        break;
      case 'g' :
        iLoss = (optarg==0 ? 0 : atoi(optarg));
        break;
      case 'm' :
	if( optarg != 0 ) {
	  setLowCutoff = true;
	  lowCutoff = atof(optarg);
	}
	break;
      case 's' :
	useStandardAB = true;
	break;
      case 'e' :
	skipInitialEventReweighting = true;
	break;
      case 'u' :
	weightedDataOut = optarg;
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
      case 'R' :
	resumeNNFile = optarg;
	break;
      case 'S' :
	resumeSNNSFile = optarg;
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
      case 'Z' :
	stringVarsDoNotFeed = optarg;
	break;
      }
  }

  // Get training file.
  string trFile = argv[argc-1];

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
    for( unsigned int i=0;i<inputClasses.size();i++ ) {
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
    for( unsigned int i=0;i<valVars.size();i++ ) 
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
    for( unsigned int i=0;i<inputClasses.size();i++ ) {
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
      if( adaCycles > 1 ) {
	loss.reset(new SprAverageLoss(&SprLoss::quadratic,
				      &SprTransformation::logit));
      }
      else {
	loss.reset(new SprAverageLoss(&SprLoss::quadratic));
      }
      cout << "Per-event loss set to "
           << "Quadratic loss (y-f(x))^2 " << endl;
      useStandardAB = true;
      break;
    case 2 :
      if( adaCycles > 1 ) {
	loss.reset(new SprAverageLoss(&SprLoss::exponential));
      }
      else {
	loss.reset(new SprAverageLoss(&SprLoss::exponential,
				      &SprTransformation::logitInverse));
      }
      cout << "Per-event loss set to "
           << "Exponential loss exp(-y*f(x)) " << endl;
      useStandardAB = true;
      break;
    default :
      cout << "No per-event loss is chosen. Will use the default." << endl;
      break;
    }

  // make AdaBoost mode
  SprTrainedAdaBoost::AdaBoostMode abMode = SprTrainedAdaBoost::Discrete;
  switch( iAdaBoostMode )
    {
    case 1 :
      abMode = SprTrainedAdaBoost::Discrete;
      cout << "Will train Discrete AdaBoost." << endl;
      break;
    case 2 :
      abMode = SprTrainedAdaBoost::Real;
      cout << "Will train Real AdaBoost." << endl;
      break;
    case 3 :
      abMode = SprTrainedAdaBoost::Epsilon;
      cout << "Will train Epsilon AdaBoost." << endl;
      break;
   default :
      cout << "Will train Discrete AdaBoost." << endl;
      break;
    }

  // sanity check
  int resume = int(!resumeFile.empty()) 
    + int(!resumeNNFile.empty())
    + int(!resumeSNNSFile.empty());
  if( resume > 1 ) {
    cerr << "Reading more than one classifier configuration is not allowed." 
	 << " Requested: " << resume << endl;
    return 5;
  }
  if( (!resumeNNFile.empty() || !resumeSNNSFile.empty()) 
      && !netConfig.empty() ) {
    cerr << "What do you want to do - read NN configuration from a file " 
	 << "or specify configuration on the command line? "
	 << "Life is tough - you cannot do both." << endl;
    return 5;
  }

  // make a single NN
  auto_ptr<SprStdBackprop> stdnn;
  if( adaCycles>0 && resumeNNFile.empty() && resumeSNNSFile.empty() ) {
    stdnn.reset(new SprStdBackprop(filter.get(),
				   netConfig.c_str(),
				   nnCycles,
				   eta));
    if( !stdnn->init(initEta,initPoints) ) {
      cerr << "Unable to initialize neural net." << endl;
      return 6;
    }
  }
  else {
    stdnn.reset(new SprStdBackprop(filter.get(),
				   nnCycles,
				   eta));
  }
  
  // read saved NN from file
  SprTrainedStdBackprop* trainedNN = 0;
  if( !resumeSNNSFile.empty() ) {
    if( !stdnn->readSNNS(resumeSNNSFile.c_str()) ) {
      cerr << "Unable to read SNNS configuration from file " 
	   << resumeSNNSFile.c_str() << endl;
      return 6;
    }
    trainedNN = stdnn->makeTrained();
    cout << "Read SNNS configuration from file " 
	 << resumeSNNSFile.c_str() << endl;
  }
  if( !resumeNNFile.empty() ) {
    if( !SprClassifierReader::readTrainable(resumeNNFile.c_str(),
					    stdnn.get(),verbose) ) {
      cerr << "Unable to read SPR NN configuration from file " 
	   << resumeNNFile.c_str() << endl;
      return 6;
    }
    trainedNN = stdnn->makeTrained();
    cout << "Read SPR neural net configuration from file " 
	 << resumeNNFile.c_str() << endl;
  }
	      
  // make classifier to train
  auto_ptr<SprAbsClassifier> classifier;
  if( adaCycles != 1 ) {
    // make AdaBoost
    SprAdaBoost* ab = new SprAdaBoost(filter.get(),
				      adaCycles,
				      useStandardAB,
				      abMode);
    cout << "Setting epsilon to " << epsilon << endl;
    ab->setEpsilon(epsilon);
    
    // skip reweigting
    if( skipInitialEventReweighting ) ab->skipInitialEventReweighting(true);

    // set validation
    if( valFilter.get()!=0 && !valFilter->empty() )
      ab->setValidation(valFilter.get(),valPrint,loss.get());
    
    // read saved AdaBoost
    if( resumeFile.empty() ) {
      if( trainedNN != 0 ) {
	if( !ab->addTrained(trainedNN,true) ) {
	  cerr << "Unable to add first trained NN to AdaBoost." << endl;
	  return 6;
	}
      }
    }
    else {
      if( !SprClassifierReader::readTrainable(resumeFile.c_str(),
					      ab,verbose) ) {
	cerr << "Failed to read saved AdaBoost from file " 
	     << resumeFile.c_str() << endl;
	return 6;
      }
      cout << "Read saved AdaBoost from file " << resumeFile.c_str()
	   << " with " << ab->nTrained() << " trained classifiers." << endl;
    }

    // add a trainable NN
    if( !ab->addTrainable(stdnn.get()) ) {
      cerr << "Unable to add neural net to AdaBoost." << endl;
      return 6;
    }
    
    // reset classifier
    classifier.reset(ab);
  }
  else {
    // set validation
    if( valFilter.get()!=0 && !valFilter->empty() )
      stdnn->setValidation(valFilter.get(),valPrint,loss.get());
    
    // reset classifier
    classifier.reset(stdnn.release());
  }
  
  // train
  if( !classifier->train(verbose) ) {
    cerr << "Training terminated with error." << endl;
    return 7;
  }
  else {
    cout << "Training done." << endl;
    if( adaCycles != 1 ) {
      SprAdaBoost* ab = static_cast<SprAdaBoost*>(classifier.get());
      cout << "AdaBoost finished training with " << ab->nTrained() 
	   << " classifiers." << endl;    
    }
  }

  // save trained classifier
  if( !outFile.empty() ) {
    if( !classifier->store(outFile.c_str()) ) {
      cerr << "Cannot store classifier in file " << outFile.c_str() << endl;
      return 8;
    }
  }

  // save reweighted data
  if( adaCycles > 1 ) {
    if( !weightedDataOut.empty() ) {
      SprAdaBoost* ab = static_cast<SprAdaBoost*>(classifier.get());
      if( !ab->storeData(weightedDataOut.c_str()) ) {
	cerr << "Cannot store weighted AdaBoost data to file " 
	     << weightedDataOut.c_str() << endl;
	return 9;
      }
    }
  }

  // make a trained AdaBoost
  auto_ptr<SprAbsTrainedClassifier> trained(classifier->makeTrained());
  if( trained.get() == 0 ) {
    cerr << "Unable to get trained classifier." << endl;
    return 9;
  }

  // make histogram if requested
  if( tupleFile.empty() ) 
    return 0;

  // make a writer
  auto_ptr<SprAbsWriter> tuple(SprRWFactory::makeWriter(writeMode,"training"));
  if( !tuple->init(tupleFile.c_str()) ) {
    cerr << "Unable to open output file " << tupleFile.c_str() << endl;
    return 10;
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

  // feed
  SprDataFeeder feeder(filter.get(),tuple.get(),mapper);
  string classifierName;
  if( adaCycles != 1 )
    classifierName = "adann";
  else
    classifierName = "nn";
  feeder.addClassifier(trained.get(),classifierName.c_str());
  if( !feeder.feed(1000) ) {
    cerr << "Cannot feed data into file " << tupleFile.c_str() << endl;
    return 11;
  }

  // exit
  return 0;
}
