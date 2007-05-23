//$Id: SprBoosterApp.cc,v 1.1 2007/02/05 21:49:45 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprSimpleReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprMyWriter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerBootstrap.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAverageLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformation.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierReader.hh"

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
  cout << "\t-M AdaBoost mode                                   " << endl;
  cout << "\t\t 1 = Discrete AdaBoost (default)                 " << endl;
  cout << "\t\t 2 = Real AdaBoost                               " << endl;
  cout << "\t\t 3 = Epsilon AdaBoost                            " << endl;
  cout << "\t-E epsilon for Epsilon and Real AdaBoosts (def=0.01)" << endl;
  cout << "\t-a input ascii file mode (see SprSimpleReader.hh)  " << endl;
  cout << "\t-n number of AdaBoost training cycles              " << endl;
  cout << "\t-y list of input classes (see SprAbsFilter.hh)     " << endl;
  cout << "\t-g per-event loss for (cross-)validation           " << endl;
  cout << "\t\t 1 - quadratic loss (y-f(x))^2                   " << endl;
  cout << "\t\t 2 - exponential loss exp(-y*f(x))               " << endl;
  cout << "\t-b bootstrap training sample                       " << endl;
  cout << "\t-m replace data values below this cutoff with medians" << endl;
  cout << "\t-e skip initial event reweighting when resuming    " << endl;
  cout << "\t-u store data with modified weights to file        " << endl;
  cout << "\t-v verbose level (0=silent default,1,2)            " << endl;
  cout << "\t-f store trained AdaBoost to file                  " << endl;
  cout << "\t-r resume training for AdaBoost stored in file     " << endl;
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
  for( int i=0;i<criteria.size();i++ ) delete criteria[i];
  for( int i=0;i<classifiers.size();i++ ) delete classifiers[i];
  for( int i=0;i<bstraps.size();i++ ) delete bstraps[i];
}


int main(int argc, char ** argv)
{
  // check command line
  if( argc < 3 ) {
    help(argv[0]);
    return 1;
  }

  // init
  int readMode = 1;
  unsigned cycles = 0;
  int verbose = 0;
  string outFile;
  string resumeFile;
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
  int iLoss = 0;
  bool bagInput = false;
  string inputClassesString;

  // decode command line
  int c;
  extern char* optarg;
  //  extern int optind;
  while((c = getopt(argc,argv,"hM:E:a:n:y:g:bm:eu:v:f:r:t:d:w:V:z:")) != EOF ) {
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
      case 'a' :
	readMode = (optarg==0 ? 1 : atoi(optarg));
	break;
      case 'n' :
	cycles = (optarg==0 ? 1 : atoi(optarg));
	break;
      case 'y' :
	inputClassesString = optarg;
	break;
      case 'g' :
        iLoss = (optarg==0 ? 0 : atoi(optarg));
        break;
      case 'b' :
	bagInput = true;
	break;
      case 'm' :
	if( optarg != 0 ) {
	  setLowCutoff = true;
	  lowCutoff = atof(optarg);
	}
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

  // read training data from file
  auto_ptr<SprAbsFilter> filter(reader.read(trFile.c_str()));
  if( filter.get() == 0 ) {
    cerr << "Unable to read data from file " << trFile.c_str() << endl;
    return 2;
  }
  vector<string> vars;
  filter->vars(vars);
  cout << "Read data from file " << trFile.c_str() << " for variables";
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
  if( !valFile.empty() && valPrint!=0 ) {
    SprSimpleReader valReader(readMode);
    if( !includeSet.empty() ) {
      if( !valReader.chooseVars(includeSet) ) {
	cerr << "Unable to include variables in validation set." << endl;
	return 2;
      }
    }
    if( !excludeSet.empty() ) {
      if( !valReader.chooseAllBut(excludeSet) ) {
	cerr << "Unable to exclude variables from validation set." << endl;
	return 2;
      }
    }
    valFilter.reset(valReader.read(valFile.c_str()));
    if( valFilter.get() == 0 ) {
      cerr << "Unable to read data from file " << valFile.c_str() << endl;
      return 2;
    }
    vector<string> valVars;
    valFilter->vars(valVars);
    cout << "Read validation data from file " << valFile.c_str() 
	 << " for variables";
    for( int i=0;i<valVars.size();i++ ) 
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
    for( int i=0;i<inputClasses.size();i++ ) {
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

  // make per-event loss
  auto_ptr<SprAverageLoss> loss;
  switch( iLoss )
    {
    case 1 :
      loss.reset(new SprAverageLoss(&SprLoss::quadratic,
				    &SprTransformation::logit));
      cout << "Per-event loss set to "
           << "Quadratic loss (y-f(x))^2 " << endl;
      useStandardAB = true;
      break;
    case 2 :
      loss.reset(new SprAverageLoss(&SprLoss::exponential));
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
  bool discreteTree = (abMode!=SprTrainedAdaBoost::Real);
  bool mixedNodesTree = (abMode==SprTrainedAdaBoost::Real);
  bool readOneEntry = false;
  if( !SprClassifierReader::readTrainableConfig(file,nLine,filter.get(),
						discreteTree,mixedNodesTree,
						criteria,bstraps,destroyC,useC,
						readOneEntry) ) {
    cerr << "Unable to read weak classifier configurations from file " 
	 << configFile.c_str() << endl;
    prepareExit(criteria,destroyC,bstraps);
    return 4;
  }
  cout << "Finished reading " << useC.size() << " classifiers from file "
       << configFile.c_str() << endl;

  // make AdaBoost
  SprAdaBoost ab(filter.get(),cycles,useStandardAB,abMode,bagInput);
  cout << "Setting epsilon to " << epsilon << endl;
  ab.setEpsilon(epsilon);

  // set validation
  if( valFilter.get()!=0 && !valFilter->empty() )
    ab.setValidation(valFilter.get(),valPrint,loss.get());

  // read saved Boost from file
  if( !resumeFile.empty() ) {
    if( !SprClassifierReader::readTrainable(resumeFile.c_str(),
					    &ab,verbose) ) {
      cerr << "Failed to read saved AdaBoost from file " 
	   << resumeFile.c_str() << endl;
      prepareExit(criteria,destroyC,bstraps);
      return 5;
    }
    cout << "Read saved AdaBoost from file " << resumeFile.c_str()
	 << " with " << ab.nTrained() << " trained classifiers." << endl;
  }
  if( skipInitialEventReweighting ) ab.skipInitialEventReweighting(true);

  // add trainable classifiers
  for( int i=0;i<useC.size();i++ ) {
    if( !ab.addTrainable(useC[i].first,useC[i].second) ) {
      cerr << "Unable to add classifier " << i << " of type " 
	   << useC[i].first->name() << " to AdaBoost." << endl;
      prepareExit(criteria,destroyC,bstraps);
      return 6;
    }
  }

  // train
  if( !ab.train(verbose) )
    cerr << "AdaBoost terminated with error." << endl;
  if( ab.nTrained() == 0 ) {
    cerr << "Unable to train AdaBoost." << endl;
    prepareExit(criteria,destroyC,bstraps);
    return 7;
  }
  else {
    cout << "AdaBoost finished training with " << ab.nTrained() 
	 << " classifiers." << endl;
  }

  // save trained AdaBoost
  if( !outFile.empty() ) {
    if( !ab.store(outFile.c_str()) ) {
      cerr << "Cannot store AdaBoost in file " << outFile.c_str() << endl;
      prepareExit(criteria,destroyC,bstraps);
      return 8;
    }
  }

  // save reweighted data
  if( !weightedDataOut.empty() ) {
    if( !ab.storeData(weightedDataOut.c_str()) ) {
      cerr << "Cannot store weighted AdaBoost data to file " 
	   << weightedDataOut.c_str() << endl;
      prepareExit(criteria,destroyC,bstraps);
      return 9;
    }
  }

  // exit
  prepareExit(criteria,destroyC,bstraps);
  return 0;
}
