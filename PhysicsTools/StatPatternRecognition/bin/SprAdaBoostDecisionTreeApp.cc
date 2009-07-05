//$Id: SprAdaBoostDecisionTreeApp.cc,v 1.5 2007/11/12 06:19:11 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDecisionTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedDecisionTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTopdownTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedTopdownTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsWriter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDataFeeder.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRWFactory.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassGiniIndex.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassCrossEntropy.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassIDFraction.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprCrossValidator.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerBootstrap.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAverageLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformation.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
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
       << " training_data_file" << endl;
  cout << "\t Options: " << endl;
  cout << "\t-h --- help                                        " << endl;
  cout << "\t-j use regular tree instead of faster topdown tree " << endl;
  cout << "\t-M AdaBoost mode                                   " << endl;
  cout << "\t\t 1 = Discrete AdaBoost (default)                 " << endl;
  cout << "\t\t 2 = Real AdaBoost                               " << endl;
  cout << "\t\t 3 = Epsilon AdaBoost                            " << endl;
  cout << "\t-E epsilon for Epsilon and Real AdaBoosts (def=0.01)" << endl;
  cout << "\t-o output Tuple file                               " << endl;
  cout << "\t-a input ascii file mode (see SprSimpleReader.hh)  " << endl;
  cout << "\t-A save output data in ascii instead of Root       " << endl;
  cout << "\t-n number of AdaBoost training cycles              " << endl;
  cout << "\t-l minimal number of entries per tree leaf (def=1) " << endl;
  cout << "\t-y list of input classes (see SprAbsFilter.hh)     " << endl;
  cout << "\t-Q apply variable transformation saved in file     " << endl;
  cout << "\t-c criterion for optimization                      " << endl;
  cout << "\t\t 1 = correctly classified fraction               " << endl;
  cout << "\t\t 5 = Gini index (default)                        " << endl;
  cout << "\t\t 6 = cross-entropy                               " << endl;
  cout << "\t-g per-event loss for (cross-)validation           " << endl;
  cout << "\t\t 1 - quadratic loss (y-f(x))^2                   " << endl;
  cout << "\t\t 2 - exponential loss exp(-y*f(x))               " << endl;
  cout << "\t\t 3 - misid fraction                              " << endl;
  cout << "\t-b max number of sampled features (def=0 no sampling)" << endl;
  cout << "\t-m replace data values below this cutoff with medians" << endl;
  cout << "\t-i count splits on input variables                 " << endl;
  cout << "\t-s use standard AdaBoost (see SprTrainedAdaBoost.hh)"<< endl;
  cout << "\t-e skip initial event reweighting when resuming    " << endl;
  cout << "\t-u store data with modified weights to file        " << endl;
  cout << "\t-v verbose level (0=silent default,1,2)            " << endl;
  cout << "\t-f store trained AdaBoost to file                  " << endl;
  cout << "\t-F generate code for AdaBoost and store to file    " << endl;
  cout << "\t-r resume training for AdaBoost stored in file     " << endl;
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
  cout << "\t-x cross-validate by splitting data into a given "
       << "number of pieces" << endl;
  cout << "\t-q a set of minimal node sizes for cross-validation" << endl;
  cout << "\t\t Node sizes must be listed in quotes and separated by commas." 
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
  unsigned cycles = 0;
  unsigned nmin = 1;
  int iCrit = 5;
  int verbose = 0;
  string outFile;
  string codeFile;
  string resumeFile;
  string valFile;
  unsigned valPrint = 0;
  bool scaleWeights = false;
  double sW = 1.;
  bool countTreeSplits = false;
  bool useStandardAB = false;
  int iAdaBoostMode = 1;
  double epsilon = 0.01;
  bool skipInitialEventReweighting = false;
  string weightedDataOut;
  bool setLowCutoff = false;
  double lowCutoff = 0;
  string includeList, excludeList;
  unsigned nCross = 0;
  string nodeValidationString;
  int nFeaturesToSample = 0;
  bool bagInput = false;
  bool useTopdown = true;
  int iLoss = 0;
  string inputClassesString;
  string stringVarsDoNotFeed;
  bool split = false;
  double splitFactor = 0;
  bool splitRandomize = false;
  string transformerFile;

  // decode command line
  int c;
  extern char* optarg;
  //  extern int optind;
  while((c = getopt(argc,argv,"hjM:E:o:a:An:l:y:Q:c:g:b:m:iseu:v:f:F:r:K:Dt:d:w:V:z:Z:x:q:")) != EOF ) {
    switch( c )
      {
      case 'h' :
	help(argv[0]);
	return 1;
      case 'j' :
	useTopdown = false;
	break;
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
	cycles = (optarg==0 ? 1 : atoi(optarg));
	break;
      case 'l' :
	nmin = (optarg==0 ? 1 : atoi(optarg));
	break;
      case 'y' :
	inputClassesString = optarg;
	break;
      case 'Q' :
        transformerFile = optarg;
        break;
      case 'c' :
        iCrit = (optarg==0 ? 5 : atoi(optarg));
        break;
      case 'g' :
        iLoss = (optarg==0 ? 0 : atoi(optarg));
        break;
      case 'b' :
	bagInput = true;
	nFeaturesToSample = (optarg==0 ? 0 : atoi(optarg));
	break;
      case 'm' :
	if( optarg != 0 ) {
	  setLowCutoff = true;
	  lowCutoff = atof(optarg);
	}
	break;
      case 'i' :
	countTreeSplits = true;
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
      case 'F' :
	codeFile = optarg;
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
      case 'x' :
	nCross = (optarg==0 ? 0 : atoi(optarg));
	break;
      case 'q' :
	nodeValidationString = optarg;
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

  // make optimization criterion
  auto_ptr<SprAbsTwoClassCriterion> crit;
  switch( iCrit )
    {
    case 1 :
      crit.reset(new SprTwoClassIDFraction);
      cout << "Optimization criterion set to "
           << "Fraction of correctly classified events " << endl;
      break;
    case 5 :
      crit.reset(new SprTwoClassGiniIndex);
      cout << "Optimization criterion set to "
	   << "Gini index  -1+p^2+q^2 " << endl;
      break;
    case 6 :
      crit.reset(new SprTwoClassCrossEntropy);
      cout << "Optimization criterion set to "
	   << "Cross-entropy p*log(p)+q*log(q) " << endl;
      break;
    default :
      cerr << "Unable to make initialization criterion." << endl;
      return 3;
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
      break;
    case 2 :
      loss.reset(new SprAverageLoss(&SprLoss::exponential));
      cout << "Per-event loss set to "
           << "Exponential loss exp(-y*f(x)) " << endl;
      break;
    case 3 :
      loss.reset(new SprAverageLoss(&SprLoss::correct_id,
			       &SprTransformation::inftyRangeToDiscrete01));
      cout << "Per-event loss set to "
	   << "Misid rate int(y==f(x)) " << endl;
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

  // make bootstrap for resampling input features
  auto_ptr<SprIntegerBootstrap> bootstrap;
  if( nFeaturesToSample > (int)filter->dim() ) 
    nFeaturesToSample = filter->dim();
  if( nFeaturesToSample > 0 ) {
    bootstrap.reset(new SprIntegerBootstrap(filter->dim(),nFeaturesToSample));
    if( !resumeFile.empty() ) bootstrap->init(-1);
  }

  // make decision tree
  bool discrete = true;
  if( abMode==SprTrainedAdaBoost::Real ) discrete = false;
  bool doMerge = false;
  auto_ptr<SprDecisionTree> tree;
  if( useTopdown )
    tree.reset(new SprTopdownTree(filter.get(),crit.get(),
				  nmin,discrete,bootstrap.get()));
  else
    tree.reset(new SprDecisionTree(filter.get(),crit.get(),
				   nmin,doMerge,discrete,bootstrap.get()));
  if( countTreeSplits ) tree->startSplitCounter();
  if( abMode == SprTrainedAdaBoost::Real ) tree->forceMixedNodes();
  tree->useFastSort();

  // if cross-validation requested, cross-validate and exit
  if( nCross > 0 ) {
    // message
    cout << "Will cross-validate by dividing training data into " 
	 << nCross << " subsamples." << endl;
    vector<vector<int> > nodeMinSize;

    // decode validation string
    if( !nodeValidationString.empty() )
      SprStringParser::parseToInts(nodeValidationString.c_str(),nodeMinSize);
    else {
      nodeMinSize.resize(1);
      nodeMinSize[0].push_back(nmin);
    }
    if( nodeMinSize.empty() || nodeMinSize[0].empty() ) {
      cerr << "Unable to determine node size for cross-validation." << endl;
      return 4;
    }
    else {
      cout << "Will cross-validate for trees with minimal node sizes: ";
      for( unsigned int i=0;i<nodeMinSize[0].size();i++ )
	cout << nodeMinSize[0][i] << " ";
      cout << endl;
    }

    // loop over nodes to prepare classifiers
    vector<SprDecisionTree*> trees(nodeMinSize[0].size());
    vector<SprAbsClassifier*> classifiers(nodeMinSize[0].size());
    for( unsigned int i=0;i<nodeMinSize[0].size();i++ ) {
      SprDecisionTree* tree1 = 0;
      if( useTopdown )
	tree1 = 
	  new SprTopdownTree(filter.get(),crit.get(),
			     nodeMinSize[0][i],
			     discrete,bootstrap.get());
      else
	tree1 =
	  new SprDecisionTree(filter.get(),crit.get(),
			      nodeMinSize[0][i],doMerge,
			      discrete,bootstrap.get());
      if( abMode == SprTrainedAdaBoost::Real ) tree1->forceMixedNodes();
      tree1->useFastSort();
      SprAdaBoost* ab1 = new SprAdaBoost(filter.get(),cycles,
					 useStandardAB,abMode,bagInput);
      cout << "Setting epsilon to " << epsilon << endl;
      ab1->setEpsilon(epsilon);
      if( !ab1->addTrainable(tree1,SprUtils::lowerBound(0.5)) ) {
	cerr << "Unable to add decision tree to AdaBoost for CV." << endl;
	for( unsigned int j=0;j<trees.size();j++ ) {
	  delete trees[j];
	  delete classifiers[j];
	}
	return 4;
      }
      trees[i] = tree1;
      classifiers[i] = ab1;
    }

    // cross-validate
    vector<double> cvFom;
    SprCrossValidator cv(filter.get(),nCross);
    if( !cv.validate(crit.get(),loss.get(),classifiers,
		     inputClasses[0],inputClasses[1],
		     SprUtils::lowerBound(0.5),cvFom,verbose) ) {
      cerr << "Unable to cross-validate." << endl;
      for( unsigned int j=0;j<trees.size();j++ ) {
	delete trees[j];
	delete classifiers[j];
      }
      return 4;
    }
    else {
      cout << "Cross-validated FOMs:" << endl;
      for( unsigned int i=0;i<cvFom.size();i++ ) {
	cout << "Node size=" << setw(8) << nodeMinSize[0][i] 
	     << "      FOM=" << setw(10) << cvFom[i] << endl;
      }
    }

    // cleanup
    for( unsigned int j=0;j<trees.size();j++ ) {
      delete trees[j];
      delete classifiers[j];
    }

    // normal exit
    return 0;
  }// end cross-validation

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
      return 5;
    }
    cout << "Read saved AdaBoost from file " << resumeFile.c_str()
	 << " with " << ab.nTrained() << " trained classifiers." << endl;
  }
  if( skipInitialEventReweighting ) ab.skipInitialEventReweighting(true);

  // add trainable tree
  if( !ab.addTrainable(tree.get(),SprUtils::lowerBound(0.5)) ) {
    cerr << "Unable to add decision tree to AdaBoost." << endl;
    return 6;
  }

  // train
  if( !ab.train(verbose) )
    cerr << "AdaBoost terminated with error." << endl;
  if( ab.nTrained() == 0 ) {
    cerr << "Unable to train AdaBoost." << endl;
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
      return 8;
    }
  }

  // print out counted splits
  if( countTreeSplits ) tree->printSplitCounter(cout);

  // save reweighted data
  if( !weightedDataOut.empty() ) {
    if( !ab.storeData(weightedDataOut.c_str()) ) {
      cerr << "Cannot store weighted AdaBoost data to file " 
	   << weightedDataOut.c_str() << endl;
      return 9;
    }
  }

  // make a trained AdaBoost
  auto_ptr<SprTrainedAdaBoost> trainedAda(ab.makeTrained());
  if( trainedAda.get() == 0 ) {
    cerr << "Unable to get trained AdaBoost." << endl;
    return 7;
  }

  // store code into file
  if( !codeFile.empty() ) {
    if( !trainedAda->storeCode(codeFile.c_str()) ) {
      cerr << "Unable to store code for trained AdaBoost." << endl;
      return 8;
    }
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
  feeder.addClassifier(trainedAda.get(),"ada");
  if( !feeder.feed(1000) ) {
    cerr << "Cannot feed data into file " << tupleFile.c_str() << endl;
    return 11;
  }

  // exit
  return 0;
}
