//$Id: SprBaggerDecisionTreeApp.cc,v 1.5 2007/02/05 21:49:45 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBagger.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprArcE4.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedBagger.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTopdownTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedTopdownTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDecisionTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedDecisionTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprSimpleReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDataFeeder.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprMyWriter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassSignalSignif.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassIDFraction.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassTaggerEff.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassPurity.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassGiniIndex.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassCrossEntropy.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassUniformPriorUL90.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassBKDiscovery.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassPunzi.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerBootstrap.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprCrossValidator.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAverageLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformation.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierReader.hh"

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
  cout << "\t-k discrete decision tree output (default=continuous)"<< endl;
  cout << "\t-o output Tuple file                               " << endl;
  cout << "\t-a input ascii file mode (see SprSimpleReader.hh)  " << endl;
  cout << "\t-n number of Bagger training cycles                " << endl;
  cout << "\t-l minimal number of entries per tree leaf (def=1) " << endl;
  cout << "\t-s max number of sampled features (def=0 no sampling)"<< endl;
  cout << "\t-y list of input classes (see SprAbsFilter.hh)     " << endl;
  cout << "\t-b use a version of Breiman's arc-x4 algorithm     " << endl;
  cout << "\t-v verbose level (0=silent default,1,2)            " << endl;
  cout << "\t-f store trained Bagger to file                    " << endl;
  cout << "\t-c criterion for optimization                      " << endl;
  cout << "\t\t 1 = correctly classified fraction               " << endl;
  cout << "\t\t 2 = signal significance s/sqrt(s+b)             " << endl;
  cout << "\t\t 3 = purity s/(s+b)                              " << endl;
  cout << "\t\t 4 = tagger efficiency Q                         " << endl;
  cout << "\t\t 5 = Gini index (default)                        " << endl;
  cout << "\t\t 6 = cross-entropy                               " << endl;
  cout << "\t\t 7 = 90% Bayesian upper limit with uniform prior " << endl;
  cout << "\t\t 8 = discovery potential 2*(sqrt(s+b)-sqrt(b))   " << endl;
  cout << "\t\t 9 = Punzi's sensitivity s/(0.5*nSigma+sqrt(b))  " << endl;
  cout << "\t\t -P background normalization factor for Punzi FOM" << endl;
  cout << "\t-g per-event loss for (cross-)validation           " << endl;
  cout << "\t\t 1 - quadratic loss (y-f(x))^2                   " << endl;
  cout << "\t\t 2 - exponential loss exp(-y*f(x))               " << endl;
  cout << "\t-m replace data values below this cutoff with medians" << endl;
  cout << "\t-i count splits on input variables                 " << endl;
  cout << "\t-r resume training for Bagger stored in file       " << endl;
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
  string hbkFile;
  int readMode = 1;
  unsigned cycles = 0;
  unsigned nmin = 1;
  int verbose = 0;
  string outFile;
  string resumeFile;
  int iCrit = 5;
  string valFile;
  unsigned valPrint = 0;
  bool scaleWeights = false;
  double sW = 1.;
  int nFeaturesToSample = 0;
  bool countTreeSplits = false;
  bool setLowCutoff = false;
  double lowCutoff = 0;
  string includeList, excludeList;
  unsigned nCross = 0;
  string nodeValidationString;
  bool useTopdown = true;
  bool discrete = false;
  int iLoss = 0;
  string inputClassesString;
  bool useArcE4 = false;
  double bW = 1.;
  string stringVarsDoNotFeed;

  // decode command line
  int c;
  extern char* optarg;
  //  extern int optind;
  while( (c = getopt(argc,argv,"hjko:a:n:l:s:y:bv:f:c:P:g:m:ir:t:d:w:V:z:Z:x:q:")) 
	 != EOF ) {
    switch( c )
      {
      case 'h' :
	help(argv[0]);
	return 1;
      case 'j' :
	useTopdown = false;
	break;
      case 'k' :
	discrete = true;
	break;
      case 'o' :
	hbkFile = optarg;
	break;
      case 'a' :
	readMode = (optarg==0 ? 1 : atoi(optarg));
	break;
      case 'n' :
	cycles = (optarg==0 ? 1 : atoi(optarg));
	break;
      case 'l' :
	nmin = (optarg==0 ? 1 : atoi(optarg));
	break;
      case 's' :
	nFeaturesToSample = (optarg==0 ? 0 : atoi(optarg));
	break;
      case 'y' :
	inputClassesString = optarg;
	break;
      case 'b' :
	useArcE4 = true;
	break;
      case 'v' :
	verbose = (optarg==0 ? 0 : atoi(optarg));
	break;
      case 'f' :
	outFile = optarg;
	break;
      case 'c' :
        iCrit = (optarg==0 ? 5 : atoi(optarg));
        break;
      case 'P' :
	bW = (optarg==0 ? 1 : atof(optarg));
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
      case 'i' :
	countTreeSplits = true;
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
      cout << "Folowing variables have been included in optimization: ";
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
      cout << "Folowing variables have been excluded from optimization: ";
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
  cout << "Read data from file " << trFile.c_str() 
       << " for variables";
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

  // make optimization criterion
  auto_ptr<SprAbsTwoClassCriterion> crit;
  switch( iCrit )
    {
    case 1 :
      crit.reset(new SprTwoClassIDFraction);
      cout << "Optimization criterion set to "
           << "Fraction of correctly classified events " << endl;
      break;
    case 2 :
      crit.reset(new SprTwoClassSignalSignif);
      cout << "Optimization criterion set to "
           << "Signal significance S/sqrt(S+B) " << endl;
      break;
    case 3 :
      crit.reset(new SprTwoClassPurity);
      cout << "Optimization criterion set to "
           << "Purity S/(S+B) " << endl;
      break;
    case 4 :
      crit.reset(new SprTwoClassTaggerEff);
      cout << "Optimization criterion set to "
           << "Tagging efficiency Q = e*(1-2w)^2 " << endl;
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
    case 7 :
      crit.reset(new SprTwoClassUniformPriorUL90);
      cout << "Optimization criterion set to "
           << "Inverse of 90% Bayesian upper limit with uniform prior" << endl;
      break;
    case 8 :
      crit.reset(new SprTwoClassBKDiscovery);
      cout << "Optimization criterion set to "
	   << "Discovery potential 2*(sqrt(S+B)-sqrt(B))" << endl;
      break;
    case 9 :
      crit.reset(new SprTwoClassPunzi(bW));
      cout << "Optimization criterion set to "
	   << "Punzi's sensitivity S/(0.5*nSigma+sqrt(B))" << endl;
      break;
    default :
      cerr << "Unable to make initialization criterion." << endl;
      return 3;
    }

  // check criterion vs classifier
  if( useArcE4 && !crit->symmetric() ) {
    cerr << "Unable to use arc-e4 with an asymmetric criterion." << endl;
    return 3;
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

  // make bootstrap for resampling input features
  auto_ptr<SprIntegerBootstrap> bootstrap;
  if( nFeaturesToSample > filter->dim() ) 
    nFeaturesToSample = filter->dim();
  if( nFeaturesToSample > 0 ) {
    bootstrap.reset(new SprIntegerBootstrap(filter->dim(),nFeaturesToSample));
    if( !resumeFile.empty() ) bootstrap->init(-1);
  }

  // make decision tree
  bool doMerge = !crit->symmetric();
  auto_ptr<SprDecisionTree> tree;
  if( useTopdown ) {
    tree.reset(new SprTopdownTree(filter.get(),crit.get(),nmin,doMerge,
				  discrete,bootstrap.get()));
  }
  else {
    tree.reset(new SprDecisionTree(filter.get(),crit.get(),nmin,doMerge,
				   discrete,bootstrap.get()));
  }
  if( countTreeSplits ) tree->startSplitCounter();

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
      for( int i=0;i<nodeMinSize[0].size();i++ )
	cout << nodeMinSize[0][i] << " ";
      cout << endl;
    }

    // loop over nodes to prepare classifiers
    vector<SprDecisionTree*> trees(nodeMinSize[0].size());
    vector<SprAbsClassifier*> classifiers(nodeMinSize[0].size());
    for( int i=0;i<nodeMinSize[0].size();i++ ) {
      SprDecisionTree* tree1 = 0;
      if( useTopdown ) {
	tree1 = new SprTopdownTree(filter.get(),crit.get(),nodeMinSize[0][i],
				   doMerge,discrete,bootstrap.get());
      }
      else {
	tree1 = new SprDecisionTree(filter.get(),crit.get(),nodeMinSize[0][i],
				    doMerge,discrete,bootstrap.get());
      }
      SprBagger* bagger1 = 0;
      if( useArcE4 )
	bagger1 = new SprArcE4(filter.get(),cycles,discrete);
      else
	bagger1 = new SprBagger(filter.get(),cycles,discrete);
      if( !bagger1->addTrainable(tree1) ) {
	cerr << "Unable to add decision tree to Bagger for CV." << endl;
	for( int j=0;j<trees.size();j++ ) {
	  delete trees[j];
	  delete classifiers[j];
	}
	return 4;
      }
      trees[i] = tree1;
      classifiers[i] = bagger1;
    }

    // cross-validate
    vector<double> cvFom;
    SprCrossValidator cv(filter.get(),nCross);
    if( !cv.validate(crit.get(),loss.get(),classifiers,
		     inputClasses[0],inputClasses[1],
		     SprUtils::lowerBound(0.5),cvFom,verbose) ) {
      cerr << "Unable to cross-validate." << endl;
      for( int j=0;j<trees.size();j++ ) {
	delete trees[j];
	delete classifiers[j];
      }
      return 4;
    }
    else {
      cout << "Cross-validated FOMs:" << endl;
      for( int i=0;i<cvFom.size();i++ ) {
	cout << "Node size=" << setw(8) << nodeMinSize[0][i] 
	     << "      FOM=" << setw(10) << cvFom[i] << endl;
      }
    }

    // cleanup
    for( int j=0;j<trees.size();j++ ) {
      delete trees[j];
      delete classifiers[j];
    }

    // normal exit
    return 0;
  }// end cross-validation

  // make Bagger
  auto_ptr<SprBagger> bagger;
  if( useArcE4 )
    bagger.reset(new SprArcE4(filter.get(),cycles,discrete));
  else
    bagger.reset(new SprBagger(filter.get(),cycles,discrete));

  // set validation
  if( valFilter.get()!=0 && !valFilter->empty() )
    bagger->setValidation(valFilter.get(),valPrint,crit.get(),loss.get());

  // read saved Bagger from file
  if( !resumeFile.empty() ) {
    if( !SprClassifierReader::readTrainable(resumeFile.c_str(),
                                            bagger.get(),verbose) ) {
      cerr << "Failed to read saved Bagger from file "
           << resumeFile.c_str() << endl;
      return 5;
    }
    cout << "Read saved Bagger from file " << resumeFile.c_str()
	 << " with " << bagger->nTrained() << " trained classifiers." 
	 << endl;
  }

  // add trainable tree
  if( !bagger->addTrainable(tree.get()) ) {
    cerr << "Unable to add decision tree to Bagger." << endl;
    return 6;
  }

  // train
  if( !bagger->train(verbose) )
    cerr << "Bagger terminated with error." << endl;
  if( bagger->nTrained() == 0 ) {
    cerr << "Unable to train Bagger." << endl;
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
      return 8;
    }
  }

  // print out counted splits
  if( countTreeSplits ) tree->printSplitCounter(cout);

  // make a trained Bagger
  auto_ptr<SprTrainedBagger> trainedBagger(bagger->makeTrained());
  if( trainedBagger.get() == 0 ) {
    cerr << "Unable to get trained Bagger." << endl;
    return 7;
  }

  // make histogram if requested
  if( hbkFile.empty() ) 
    return 0;

  // make a writer
  SprMyWriter hbk("training");
  if( !hbk.init(hbkFile.c_str()) ) {
    cerr << "Unable to open output file " << hbkFile.c_str() << endl;
    return 9;
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

  // feed
  SprDataFeeder feeder(filter.get(),&hbk,mapper);
  feeder.addClassifier(trainedBagger.get(),"bag");
  if( !feeder.feed(1000) ) {
    cerr << "Cannot feed data into file " << hbkFile.c_str() << endl;
    return 10;
  }

  // exit
  return 0;
}
