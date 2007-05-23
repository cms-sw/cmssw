//$Id: SprMultiClassApp.cc,v 1.1 2007/02/05 21:49:45 narsky Exp $
/*
  Note: "-y" option has a different meaning for this executable than
  for other executables in the package. Instead of specifying what
  classes should be treated as background and what classes should be
  treated as signal, the "-y" option simply selects input classes for
  inclusion in the multi-class algorithm. Therefore, entering groups
  of classes separated by semicolons or specifying "." as an input
  class would make no sense. This executable expects a list of classes
  separated by commas.
*/

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprMultiClassLearner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedMultiClassLearner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprSimpleReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDataFeeder.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprMyWriter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerBootstrap.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformation.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAverageLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprMultiClassReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"

#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <memory>

using namespace std;


void help(const char* prog) 
{
  cout << "Usage:  " << prog << " training_data_file" << endl;
  cout << "\t Options: " << endl;
  cout << "\t-h --- help                                             " << endl;
  cout << "\t-o output Tuple file                                    " << endl;
  cout << "\t-a input ascii file mode (see SprSimpleReader.hh)       " << endl;
  cout << "\t-y list of input classes                                " << endl;
  cout << "\t\t Classes must be listed in quotes and separated by commas." 
       << endl;
  cout << "\t-e Multi class mode (OneVsAll=1 (default); OneVsOne=2)  " << endl;
  cout << "\t-c file with trainable classifier configurations        " << endl;
  cout << "\t-g per-event loss to be displayed for each input class  " << endl;
  cout << "\t\t 1 - quadratic loss (y-f(x))^2                        " << endl;
  cout << "\t\t 2 - exponential loss exp(-y*f(x))                    " << endl;
   cout << "\t-m replace data values below this cutoff with medians  " << endl;
  cout << "\t-v verbose level (0=silent default,1,2)                 " << endl;
  cout << "\t-f store trained multi class learner to file            " << endl;
  cout << "\t-r read multi class learner configuration stored in file" << endl;
  cout << "\t-t read validation/test data from a file           " << endl;
  cout << "\t\t (must be in same format as input data!!!        " << endl;
  cout << "\t-V include only these input variables              " << endl;
  cout << "\t-z exclude input variables from the list           " << endl;
  cout << "\t-Z exclude input variables from the list, "
       << "but put them in the output file " << endl;
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
  if( argc < 2 ) {
    help(argv[0]);
    return 1;
  }

  // init
  string hbkFile;
  int readMode = 1;
  int verbose = 0;
  string outFile;
  string resumeFile;
  string configFile;
  string valFile;
  bool scaleWeights = false;
  double sW = 1.;
  bool setLowCutoff = false;
  double lowCutoff = 0;
  string includeList, excludeList;
  string inputClassesString;
  int iLoss = 1;
  int iMode = 1;
  string stringVarsDoNotFeed;

  // decode command line
  int c;
  extern char* optarg;
  //  extern int optind;
  while( (c = getopt(argc,argv,"ho:a:y:e:c:g:m:v:f:r:t:V:z:Z:")) != EOF ) {
    switch( c )
      {
      case 'h' :
	help(argv[0]);
	return 1;
      case 'o' :
	hbkFile = optarg;
	break;
      case 'a' :
	readMode = (optarg==0 ? 1 : atoi(optarg));
	break;
      case 'y' :
	inputClassesString = optarg;
	break;
      case 'e' :
        iMode = (optarg==0 ? 1 : atoi(optarg));
        break;
      case 'c' :
	configFile = optarg;
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
      case 't' :
	valFile = optarg;
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

  // sanity check
  if( configFile.empty() && resumeFile.empty()) {
    cerr << "No classifier configuration file specified." << endl;
    return 1;
  }
  if( !configFile.empty() && !resumeFile.empty() ) {
    cerr << "Cannot train and use saved configuration at the same time." << endl;
    return 1;
  }

  // Must have 2 arguments after all options.
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

  // decode input classes
  if( inputClassesString.empty() ) {
    cerr << "No input classes specified." << endl;
    return 2;
  }
  vector<vector<int> > inputIntClasses;
  SprStringParser::parseToInts(inputClassesString.c_str(),inputIntClasses);
  if( inputIntClasses.empty() || inputIntClasses[0].size()<2 ) {
    cerr << "Found less than 2 classes in the input class string." << endl;
    return 2;
  }
  vector<SprClass> inputClasses(inputIntClasses[0].size());
  for( int i=0;i<inputIntClasses[0].size();i++ )
    inputClasses[i] = inputIntClasses[0][i];

  // filter training data by class
  filter->chooseClasses(inputClasses);
  if( !filter->filter() ) {
    cerr << "Unable to filter training data by class." << endl;
    return 2;
  }
  cout << "Training data filtered by class." << endl;
  for( int i=0;i<inputClasses.size();i++ ) {
    cout << "Points in class " << inputClasses[i] << ":   " 
	 << filter->ptsInClass(inputClasses[i]) << endl;
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
  if( !valFile.empty() ) {
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
    valFilter->chooseClasses(inputClasses);
    if( !valFilter->filter() ) {
      cerr << "Unable to filter validation data by class." << endl;
      return 2;
    }
    cout << "Validation data filtered by class." << endl;
    for( int i=0;i<inputClasses.size();i++ ) {
      cout << "Points in class " << inputClasses[i] << ":   " 
	   << valFilter->ptsInClass(inputClasses[i]) << endl;
    }
  }

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

  // prepare trained classifier holder
  auto_ptr<SprTrainedMultiClassLearner> trainedMulti;

  // prepare vectors of objects
  vector<SprAbsTwoClassCriterion*> criteria;
  vector<SprAbsClassifier*> destroyC;// classifiers to be deleted
  vector<SprIntegerBootstrap*> bstraps;
  vector<SprCCPair> useC;// classifiers and cuts to be used

  // open file with classifier configs
  if( !configFile.empty() ) {
    ifstream file(configFile.c_str());
    if( !file ) {
      cerr << "Unable to open file " << configFile.c_str() << endl;
      return 3;
    }
    
    // read classifier params
    unsigned nLine = 0;
    bool discreteTree = false;
    bool mixedNodesTree = true;
    bool readOneEntry = true;
    if( !SprClassifierReader::readTrainableConfig(file,nLine,filter.get(),
						  discreteTree,mixedNodesTree,
						  criteria,
						  bstraps,destroyC,useC,
						  readOneEntry) ) {
      cerr << "Unable to read classifier configurations from file " 
	   << configFile.c_str() << endl;
      prepareExit(criteria,destroyC,bstraps);
      return 3;
    }
    cout << "Finished reading " << useC.size() << " classifiers from file "
	 << configFile.c_str() << endl;
    assert( useC.size() == 1 );
    SprAbsClassifier* trainable = useC[0].first;

    // find the multi class mode
    SprMultiClassLearner::MultiClassMode multiClassMode 
      = SprMultiClassLearner::OneVsAll;
    switch( iMode )
      {
      case 1 :
        multiClassMode = SprMultiClassLearner::OneVsAll;
        cout << "Multi class learning mode set to OneVsAll." << endl;
        break;
      case 2 :
        multiClassMode = SprMultiClassLearner::OneVsOne;
      	cout << "Multi class learning mode set to OneVsOne." << endl;
  	break;
      default :
        cerr << "No multi class learning mode chosen." << endl;
        prepareExit(criteria,destroyC,bstraps);
        return 4;
      }

    // make a multi class learner
    SprMatrix indicator;
    SprMultiClassLearner multi(filter.get(),trainable,inputIntClasses[0],
       			       indicator,multiClassMode);

    // train
    if( resumeFile.empty() ) {
      if( !multi.train(verbose) ) {
        cerr << "Unable to train Multi class learner." << endl;
        prepareExit(criteria,destroyC,bstraps);
        return 5;
      }
      else {
        trainedMulti.reset(multi.makeTrained());
        cout << "Multi class learner finished successfully." << endl;
      }
    }

    // save trained multi class learner
    if( !outFile.empty() ) {
      if( !multi.store(outFile.c_str()) ) {
        cerr << "Cannot store multi class learner in file " 
	     << outFile.c_str() << endl;
        prepareExit(criteria,destroyC,bstraps);
        return 6;
      }
    }
  }

  // read saved learner from file
  if( !resumeFile.empty() ) {
    SprMultiClassReader multiReader;
    if( !multiReader.read(resumeFile.c_str()) ) {
      cerr << "Failed to read saved multi class learner from file " 
	   << resumeFile.c_str() << endl;
      prepareExit(criteria,destroyC,bstraps);
      return 7;
    }
    else {
      trainedMulti.reset(multiReader.makeTrained());
      cout << "Read saved multi class learner from file " 
	   << resumeFile.c_str() << endl;
      trainedMulti->printIndicatorMatrix(cout);
    }
  }

  // by now the trained learner should be filled
  if( trainedMulti.get() == 0 ) {
    cerr << "Trained multi learner has not been set." << endl;
    prepareExit(criteria,destroyC,bstraps);
    return 8;
  }

  // set loss
  switch( iLoss )
    {
    case 1 :
      trainedMulti->setLoss(&SprLoss::quadratic,
			    &SprTransformation::zeroOneToMinusPlusOne);
      cout << "Per-event loss set to "
           << "Quadratic loss (y-f(x))^2 " << endl;
      break;
    case 2 :
      trainedMulti->setLoss(&SprLoss::exponential,
			    &SprTransformation::logitInverse);
      cout << "Per-event loss set to "
           << "Exponential loss exp(-y*f(x)) " << endl;
      break;
    default :
      cerr << "No per-event loss specified." << endl;
      prepareExit(criteria,destroyC,bstraps);
      return 9;
    }

  // analyze validation data
  if( valFilter.get() != 0 ) {
    SprAverageLoss loss(&SprLoss::correct_id);
    map<int,double> output;
    for( int i=0;i<valFilter->size();i++ ) {
      const SprPoint* p = (*(valFilter.get()))[i];
      loss.update(p->class_,
		  double(trainedMulti->response(p,output)),
		  valFilter->w(i));
    }
    cout << "=====================================" << endl;
    cout << "Validation misid fraction = " << loss.value() << endl;
    cout << "=====================================" << endl;
  }

  // make histogram if requested
  if( hbkFile.empty() ) {
    prepareExit(criteria,destroyC,bstraps);
    return 0;
  }

  // make a writer
  SprMyWriter hbk("training");
  if( !hbk.init(hbkFile.c_str()) ) {
    cerr << "Unable to open output file " << hbkFile.c_str() << endl;
    prepareExit(criteria,destroyC,bstraps);
    return 10;
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
  feeder.addMultiClassLearner(trainedMulti.get(),"multi");
  if( !feeder.feed(1000) ) {
    cerr << "Cannot feed data into file " << hbkFile.c_str() << endl;
    prepareExit(criteria,destroyC,bstraps);
    return 11;
  }

  // cleanup
  prepareExit(criteria,destroyC,bstraps);

  // exit
  return 0;
}
