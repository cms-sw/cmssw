//$Id: SprMultiClassApp.cc,v 1.5 2007/11/12 06:19:11 narsky Exp $
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
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsWriter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDataFeeder.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRWFactory.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerBootstrap.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformation.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAverageLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprMultiClassReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprMultiClassPlotter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsVarTransformer.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprVarTransformerReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformerFilter.hh"

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
  cout << "\t-A save output data in ascii instead of Root            " << endl;
  cout << "\t-y list of input classes                                " << endl;
  cout << "\t\t Classes must be listed in quotes and separated by commas." 
       << endl;
  cout << "\t-Q apply variable transformation saved in file     " << endl;
  cout << "\t-e Multi class mode                                     " << endl;
  cout << "\t\t 1 - OneVsAll (default)                               " << endl;
  cout << "\t\t 2 - OneVsOne                                         " << endl;
  cout << "\t\t 3 - user-defined (must use -i option)                " << endl;
  cout << "\t-i input file with user-defined indicator matrix        " << endl;
  cout << "\t-c file with trainable classifier configurations        " << endl;
  cout << "\t-g per-event loss to be displayed for each input class  " << endl;
  cout << "\t\t 1 - quadratic loss (y-f(x))^2                        " << endl;
  cout << "\t\t 2 - exponential loss exp(-y*f(x))                    " << endl;
  cout << "\t-m replace data values below this cutoff with medians   " << endl;
  cout << "\t-v verbose level (0=silent default,1,2)                 " << endl;
  cout << "\t-f store trained multi class learner to file            " << endl;
  cout << "\t-r read multi class learner configuration stored in file" << endl;
  cout << "\t-K keep this fraction in training set and          " << endl;
  cout << "\t\t put the rest into validation set                " << endl;
  cout << "\t-D randomize training set split-up                 " << endl;
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
  for( unsigned int i=0;i<criteria.size();i++ ) delete criteria[i];
  for( unsigned int i=0;i<classifiers.size();i++ ) delete classifiers[i];
  for( unsigned int i=0;i<bstraps.size();i++ ) delete bstraps[i];
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
  string indicatorFile;
  string stringVarsDoNotFeed;
  bool split = false;
  double splitFactor = 0;
  bool splitRandomize = false;
  string transformerFile;

  // decode command line
  int c;
  extern char* optarg;
  //  extern int optind;
  while( (c = getopt(argc,argv,"ho:a:Ay:Q:e:i:c:g:m:v:f:r:K:Dt:V:z:Z:")) != EOF ) {
    switch( c )
      {
      case 'h' :
	help(argv[0]);
	return 1;
      case 'o' :
	tupleFile = optarg;
	break;
      case 'a' :
	readMode = (optarg==0 ? 0 : atoi(optarg));
	break;
      case 'A' :
	writeMode = SprRWFactory::Ascii;
	break;
      case 'y' :
	inputClassesString = optarg;
	break;
      case 'Q' :
        transformerFile = optarg;
        break;
      case 'e' :
        iMode = (optarg==0 ? 1 : atoi(optarg));
        break;
      case 'i' :
	indicatorFile = optarg;
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
  for( unsigned int i=0;i<inputIntClasses[0].size();i++ )
    inputClasses[i] = inputIntClasses[0][i];

  // filter training data by class
  filter->chooseClasses(inputClasses);
  if( !filter->filter() ) {
    cerr << "Unable to filter training data by class." << endl;
    return 2;
  }
  cout << "Training data filtered by class." << endl;
  for( unsigned int i=0;i<inputClasses.size();i++ ) {
    unsigned npts = filter->ptsInClass(inputClasses[i]);
    if( npts == 0 ) {
      cerr << "Error!!! No points in class " << inputClasses[i] << endl;
      return 2;
    }
    cout << "Points in class " << inputClasses[i] << ":   " << npts << endl;
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
  }

  // filter validation data by class
  if( valFilter.get() != 0 ) {
    valFilter->chooseClasses(inputClasses);
    if( !valFilter->filter() ) {
      cerr << "Unable to filter validation data by class." << endl;
      return 2;
    }
    cout << "Validation data filtered by class." << endl;
    for( unsigned int i=0;i<inputClasses.size();i++ ) {
     unsigned npts = valFilter->ptsInClass(inputClasses[i]);
     if( npts == 0 )
       cerr << "Warning!!! No points in class " << inputClasses[i] << endl;
     cout << "Points in class " << inputClasses[i] << ":   " << npts << endl;
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
    bool mixedNodesTree = false;
    bool fastSort = false;
    bool readOneEntry = true;
    if( !SprClassifierReader::readTrainableConfig(file,nLine,filter.get(),
						  discreteTree,mixedNodesTree,
						  fastSort,criteria,
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
      case 3:
	if( indicatorFile.empty() ) {
	  cerr << "No indicator matrix specified." << endl;
	  return 4;
	}
	multiClassMode = SprMultiClassLearner::User;
	cout << "Multi class learning mode set to User." << endl;
	break;
      default :
        cerr << "No multi class learning mode chosen." << endl;
        prepareExit(criteria,destroyC,bstraps);
        return 4;
      }

    // get indicator matrix
    SprMatrix indicator;
    if( multiClassMode==SprMultiClassLearner::User 
	&& !indicatorFile.empty() ) {
      if( !SprMultiClassReader::readIndicatorMatrix(indicatorFile.c_str(),
						    indicator) ) {
	cerr << "Unable to read indicator matrix from file " 
	     << indicatorFile.c_str() << endl;
	return 4;
      }
    }

    // make a multi class learner
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

    // compute response
    vector<SprMultiClassPlotter::Response> responses(valFilter->size());
    for( unsigned int i=0;i<valFilter->size();i++ ) {
      if( ((i+1)%1000) == 0 )
	cout << "Computing response for validation point " << i+1 << endl;

      // get point, class and weight
      const SprPoint* p = (*(valFilter.get()))[i];
      int cls = p->class_;
      double w = valFilter->w(i);

      // compute loss
      map<int,double> output;
      int resp = trainedMulti->response(p,output);
      responses[i] = SprMultiClassPlotter::Response(cls,w,resp,output);
    }    

    // get the loss table
    SprMultiClassPlotter plotter(responses);
    vector<int> classes;
    trainedMulti->classes(classes);
    map<int,vector<double> > lossTable;
    map<int,double> weightInClass;
    double totalLoss = plotter.multiClassTable(classes,lossTable,
					       weightInClass);

    // print out
    cout << "=====================================" << endl;
    cout << "Overall validation misid fraction = " << totalLoss << endl;
    cout << "=====================================" << endl;
    cout << "Classification table: Fractions of total class weight" << endl;
    char s[200];
    sprintf(s,"True Class \\ Classification |");
    string temp = "------------------------------";
    cout << s;
    for( unsigned int i=0;i<classes.size();i++ ) {
      sprintf(s," %5i      |",classes[i]);
      cout << s;
      temp += "-------------";
    }
    sprintf(s,"   Total weight in class |");
    temp += "-------------------------";
    cout << s << endl;
    cout << temp.c_str() << endl;
    for( map<int,vector<double> >::const_iterator
	   i=lossTable.begin();i!=lossTable.end();i++ ) {
      sprintf(s,"%5i                       |",i->first);
      cout << s;
      for( unsigned int j=0;j<i->second.size();j++ ) {
	sprintf(s," %10.4f |",i->second[j]);
	cout << s;
      }
      sprintf(s,"              %10.4f |",weightInClass[i->first]);
      cout << s << endl;
    }
    cout << temp.c_str() << endl;
  }

  // make histogram if requested
  if( tupleFile.empty() ) {
    prepareExit(criteria,destroyC,bstraps);
    return 0;
  }

  // make a writer
  auto_ptr<SprAbsWriter> tuple(SprRWFactory::makeWriter(writeMode,"training"));
  if( !tuple->init(tupleFile.c_str()) ) {
    cerr << "Unable to open output file " << tupleFile.c_str() << endl;
    prepareExit(criteria,destroyC,bstraps);
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
  feeder.addMultiClassLearner(trainedMulti.get(),"multi");
  if( !feeder.feed(1000) ) {
    cerr << "Cannot feed data into file " << tupleFile.c_str() << endl;
    prepareExit(criteria,destroyC,bstraps);
    return 11;
  }

  // cleanup
  prepareExit(criteria,destroyC,bstraps);

  // exit
  return 0;
}
