//$Id: SprBumpHunterApp.cc,v 1.4 2006/11/26 02:04:31 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBumpHunter.hh"
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
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassBgrndSmoother.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"

#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <memory>

using namespace std;


void help(const char* prog) 
{
  cout << "Usage:  " << prog 
       << " training_data_file" << endl;
  cout << "\t Options: " << endl;
  cout << "\t-h --- help                                        " << endl;
  cout << "\t-o output Tuple file                                 " << endl;
  cout << "\t-a input ascii file mode (see SprSimpleReader.hh)  " << endl;
  cout << "\t-n minimal number of events per bump (def=1)       " << endl;
  cout << "\t-b requested number of bumps (def=1)               " << endl;
  cout << "\t-x max fraction of events peeled off in one try    " << endl;
  cout << "\t-y list of input classes (see SprAbsFilter.hh)     " << endl;
  cout << "\t-v verbose level (0=silent default,1,2)            " << endl;
  cout << "\t-f store trained bump hunter to file               " << endl;
  cout << "\t-c criterion for optimization                      " << endl;
  cout << "\t\t 1 = correctly classified fraction               " << endl;
  cout << "\t\t 2 = signal significance s/sqrt(s+b)             " << endl;
  cout << "\t\t 3 = purity s/(s+b) (default)                    " << endl;
  cout << "\t\t 4 = tagger efficiency Q                         " << endl;
  cout << "\t\t 5 = Gini index                                  " << endl;
  cout << "\t\t 6 = cross-entropy                               " << endl;
  cout << "\t\t 7 = 90% Bayesian upper limit with uniform prior " << endl;
  cout << "\t\t 8 = discovery potential 2*(sqrt(s+b)-sqrt(b))   " << endl;
  cout << "\t\t 9 = Punzi's sensitivity s/(0.5*nSigma+sqrt(b))  " << endl;
  cout << "\t\t 10= background-smoothed Punzi's sensitivity     " << endl;
  cout << "\t\t -P background normalization factor for Punzi FOM" << endl;
  cout << "\t\t -L lambda for the background-smoothed FOM       " << endl;
  cout << "\t\t -O omega for the background-smoothed FOM        " << endl;
  cout << "\t-t read validation/test data from a file           " << endl;
  cout << "\t\t (must be in same format as input data!!!        " << endl;
  cout << "\t-p output file to store validation/test data       " << endl;
  cout << "\t-w scale all signal weights by this factor         " << endl;
  cout << "\t-V include only these input variables              " << endl;
  cout << "\t-z exclude input variables from the list           " << endl;
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
  string hbkFile;
  int readMode = 1;
  unsigned nmin = 1;
  int verbose = 0;
  string outFile;
  string resumeFile;
  int iCrit = 3;
  string valFile;
  string valHbkFile;
  int nbump = 1;
  double apeel = 1.;
  bool scaleWeights = false;
  double sW = 1.;
  string includeList, excludeList;
  string inputClassesString;
  double bW = 1.;
  double lambda = 2.;
  double omega = 5.;
   
  // decode command line
  int c;
  extern char* optarg;
  while( (c = getopt(argc,argv,"ho:a:n:v:f:c:P:L:O:t:p:b:x:y:w:V:z:")) != EOF ) {
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
      case 'n' :
	nmin = (optarg==0 ? 1 : atoi(optarg));
	break;
      case 'v' :
	verbose = (optarg==0 ? 0 : atoi(optarg));
	break;
      case 'f' :
	outFile = optarg;
	break;
      case 'c' :
        iCrit = (optarg==0 ? 3 : atoi(optarg));
        break;
      case 'P' :
	bW = (optarg==0 ? 1. : atof(optarg));
	break;
      case 'L' :
	lambda = (optarg==0 ? 2. : atof(optarg));
	break;
      case 'O' :
	omega = (optarg==0 ? 5. : atof(optarg));
	break;
      case 't' :
        valFile = optarg;
        break;
      case 'p' :
        valHbkFile = optarg;
        break;
      case 'b' :
	nbump = (optarg==0 ? 1 : atoi(optarg));
	break;
      case 'x' :
	apeel = (optarg==0 ? 1. : atof(optarg));
	break;
      case 'y' :
	inputClassesString = optarg;
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
    case 10 :
      crit.reset(new SprTwoClassBgrndSmoother(bW,lambda,omega));
      cout << "Optimization criterion set to "
	   << "background-smoothed Punzi's sensitivity" << endl;
      break;
    default :
      cerr << "Unable to make initialization criterion." << endl;
      return 3;
    }

  // make decision tree
  SprBumpHunter bump(filter.get(),crit.get(),nbump,nmin,apeel);

  // train
  if( !bump.train(verbose) ) {
    cerr << "Unable to find bumps." << endl;
    return 4;
  }

  // save trained decision tree
  if( !outFile.empty() ) {
    if( !bump.store(outFile.c_str()) ) {
      cerr << "Cannot store bump hunter in file " << outFile.c_str() << endl;
      return 5;
    }
  }

  // make trained decision tree
  auto_ptr<SprTrainedDecisionTree> trainedTree(bump.makeTrained());

  // compute FOM for the validation data
  if( valFilter.get() != 0 ) {
    double wcor0(0), wmis0(0), wcor1(0), wmis1(0);
    int ncor0(0), nmis0(0), ncor1(0), nmis1(0);
    for( int i=0;i<valFilter->size();i++ ) {
      const SprPoint* p = (*valFilter.get())[i];
      double w = valFilter->w(i);
      if( trainedTree->accept(p) ) {
	if(      p->class_ == inputClasses[0] ) {
	  wmis0 += w;
	  nmis0++;
	}
	else if( p->class_ == inputClasses[1] ) {
	  wcor1 += w;
	  ncor1++;
	}
      }
      else {
	if(      p->class_ == inputClasses[0] ) {
	  wcor0 += w;
	  ncor0++;
	}
	else if( p->class_ == inputClasses[1] ) {
	  wmis1 += w;
	  nmis1++;
	}
      }
    }
    double vFom = crit->fom(wcor0,wmis0,wcor1,wmis1);
    cout << "=====================================================" << endl;
    cout << "Validation FOM=" << vFom << endl;
    cout << "Content of the signal region:"
	 << "   W0=" << wmis0 << "  W1=" << wcor1 
	 << "   N0=" << nmis0 << "  N1=" << ncor1 
	 << endl;
    cout << "=====================================================" << endl;
  }

  // make histogram if requested
  if( hbkFile.empty() && valHbkFile.empty() ) return 0;

  // make a wrapper to store box numbers
  class BoxNumberWrapper : public SprTrainedDecisionTree {
  public:
    virtual ~BoxNumberWrapper() {}
    BoxNumberWrapper(const SprTrainedDecisionTree& tree)
      : SprTrainedDecisionTree(tree) {}
    double response(const std::vector<double>& v) const {
      return this->nBox(v);
    }
  };

  // feed training data
  if( !hbkFile.empty() ) {
    // make a writer
    SprMyWriter hbk("training");
    if( !hbk.init(hbkFile.c_str()) ) {
      cerr << "Unable to open output file " << hbkFile.c_str() << endl;
      return 6;
    }
    // wrap
    BoxNumberWrapper boxNumber(*(trainedTree.get()));
    // feed 
    SprDataFeeder feeder(filter.get(),&hbk);
    feeder.addClassifier(trainedTree.get(),"bump");
    feeder.addClassifier(&boxNumber,"box");
    if( !feeder.feed(1000) ) {
      cerr << "Cannot feed data into file " << hbkFile.c_str() << endl;
      return 6;
    }
  }

  if( !valHbkFile.empty() ) {
    // make a writer
    SprMyWriter hbk("training");
    if( !hbk.init(valHbkFile.c_str()) ) {
      cerr << "Unable to open output file " << valHbkFile.c_str() << endl;
      return 7;
    }
    // wrap
    BoxNumberWrapper boxNumber(*(trainedTree.get()));
    // feed 
    SprDataFeeder feeder(valFilter.get(),&hbk);
    feeder.addClassifier(trainedTree.get(),"bump");
    feeder.addClassifier(&boxNumber,"box");
    if( !feeder.feed(1000) ) {
      cerr << "Cannot feed data into file " << valHbkFile.c_str() << endl;
      return 7;
    }
  }

  // exit
  return 0;
}
