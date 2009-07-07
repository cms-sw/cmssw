//$Id: SprExploratoryAnalysisApp.cc,v 1.4 2007/11/12 06:19:11 narsky Exp $
/*
  This executable is intended for exploratory analysis of data.

  First, it computes correlations between input variables, separately
  for signal and background.

  Second, it computes a correlation between each input variable and the true
  class label. This can help the user to select most powerful discriminating
  variables.

  Then the executable finds the best two-sided interval for each variable
  to optimize the chosen figure of merit. As an option, the executable
  computes correlations between the found intervals.
*/

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBumpHunter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedDecisionTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRWFactory.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassSignalSignif.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassIDFraction.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassTaggerEff.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassPurity.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassGiniIndex.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassCrossEntropy.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassUniformPriorUL90.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassBKDiscovery.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassPunzi.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDataMoments.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsVarTransformer.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprVarTransformerReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformerFilter.hh"
#include "PhysicsTools/StatPatternRecognition/src/SprSymMatrix.hh"
#include "PhysicsTools/StatPatternRecognition/src/SprVector.hh"

#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <functional>
#include <utility>
#include <iomanip>
#include <cmath>
#include <memory>

using namespace std;


struct SEACmpPairFirst
  : public binary_function<pair<double,int>,pair<double,int>,bool> {
  bool operator()(const pair<double,int>& l, const pair<double,int>& r)
    const {
    return (l.first < r.first);
  }
};
 

void cleanup(vector<const SprTrainedDecisionTree*>& trained)
{
  for( unsigned int i=0;i<trained.size();i++ )
    delete trained[i];
}


void help(const char* prog) 
{
  cout << "Usage:  " << prog 
       << " training_data_file" << endl;
  cout << "\t Options: " << endl;
  cout << "\t-h --- help                                        " << endl;
  cout << "\t-a input ascii file mode (see SprSimpleReader.hh)  " << endl;
  cout << "\t-y list of input classes (see SprAbsFilter.hh)     " << endl;
  cout << "\t-Q apply variable transformation saved in file     " << endl;
  cout << "\t-c criterion for optimization                      " << endl;
  cout << "\t\t 1 = correctly classified fraction (default)     " << endl;
  cout << "\t\t 2 = signal significance s/sqrt(s+b)             " << endl;
  cout << "\t\t 3 = purity s/(s+b)                              " << endl;
  cout << "\t\t 4 = tagger efficiency Q                         " << endl;
  cout << "\t\t 5 = Gini index                                  " << endl;
  cout << "\t\t 6 = cross-entropy                               " << endl;
  cout << "\t\t 7 = 90% Bayesian upper limit with uniform prior " << endl;
  cout << "\t\t 8 = discovery potential 2*(sqrt(s+b)-sqrt(b))   " << endl;
  cout << "\t\t 9 = Punzi's sensitivity s/(0.5*nSigma+sqrt(b))  " << endl;
  cout << "\t\t -P background normalization factor for Punzi FOM" << endl;
  cout << "\t-r compute correlations among intervals            " << endl;
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
  string tupleFile;
  int readMode = 0;
  int iCrit = 1;
  bool computeCorr = false;
  bool scaleWeights = false;
  double sW = 1.;
  string includeList, excludeList;
  string inputClassesString;
  double bW = 1.;
  string transformerFile;

  // decode command line
  int c;
  extern char* optarg;
  //  extern int optind;
  while( (c = getopt(argc,argv,"ha:y:Q:c:P:rw:V:z:")) != EOF ) {
    switch( c )
      {
      case 'h' :
	help(argv[0]);
	return 1;
      case 'a' :
	readMode = (optarg==0 ? 0 : atoi(optarg));
	break;
      case 'y' :
	inputClassesString = optarg;
	break;
      case 'Q' :
        transformerFile = optarg;
        break;
      case 'c' :
	iCrit = (optarg==0 ? 1 : atoi(optarg));
	break;
      case 'P' :
	bW = (optarg==0 ? 1 : atof(optarg));
	break;
      case 'r' :
	computeCorr = true;
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

  // print out variables
  assert( vars.size() == filter->dim() );
  cout << "=================================" << endl;
  cout << "Input variables:" << endl;
  for( unsigned int i=0;i<vars.size();i++ )
    cout << i << " " << vars[i] << endl;
  cout << "=================================" << endl;

  // scale weights
  if( scaleWeights )
    filter->scaleWeights(inputClasses[1],sW);

  // apply transformation of variables to data
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

  // always compute covariance matrix for all supplied variables
  SprSymMatrix cov;
  SprVector mean;
  SprDataMoments moms(filter.get());

  // identify useless variables
  if( !moms.covariance(cov,mean) ) {
    cerr << "Unable to compute covariance matrix for entire data." << endl;
    return 4;
  }
  cout << "Variables with zero variance:    ";
  for( unsigned int i=0;i<vars.size();i++ ) {
    if( cov[i][i] < SprUtils::eps() ) 
      cout << vars[i].c_str() << ",";
  }
  cout << endl;

  // do background and signal
  for( unsigned int c=0;c<2;c++ ) {
    vector<SprClass> classes(1);
    classes[0] = inputClasses[c];
    filter->chooseClasses(classes);
    if( !filter->filter() ) {
      cerr << "Unable to filter class " << c << endl;
      return 4;
    }
    if( !moms.covariance(cov,mean) ) {
      cerr << "Unable to compute covariance matrix for input variables." 
	   << endl;
      return 4;
    }
    // output
    cout << "===============================================" << endl;
    cout << "Covariance matrix computed with " 
	 << filter->ptsInClass(classes[0]) << " events." << endl;
    cout << "Input variable correlations in class " << c << ":" << endl;
    cout << "Column  ";
    for( unsigned int i=0;i<filter->dim();i++ )
      cout << setw(10) << i << " ";
    cout << endl;
    cout << "--------";
    for( unsigned int i=0;i<filter->dim();i++ )
      cout << setw(10) << "----------" << "-";
    cout << endl;
    for( unsigned int i=0;i<filter->dim();i++ ) {
      cout << "Row " << i << " |    ";
      for( unsigned int j=0;j<filter->dim();j++ )
	cout << setw(10) << cov[i][j]/sqrt(cov[i][i])/sqrt(cov[j][j]) << " ";
      cout << endl;
    }
    cout << "===============================================" << endl;
  }
  filter->clear();

  // compute correlation with the class label
  vector<double> corrLabel(filter->dim()); 
  vector<pair<double,int> > absCorrLabel(filter->dim());
  double meani(0), vari(0);
  for( unsigned int i=0;i<filter->dim();i++ ) {
    corrLabel[i] = moms.correlClassLabel(i,meani,vari);
    absCorrLabel[i] = pair<double,int>(fabs(corrLabel[i]),i);
  }
  stable_sort(absCorrLabel.begin(),absCorrLabel.end(),not2(SEACmpPairFirst()));
  cout << "===============================================" << endl;
  cout << "Correlations with class label:" << endl;
  for( unsigned int i=0;i<filter->dim();i++ ) {
    int k = absCorrLabel[i].second;
    cout << setw(40) << vars[k] << " " << setw(10) << corrLabel[k] << endl;
  }
  cout << "===============================================" << endl;

  // compute correlation of the absolute value with the class label
  vector<double> corrLabel2(filter->dim()); 
  vector<pair<double,int> > absCorrLabel2(filter->dim());
  double meani2(0), vari2(0);
  for( unsigned int i=0;i<filter->dim();i++ ) {
    corrLabel2[i] = moms.absCorrelClassLabel(i,meani2,vari2);
    absCorrLabel2[i] = pair<double,int>(fabs(corrLabel2[i]),i);
  }
  stable_sort(absCorrLabel2.begin(),absCorrLabel2.end(),
	      not2(SEACmpPairFirst()));
  cout << "===============================================" << endl;
  cout << "Correlations of absolute values with class label:" << endl;
  for( unsigned int i=0;i<filter->dim();i++ ) {
    int k = absCorrLabel2[i].second;
    cout << setw(40) << vars[k] << " " << setw(10) << corrLabel2[k] << endl;
  }
  cout << "===============================================" << endl;

  // find optimal 1D intervals
  vector<pair<double,int> > fom(filter->dim(),
				pair<double,int>(SprUtils::min(),-1));
  vector<const SprTrainedDecisionTree*> trained(filter->dim(),0);
  vector<double> w1vec(filter->dim()), w0vec(filter->dim());
  // prepare dummy 1D data
  SprData tempData("myDummy1Ddata",vector<string>(1,"dummy"));
  vector<double> x(1);
  for( unsigned int j=0;j<filter->size();j++ ) {
    const SprPoint* p = (*filter.get())[j];
    x[0] = p->x_[0];
    tempData.insert(p->index_,p->class_,x);
  }
  // get weights
  vector<double> weights;
  filter->weights(weights);
  // make dummy filter
  SprEmptyFilter tempFilter(&tempData,weights);
  tempFilter.chooseClasses(inputClasses);
  // loop through dimensions
  for( unsigned int d=0;d<filter->dim();d++ ) {
    if( d != 0 ) {
      for( unsigned int j=0;j<filter->size();j++ )
	tempFilter[j]->x_[0] = (*filter.get())[j]->x_[d];
    }
    // make new hunter
    cout << "Optimizing interval in dimension " << d << endl;
    SprBumpHunter hunter(&tempFilter,crit.get(),1,int(0.01*filter->size()),1.);
    if( !hunter.train() ) {
      cerr << "Unable to train interval for dimension " << d << endl;
      continue;
    }
    const SprTrainedDecisionTree* t = hunter.makeTrained();
    trained[d] = t;
    // count accepted and rejected events
    double wmis0(0), wcor0(0), wmis1(0), wcor1(0);
    for( unsigned int j=0;j<filter->size();j++ ) {
      const SprPoint* p = tempFilter[j];
      double w = tempFilter.w(j);
      if(      p->class_ == inputClasses[0] ) {
	if( t->accept(p) )
	  wmis0 += w;
	else
	  wcor0 += w;
      }
      else if( p->class_ == inputClasses[1] ) {
	if( t->accept(p) )
	  wcor1 += w;
	else
	  wmis1 += w;
      }
    }
    fom[d] = pair<double,int>(crit->fom(wcor0,wmis0,wcor1,wmis1),d);
    w1vec[d] = wcor1;
    w0vec[d] = wmis0;
  }

  // sort FOMs
  stable_sort(fom.begin(),fom.end(),not2(SEACmpPairFirst()));

  // print out boxes
  double w0 = filter->weightInClass(inputClasses[0]);
  double w1 = filter->weightInClass(inputClasses[1]);
  double fmin = crit->fom(0,w0,w1,0);
  double fmax = crit->fom(0,0,w1,0);
  cout << "Possible FOM range: " << fmin << " " << fmax << endl;
  for( unsigned int i=0;i<filter->dim();i++ ) {
    SprBox limits;
    int k = fom[i].second;
    if( k>=0 && trained[k]!=0 ) trained[k]->box(0,limits);
    SprBox::const_iterator iter = limits.find(0);
    if( iter != limits.end() ) {
      cout << i << "   FOM= " << setw(8) << fom[i].first 
	   << " for variable \"" << setw(15) << vars[k] << "\""
	   << " with acceptance interval " 
	   << setw(10) << iter->second.first << " " 
	   << setw(10) << iter->second.second 
	   << "    W0=" << w0vec[k] << "  W1=" << w1vec[k] << endl;
    }
  }

  // compute correlations
  if( computeCorr ) {
    SprSymMatrix corr(filter->dim());
    for( unsigned int i=0;i<filter->dim();i++ ) {
      int c1 = fom[i].second;
      if( c1<0 || trained[c1]==0 ) {
	cerr << "Unable to compute correlations: "
	     << "There are uncomputed intervals." << endl;
	cleanup(trained);
	return 5;
      }
      for( unsigned int j=i+1;j<filter->dim();j++ ) {
	int c2 = fom[j].second;
	if( c2<0 || trained[c2]==0 ) {
	  cerr << "Unable to compute correlations: "
	       << "There are uncomputed intervals." << endl;
	  cleanup(trained);
	  return 5;
	}
	double a(0), b(0), c(0), d(0);
	for( unsigned int k=0;k<filter->size();k++ ) {
	  const SprPoint* p = (*filter.get())[k];
	  double w = filter->w(k);
	  vector<double> x1(1), x2(1);
	  x1[0] = p->x_[c1];
	  x2[0] = p->x_[c2];
	  if(      p->class_ == inputClasses[0] ) {
	    if( trained[c1]->accept(x1) ) {
	      if( trained[c2]->accept(x2) )
		d += w;
	      else
		c += w;
	    }
	    else {
	      if( trained[c2]->accept(x2) )
		b += w;
	      else
		a += w;
	    }
	  }
	  else if( p->class_ == inputClasses[1] ) {
	    if( trained[c1]->accept(x1) ) {
	      if( trained[c2]->accept(x2) )
		a += w;
	      else
		b += w;
	    }
	    else {
	      if( trained[c2]->accept(x2) )
		c += w;
	      else
		d += w;
	    }
	  }
	}
	if( (a+b)<SprUtils::eps() || (c+d)<SprUtils::eps()
	    || (a+c)<SprUtils::eps() || (b+d)<SprUtils::eps() ) {
	  cerr << "Unable to compute correlations: One of the sums is zero." 
	       << endl;
	  cleanup(trained);
	  return 5;
	}
	corr[i][j] = (a*d-b*c) / sqrt((a+b)*(c+d)*(a+c)*(b+d));
      }
      corr[i][i] = 1;
    }
    // output
    cout << "Interval correlations: " << endl;
    cout << "Column  ";
    for( unsigned int i=0;i<filter->dim();i++ )
      cout << setw(10) << i << " ";
    cout << endl;
    cout << "--------";
    for( unsigned int i=0;i<filter->dim();i++ )
      cout << setw(10) << "----------" << "-";
    cout << endl;
    for( unsigned int i=0;i<filter->dim();i++ ) {
      cout << "Row " << i << " |    ";
      for( unsigned int j=0;j<filter->dim();j++ )
	cout << setw(10) << corr[i][j] << " ";
      cout << endl;
    }
  }

  // clean up
  cleanup(trained);

  // exit
  return 0;
}
