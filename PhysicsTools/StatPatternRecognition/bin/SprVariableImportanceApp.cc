//$Id: SprVariableImportanceApp.cc,v 1.1 2007/07/06 21:46:22 narsky Exp $
//
// An executable to estimate the relative importance of variables.
// See notes in README (Variable Selection).
//

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprSimpleReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprCoordinateMapper.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedFisher.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedLogitR.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAverageLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformation.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerPermutator.hh"

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <set>
#include <vector>
#include <memory>
#include <string>
#include <cassert>
#include <algorithm>
#include <utility>
#include <functional>

using namespace std;


// sorts in descending order
struct SVICmpPairSDSecond
  : public binary_function<pair<string,double>,pair<string,double>,bool> {
  bool operator()(const pair<string,double>& l, const pair<string,double>& r)
    const {
    return (l.second > r.second);
  }
};


void help(const char* prog) 
{
  cout << "Usage:  " << prog << " classifier_config_file"
       << " input_data_file" << endl;
  cout << "\t Options: " << endl;
  cout << "\t-h --- help                                        " << endl;
  cout << "\t-y list of input classes (see SprAbsFilter.hh)     " << endl;
  cout << "\t-a input ascii file mode (see SprSimpleReader.hh)  " << endl;
  cout << "\t-v verbose level (0=silent default,1,2)            " << endl;
  cout << "\t-n number of class permutations per variable (def=1)" << endl;
  cout << "\t-w scale all signal weights by this factor         " << endl;
  cout << "\t-g per-event loss for (cross-)validation           " << endl;
  cout << "\t\t 1 - quadratic loss (y-f(x))^2                   " << endl;
  cout << "\t\t 2 - exponential loss exp(-y*f(x))               " << endl;
  cout << "\t-V include only these input variables              " << endl;
  cout << "\t-z exclude input variables from the list           " << endl;
  cout << "\t-M map variable lists from trained classifiers onto" << endl;
  cout << "\t\t variables available in input data."               << endl;
  cout << "\t\t Variables must be listed in quotes and separated by commas." 
       << endl;
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
  int verbose = 0;
  bool scaleWeights = false;
  double sW = 1.;
  string includeList, excludeList;
  string inputClassesString;
  bool mapTrainedVars = false;
  int iLoss = 0;
  int nPerm = 1;

  // decode command line
  int c;
  extern char* optarg;
  extern int optind;
  while( (c = getopt(argc,argv,"hy:a:n:v:w:g:V:z:M")) != EOF ) {
    switch( c )
      {
      case 'h' :
	help(argv[0]);
	return 1;
      case 'y' :
	inputClassesString = optarg;
	break;
      case 'a' :
	readMode = (optarg==0 ? 1 : atoi(optarg));
	break;
      case 'n' :
	nPerm = (optarg==0 ? 1 : atoi(optarg));
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
      case 'g' :
        iLoss = (optarg==0 ? 0 : atoi(optarg));
        break;
      case 'V' :
	includeList = optarg;
	break;
      case 'z' :
	excludeList = optarg;
	break;
      case 'M' :
	mapTrainedVars = true;
	break;
      }
  }

  // Must have 3 arguments on the command line
  string configFile     = argv[argc-2];
  string dataFile       = argv[argc-1];
  if( configFile.empty() ) {
    cerr << "No classifier configuration file is specified." << endl;
    return 1;
  }
  if( dataFile.empty() ) {
    cerr << "No input data file is specified." << endl;
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

  // read input data from file
  auto_ptr<SprAbsFilter> filter(reader.read(dataFile.c_str()));
  if( filter.get() == 0 ) {
    cerr << "Unable to read data from file " << dataFile.c_str() << endl;
    return 2;
  }
  vector<string> vars;
  filter->vars(vars);
  cout << "Read data from file " << dataFile.c_str() << " for variables";
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

  // make per-event loss
  auto_ptr<SprAverageLoss> loss;
  bool useStandard = false;
  switch( iLoss )
    {
    case 1 :
      loss.reset(new SprAverageLoss(&SprLoss::quadratic));
      cout << "Per-event loss set to "
           << "Quadratic loss (y-f(x))^2 " << endl;
      useStandard = false;
      break;
    case 2 :
      loss.reset(new SprAverageLoss(&SprLoss::exponential));
      cout << "Per-event loss set to "
           << "Exponential loss exp(-y*f(x)) " << endl;
      useStandard = true;
      break;
    default :
      loss.reset(new SprAverageLoss(&SprLoss::quadratic));
      cout << "No per-event loss is chosen. Will use quadratic." << endl;
      break;
    }

  // read classifier configuration
  auto_ptr<SprAbsTrainedClassifier> 
    trained(SprClassifierReader::readTrained(configFile.c_str(),verbose));
  if( trained.get() == 0 ) {
    cerr << "Unable to read classifier configuration from file "
	 << configFile.c_str() << endl;
    return 3;
  }
  cout << "Read classifier " << trained->name().c_str()
       << " with dimensionality " << trained->dim() << endl;

  // get a list of trained variables
  vector<string> trainedVars;
  trained->vars(trainedVars);
  if( verbose > 0 ) {
    cout << "Variables:      " << endl;
    for( int j=0;j<trainedVars.size();j++ ) 
      cout << trainedVars[j].c_str() << " ";
    cout << endl;
  }

  // map trained-classifier variables onto data variables
  auto_ptr<SprCoordinateMapper> mapper;
  if( mapTrainedVars || trained->name()=="Combiner" ) {
    mapper.reset(SprCoordinateMapper::createMapper(trainedVars,vars));
    if( mapper.get() == 0 ) {
      cerr << "Unable to map trained classifier vars onto data vars." << endl;
      return 4;
    }
  }

  // switch classifier output range
  if( useStandard ) {
    if(      trained->name() == "AdaBoost" ) {
      SprTrainedAdaBoost* specific 
	= static_cast<SprTrainedAdaBoost*>(trained.get());
      specific->useStandard();
    }
    else if( trained->name() == "Fisher" ) {
      SprTrainedFisher* specific 
	= static_cast<SprTrainedFisher*>(trained.get());
      specific->useStandard();
    }
    else if( trained->name() == "LogitR" ) {
      SprTrainedLogitR* specific 
	= static_cast<SprTrainedLogitR*>(trained.get());
      specific->useStandard();
    }
  }

  //
  // pass through all variables
  //
  bool first = true;
  int N = filter->size();
  SprIntegerPermutator permu(N);
  int nVars = trainedVars.size();
  vector<pair<string,double> > losses;

  // make first pass without permutations
  for( int n=0;n<N;n++ ) {
    const SprPoint* p = (*(filter.get()))[n];
    const SprPoint* mappedP = p;
    int icls = -1;
    if(      p->class_ == inputClasses[0] )
      icls = 0;
    else if( p->class_ == inputClasses[1] )
      icls = 1;
    else
      continue;
    if( mapper.get() != 0 ) mappedP = mapper->output(p);
    loss->update(icls,trained->response(mappedP),filter->w(n));
    if(  mapper.get() != 0 ) mapper->clear();
  }
  double nominalLoss = loss->value();

  // loop over permutations
  cout << "Will perform " << nPerm << " permutations per variable." << endl;
  for( int d=0;d<nVars;d++ ) {
    cout << "Permuting variable " << trainedVars[d].c_str() << endl;

    // map this var
    int mappedD = d;
    if( mapper.get() != 0 )
      mappedD = mapper->mappedIndex(d);
    assert( mappedD>=0 && mappedD<filter->dim() );

    // pass through all points permuting them
    double aveLoss = 0;
    for( int i=0;i<=nPerm;i++ ) {

      // permute this variable
      vector<unsigned> seq;
      if( !permu.sequence(seq) ) {
	cerr << "Unable to permute points." << endl;
	return 5;
      }
      
      // pass through points
      loss->reset();
      for( int n=0;n<N;n++ ) {
	SprPoint p(*(*(filter.get()))[n]);
	p.x_[mappedD] = (*(filter.get()))[seq[n]]->x_[mappedD];
	const SprPoint* mappedP = &p;
	int icls = -1;
	if(      p.class_ == inputClasses[0] )
	  icls = 0;
	else if( p.class_ == inputClasses[1] )
	  icls = 1;
	else
	  continue;
	if( mapper.get() != 0 ) mappedP = mapper->output(&p);
	loss->update(icls,trained->response(mappedP),filter->w(n));
	if( mapper.get() != 0 ) mapper->clear();
      }

      // store loss
      aveLoss += loss->value();
    }// end loop over permutations

    // get and store average loss
    aveLoss = (aveLoss-nominalLoss)/nPerm;
    losses.push_back(pair<string,double>(trainedVars[d],aveLoss));
  }// end loop over variables
  assert( losses.size() == nVars );

  //
  // process computed loss
  //
  stable_sort(losses.begin(),losses.end(),SVICmpPairSDSecond());
  cout << "==========================================================" << endl;
  char t [200];
  sprintf(t,"%35s      %15s","Variable","Change in loss");
  cout << t << endl;
  for( int d=0;d<nVars;d++ ) {
    char s [200];
    sprintf(s,"%35s      %15.10f",losses[d].first.c_str(),losses[d].second);
    cout << s << endl;
  }
  cout << "==========================================================" << endl;

  // exit
  return 0;
}
