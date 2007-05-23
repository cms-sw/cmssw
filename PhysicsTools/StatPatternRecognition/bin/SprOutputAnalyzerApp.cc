//$Id: SprOutputAnalyzerApp.cc,v 1.6 2007/02/05 21:49:46 narsky Exp $
/*
  This executable for analysis of output ascii files produced by 
  a classifier. It lets the user quickly estimate fractions of 
  surviving background given the signal efficiency.

  The executable lets you choose between looking at fractional
  efficiency (default) and absolute event weights (-W option). If -W
  option is chosen, you can adjust the weights of signal and
  background events on the fly using -s and -b command-line
  options. These simply multiply the weights by the factor you
  provide.

  -c lets you look at a specific FOM as a function of the signal and
  background efficiencies. This option will only work if -W is
  specified.

  Here is, for example, what I have been doing for the Knunu analysis:

  SprOutputAnalyzerApp -y '.:1' -C bag -W -c 9 -s 0.00032 -b 2 save.out

  That is, I trained random forest and then used -o option of
  SprBaggerDecisionTreeApp to compute classifier output for the
  validation set. SprBaggerDecisionTreeApp executable names random
  forest 'bag' and saves the values in the column with a corresponding
  name. 0.00032 is the factor by which signal events in the validation
  sample are multiplied to scale them to the actual number expected in
  the Runs 1-4 data and 2 is the corresponding factor for the
  background. I want to monitor Punzi FOM. In the end, I get something
  like this:

Input signal weights for which background will be estimated [ 1.26 ] 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3 3.1 3.2 3.3 3.4
===========================================
Table of surviving background fractions (* shows minimal value in a row; FOM shown in parentheses)
===========================================
Signal Eff   \  Classifiers  |                                          bag |
-----------------------------------------------------------------------------
              0.9000         | *     7.84885 +-      1.90363 (     0.20923) |
              1.0000         | *     9.15750 +-      2.15844 (     0.22094) |
              1.1000         | *     9.97107 +-      2.22960 (     0.23617) |
              1.2000         | *    12.98021 +-      2.54563 (     0.23516) |
              1.3000         | *    14.69828 +-      2.77771 (     0.24373) |
              1.4000         | *    17.63934 +-      3.07061 (     0.24562) |
              1.5000         | *    18.86761 +-      3.14460 (     0.25669) |
              1.6000         | *    24.41848 +-      3.68122 (     0.24839) |
              1.7000         | *    26.54597 +-      3.87213 (     0.25555) |
              1.8000         | *    29.81607 +-      4.02040 (     0.25861) |
              1.9000         | *    32.26733 +-      4.13141 (     0.26461) |
              2.0000         | *    33.90502 +-      4.20540 (     0.27312) |
              2.1000         | *    38.56942 +-      4.54545 (     0.27236) |
              2.2000         | *    44.02982 +-      4.80405 (     0.27042) |
              2.3000         | *    49.83156 +-      5.08591 (     0.26872) |
              2.4000         | *    56.12838 +-      5.42614 (     0.26691) |
              2.5000         | *    63.30685 +-      5.75517 (     0.26437) |
              2.6000         | *    77.99720 +-      6.41133 (     0.25166) |
              2.7000         | *    87.12832 +-      6.76247 (     0.24921) |
              2.8000         | *    98.63033 +-      7.11803 (     0.24494) |
              2.9000         | *   115.13059 +-      7.65837 (     0.23712) |
              3.0000         | *   133.51367 +-      8.23281 (     0.22980) |

  The 1st column shows the expected signal contribution (normalized to
  the luminosity in the data using the scale factor I provided), the
  2nd column shows expected background contribution with errors. The
  number in parentheses are the cut on the classifier output and
  monitored FOM.
*/

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
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

#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <utility>
#include <set>
#include <string>
#include <memory>
#include <iomanip>
#include <algorithm>
#include <functional>
#include <cmath>

using namespace std;


struct SOAResponse {
  int cls;
  double weight;
  map<string,double> response;

  ~SOAResponse() {}

  SOAResponse(int c, double w)
    : cls(c), weight(w), response() {}

  SOAResponse(const SOAResponse& other)
    : cls(other.cls), weight(other.weight), response(other.response) {}

  void set(const char* classifier, double r) {
    map<string,double>::iterator found = response.find(classifier);
    if( found == response.end() )
      response.insert(pair<const string,double>(classifier,r));
    else
      found->second = r;
  }
};


struct SOAFigureOfMerit {
  double cut;
  double bgrndW;
  unsigned bgrndN;
  double fom;

  ~SOAFigureOfMerit() {}

  SOAFigureOfMerit()
    : cut(SprUtils::min()), bgrndW(0), bgrndN(0), fom(0) {}

  SOAFigureOfMerit(double c, double w, unsigned n)
    : cut(c), bgrndW(w), bgrndN(n), fom(0) {}

  SOAFigureOfMerit(double c, double w, unsigned n, double f)
    : cut(c), bgrndW(w), bgrndN(n), fom(f) {}

  SOAFigureOfMerit(const SOAFigureOfMerit& other)
    : cut(other.cut), 
      bgrndW(other.bgrndW), 
      bgrndN(other.bgrndN), 
      fom(other.fom) 
  {}
};


void help(const char* prog)
{
  cout << "Usage:  " << prog
       << " ascii_file_to_analyze" << endl;
  cout << "\t Options: " << endl;
  cout << "\t-h --- help                                        " << endl;
  cout << "\t-y list of input classes (see SprAbsFilter.hh)     " << endl;
  cout << "\t-C list of classifier names (in quotes with commas)" << endl;
  cout << "\t-v verbose level (0=silent default,1,2)            " << endl;
  cout << "\t-W display absolute weights (def=relative effic-cy)" << endl;
  cout << "\t-s scale all signal weights by this factor         " << endl;
  cout << "\t-b scale all background weights by this factor     " << endl;
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
  cout << "\t-n do not show computed cut and FOM for more compact output" 
       << endl;
}


// sorts by greater, not less!!!
struct SOACmpPairDDFirst
  : public binary_function<pair<double,double>,pair<double,double>,bool> {
  bool operator()(const pair<double,double>& l, const pair<double,double>& r)
    const {
    return (l.first > r.first);
  }
};


bool answerYN(const char* question)
{
  cout << question << " y/n [y] ";
  char yn [2];
  cin.getline(yn,2,'\n');
  if( yn[0]=='\0' || yn[0]=='y' ) return true;
  return false;
}


int main(int argc, char ** argv)
{
  // check command line
  if( argc < 2 ) {
    help(argv[0]);
    return 1;
  }

  // init
  bool useAbsolute = false;
  int verbose = 0;
  string inputClassesString;
  string inputClassifiersString;
  int iCrit = 5;
  double sW = 1.;
  double bW = 1.;
  bool showCutAndFOM = true;

  // decode command line
  int c;
  extern char* optarg;
  //  extern int optind;
  while( (c = getopt(argc,argv,"hy:C:v:Ws:b:c:n")) != EOF ) {
    switch( c )
      {
      case 'h' :
        help(argv[0]);
        return 1;
      case 'y' :
        inputClassesString = optarg;
        break;
      case 'C' :
        inputClassifiersString = optarg;
        break;
      case 'v' :
        verbose = (optarg==0 ? 0 : atoi(optarg));
        break;
      case 'W' :
	useAbsolute = true;
        break;
      case 's' :
	sW = (optarg==0 ? 1 : atof(optarg));
	break;
      case 'b' :
	bW = (optarg==0 ? 1 : atof(optarg));
	break;
      case 'c' :
        iCrit = (optarg==0 ? 5 : atoi(optarg));
        break;
      case 'n' :
	showCutAndFOM = false;
	break;
      }
  }

  // There has to be 1 argument after all options.
  string analyzeFile = argv[argc-1];
  if( analyzeFile.empty() ) {
    cerr << "No input file is specified." << endl;
    return 1;
  }

  // Prepare classifier list.
  vector<vector<string> > getClassifiers;
  SprStringParser::parseToStrings(inputClassifiersString.c_str(),
				  getClassifiers);
  if( getClassifiers.empty() ) {
    cerr << "Unable to parse input classifier names." << endl;
    return 1;
  }
  vector<string> classifiers = getClassifiers[0];

  // Prepare class list.
  pair<SprClass,SprClass> classes;
  if( !SprAbsFilter::decodeClassString(inputClassesString.c_str(),classes) ) {
    cerr << "Unable to decode classes from string " 
	 << inputClassesString << endl;
    return 1;
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
      crit.reset(new SprTwoClassPunzi(1.));
      cout << "Optimization criterion set to "
	   << "Punzi's sensitivity S/(0.5*nSigma+sqrt(B))" << endl;
      break;
    default :
      cerr << "Unable to make initialization criterion." << endl;
      return 1;
    }

  // open the analyzed file and read it
  ifstream file(analyzeFile.c_str());
  if( !file ) {
    cerr << "Unable to open file " << analyzeFile.c_str() << endl;
    return 2;
  }

  // read the header
  string line;
  getline(file,line,'\n');
  if( line.empty() ) {
    cerr << "Unable to read top line from file " 
	 << analyzeFile.c_str() << endl;
    return 3;
  }
  istringstream str(line);
  string dummy;
  unsigned nRead = 1;
  set<string> foundClassifiers;
  vector<pair<unsigned,string> > position;
  while( str >> dummy ) {
    // check correct format
    if( nRead==1 && dummy!="index" ) {
      cerr << "Incorrect top line format in position " << nRead << endl;
      return 3;
    }
    if( nRead==2 && dummy!="i" ) {
      cerr << "Incorrect top line format in position " << nRead << endl;
      return 3;
    }
    if( nRead==3 && dummy!="w" ) {
      cerr << "Incorrect top line format in position " << nRead << endl;
      return 3;
    }

    // find the input name among classifier list
    if( find(classifiers.begin(),classifiers.end(),dummy) 
	!= classifiers.end() ) {
      cout << "Output of classifier \"" << dummy.c_str() 
	   << "\" found in position " << nRead << endl;
      set<string>::const_iterator found = foundClassifiers.find(dummy);
      if( found != foundClassifiers.end() ) {
	cerr << "Classifier " << dummy.c_str() 
	     << " is already in the list. Skipping..." << endl;
      }
      else {
	position.push_back(pair<unsigned,string>(nRead,dummy));
      }
    }

    // increment record number
    ++nRead;
  }

  // sanity check
  if( position.empty() ) {
    cerr << "Specified classifiers not found in the input file." << endl;
    return 4;
  }

  // read lines and fill out responses
  double wsig(0), wbgr(0);
  unsigned nsig(0), nbgr(0);
  vector<SOAResponse> responses;
  while( getline(file,line) ) {
    int index(0), icls(0);
    double weight(0);
    istringstream event(line);
    event >> index >> icls >> weight;

    // decode input class
    int cls = 0;
    if(      icls == classes.first ) {
      cls = 0;
      nbgr++;
      wbgr += weight;
    }
    else if( icls == classes.second ) {
      cls = 1;
      nsig++;
      wsig += weight;
    }
    else
      continue;

    // fill out response
    SOAResponse response(cls,weight);
    unsigned istart = 4;
    double fread = 0;
    for( int i=0;i<position.size();i++ ) {
      unsigned nRead = position[i].first;
      string classifier = position[i].second;
      for( int ipos=istart;ipos<nRead;ipos++ ) event >> fread;// rewind forward
      event >> fread;
      response.set(classifier.c_str(),fread);
      istart = nRead + 1;
    }
    
    // record response
    responses.push_back(response);
  }

  // sanity check
  assert( wsig>0 || wbgr>0 );
  if( responses.empty() ) {
    cerr << "Did not find any stored classifier responses." << endl;
    return 5;
  }
  cout << "Read " << responses.size() 
       << " points from input file "<< analyzeFile.c_str() 
       << "   Bgrnd Weight=" << bW*wbgr
       << "   Signal Weight=" << sW*wsig << endl;

  // create vector of signal efficiencies
  vector<double> effS;
  if( useAbsolute ) {
    effS.push_back(1);
    effS.push_back(10);
    effS.push_back(100);
  }
  else {
    effS.push_back(0.1);
    effS.push_back(0.2);
    effS.push_back(0.3);
    effS.push_back(0.4);
    effS.push_back(0.5);
    effS.push_back(0.6);
    effS.push_back(0.7);
    effS.push_back(0.8);
    effS.push_back(0.9);
  }

  //
  // start loop over processing response values
  //
  while( true ) {

    if( wsig>0 && wbgr>0 ) {
      // read efficiency values
      if( useAbsolute ) {
	cout << "Input signal weights for which background "
	     << "will be estimated [ ";
      }
      else {
	cout << "Input signal efficiency values for which background "
	     << "will be estimated [ ";
      }
      for( int i=0;i<effS.size();i++ )
	cout << effS[i] << " ";
      cout << "] ";
      string line;
      getline(cin,line,'\n');
      if( !line.empty() ) {
	effS.clear();
	istringstream str(line);
	double dummy = 0;
	while( str >> dummy ) effS.push_back(dummy);
      }
      if( effS.empty() ) {
	cerr << "What, are you trying to be cute? Enter values." << endl;
	return 6;
      }
      stable_sort(effS.begin(),effS.end());
      
      //
      // Estimate background fractions for these signal efficiencies.
      //
      map<string,vector<SOAFigureOfMerit> > effB;
      
      for( int iclassifier=0;iclassifier<position.size();iclassifier++ ) {
	const string classifier = position[iclassifier].second;
	
	// prepare vectors
	vector<pair<double,double> > signal;
	vector<pair<double,double> > bgrnd;
	
	// fill them
	for( int i=0;i<responses.size();i++ ) {
	  if(      responses[i].cls == 0 ) {
	    bgrnd.push_back(pair<double,
			    double>(responses[i].response[classifier],
				    responses[i].weight));
	  }
	  else if( responses[i].cls == 1 ) {
	    signal.push_back(pair<double,
			     double>(responses[i].response[classifier],
				     responses[i].weight));
	  }
	}
	
	// sort
	stable_sort(bgrnd.begin(),bgrnd.end(),SOACmpPairDDFirst());
	stable_sort(signal.begin(),signal.end(),SOACmpPairDDFirst());
	
	// find dividing point in classifier response
	vector<double> cuts(effS.size());
	double w = 0;
	int divider = 0;
	int i = 0;
	while( i<signal.size() && divider<effS.size() ) {
	  w += signal[i].second;
	  if( (useAbsolute && sW*w>effS[divider]) 
	      || (!useAbsolute && (w/wsig)>effS[divider]) ) {
	    if( i == 0 )
	      cuts[divider] = signal[0].first;
	    else
	      cuts[divider] = 0.5 * (signal[i].first + signal[i-1].first);
	    divider++;
	  }
	  i++;
	}
	
	// find background fractions
	SOAFigureOfMerit defaultFOM(cuts[cuts.size()-1],
				 (useAbsolute ? wbgr : 1),nbgr);
	pair<map<string,vector<SOAFigureOfMerit> >::iterator,bool> 
	  inserted = effB.insert(pair<const string,
				 vector<SOAFigureOfMerit> >(classifier,
				       vector<SOAFigureOfMerit>(effS.size(),
								defaultFOM)));
	assert( inserted.second );
	w = 0;
	divider = 0;
	i = 0;
	while( i<bgrnd.size() && divider<effS.size() ) {
	  if( bgrnd[i].first < cuts[divider] ) {
	    if( useAbsolute ) {
	      inserted.first->second[divider] 
		= SOAFigureOfMerit(cuts[divider],
				   bW*w,i,( crit.get()==0 ? 0 
					    : crit->fom(bW*(wbgr-w),bW*w,
					effS[divider],sW*wsig-effS[divider])));
	    }
	    else {
	      inserted.first->second[divider] = SOAFigureOfMerit(cuts[divider],
								 w/wbgr,i);
	    }
	    divider++;
	  }
	  w += bgrnd[i].second;
	  i++;
	}
      }
      
      //
      // make a table of signal and background efficiencies
      //
      cout << "===========================================" << endl;
      cout << "Table of surviving background fractions"
	   << " (* shows minimal value in a row; " 
	   << "Cut on classifier output and FOM are shown in parentheses)" 
	   << endl;
      cout << "===========================================" << endl;
      char s[200];
      sprintf(s,"Signal Eff   \\  Classifiers  |");
      cout << s;
      string temp = "------------------------------";
      for( map<string,vector<SOAFigureOfMerit> >::const_iterator
	     iter=effB.begin();iter!=effB.end();iter++ ) {
	if( showCutAndFOM )
	  sprintf(s," %65s |",iter->first.c_str());
	else
	  sprintf(s," %29s |",iter->first.c_str());
	cout << s;
	if( showCutAndFOM )
	  temp += "--------------------------------------------------------------------";
	else
	  temp += "--------------------------------";
      }
      cout << endl;
      cout << temp.c_str() << endl;
      for( int i=0;i<effS.size();i++ ) {
	sprintf(s,"          %10.4f         |",effS[i]);
	cout << s;
	vector<string> names;
	vector<double> cuts;
	vector<double> values;
	vector<double> errors;
	vector<double> fom;
	for( map<string,vector<SOAFigureOfMerit> >::const_iterator
	       iter=effB.begin();iter!=effB.end();iter++ ) {
	  names.push_back(iter->first);
	  cuts.push_back(iter->second[i].cut);
	  double value = iter->second[i].bgrndW;
	  values.push_back(value);
	  unsigned nevts = iter->second[i].bgrndN;
	  errors.push_back(( nevts>0 ? value/sqrt(double(nevts)) : 0 ));
	  fom.push_back(iter->second[i].fom);
	}
	int foundMin = min_element(values.begin(),values.end()) - values.begin();
	for( int j=0;j<names.size();j++ ) {
	  if( showCutAndFOM ) {
	    if( j == foundMin )
	      sprintf(s," *%12.5f +- %12.5f (Cut=%12.5f FOM=%12.5f) |",
		      values[j],errors[j],cuts[j],fom[j]);
	    else
	      sprintf(s,"  %12.5f +- %12.5f (Cut=%12.5f FOM=%12.5f) |",
		      values[j],errors[j],cuts[j],fom[j]);
	  }
	  else {
	    if( j == foundMin )
	      sprintf(s," *%12.5f +- %12.5f |",values[j],errors[j]);
	    else
	      sprintf(s,"  %12.5f +- %12.5f |",values[j],errors[j]);
	  }
	  cout << s;
	}
	cout << endl;
      }
      cout << "===========================================" << endl;
    }// end of wsig>0 && wbgr>0
    else {
      cout << "One of categories has no entries. "
	   << "Cannot make a table of efficiencies." << endl;
    }

    //
    // make a histogram for the requested classifier
    //
    if( answerYN("Histogram classifier output?") ) {

      // user input
      cout << "Input classifier name, low and upper limits, and step: "
	   << "(Example: bag 0 1 0.1) ----> ";
      string line;
      getline(cin,line,'\n');
      if( line.empty() ) {
	cerr << "No values given. Exit to main loop." << endl;
	continue;
      }
      istringstream str(line);
      string classifier;
      double xlo(0), xhi(0), dx(0);
      str >> classifier >> xlo >> xhi >> dx;
      if( classifier.empty() || dx<=0. || xlo>=xhi ) {
	cerr << "Incorrect parameters given. Exit to main loop." << endl;
	continue;
      }

      // prepare vectors
      vector<pair<double,double> > signal;
      vector<pair<double,double> > bgrnd;

      // fill them
      for( int i=0;i<responses.size();i++ ) {
	if(      responses[i].cls == 0 ) {
	  bgrnd.push_back(pair<double,
			  double>(responses[i].response[classifier],
				  responses[i].weight));
	}
	else if( responses[i].cls == 1 ) {
	  signal.push_back(pair<double,
			   double>(responses[i].response[classifier],
				   responses[i].weight));
	}
      }

      // sort in ascending order
      stable_sort(bgrnd.begin(),bgrnd.end(),not2(SOACmpPairDDFirst()));
      stable_sort(signal.begin(),signal.end(),not2(SOACmpPairDDFirst()));

      // book histos
      unsigned nbin = unsigned(floor((xhi-xlo)/dx)) + 1;
      vector<pair<double,double> > shist(nbin), bhist(nbin);

      // fill out histos
      int jsig(0), jbgr(0);
      for( int i=0;i<nbin;i++ ) {
	double xa = xlo + i*dx;
	double xb = xa + dx;

	// check starting indices
	if( !signal.empty() ) {
	  if( signal[jsig].first < xa ) {
	    for( int j=jsig;j<signal.size();j++ ) {
	      if( signal[j].first >= xa ) {
		jsig = j;
		break;
	      }
	    }
	  }
	}
	if( !bgrnd.empty() ) {
	  if( bgrnd[jbgr].first < xa ) {
	    for( int j=jbgr;j<bgrnd.size();j++ ) {
	      if( bgrnd[j].first >= xa ) {
		jbgr = j;
		break;
	      }
	    }
	  }
	}

	// sum entries over this bin
	double wsig = 0;
	unsigned nsig = 0;
	for( int j=jsig;j<signal.size();j++ ) {
	  if( signal[j].first >= xb ) {
	    double err = ( nsig>0 ? sW*wsig/sqrt(double(nsig)) : 0 );
	    shist[i] = pair<double,double>(sW*wsig,err);
	    jsig = j;
	    break;
	  }
	  wsig += signal[j].second;
	  nsig++;
	}
	double wbgr = 0;
	unsigned nbgr = 0;
	for( int j=jbgr;j<bgrnd.size();j++ ) {
	  if( bgrnd[j].first >= xb ) {
	    double err = ( nbgr>0 ? bW*wbgr/sqrt(double(nbgr)) : 0 );
	    bhist[i] = pair<double,double>(bW*wbgr,err);
	    jbgr = j;
	    break;
	  }
	  wbgr += bgrnd[j].second;
	  nbgr++;
	}
      }// end bin loop

      // sanity check
      assert( shist.size() == bhist.size() );

      // print out
      cout << "===========================================" << endl;
      cout << "Histogram for output of classifier " 
	   << classifier.c_str() << endl;
      char s[200];
      cout << "Xlo=" << xlo << " Xhi=" << xhi 
	   << " dX=" << dx << " Nbin=" << nbin << endl;
      sprintf(s," %14s | %30s | %30s |","Bin center","Signal","Background");
      cout << s << endl;
      for( int i=0;i<nbin;i++ ) {
	double x = xlo + (i+0.5)*dx;
	sprintf(s," %12.5f   |   %12.5f +- %12.5f |   %12.5f +- %12.5f |",
		x,
		shist[i].first,shist[i].second,
		bhist[i].first,bhist[i].second);
	cout << s << endl;
      }
      cout << "===========================================" << endl;
    }// end histogram

    // exit?
    if( !answerYN("Continue?") ) break;
  }

  // exit
  return 0;
}
