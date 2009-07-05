//$Id: SprInteractiveAnalysisApp.cc,v 1.4 2007/11/12 06:19:11 narsky Exp $
/*
  This executable is intended for interactive analysis of small samples.
  The user can interactively add and remove various classifiers with
  various configurations. Should be self-explanatory.

  Warning: for now the code does not automatically delete cache for
  failed classifiers. For example, if one BDT instance fails, you will
  need to re-create all BDT instances. This will be fixed in a future
  release.
*/

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsWriter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBinarySplit.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprFisher.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedFisher.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprLogitR.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedLogitR.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStdBackprop.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedStdBackprop.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBagger.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprArcE4.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedBagger.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTopdownTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedTopdownTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassGiniIndex.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassIDFraction.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerBootstrap.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDataFeeder.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRWFactory.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsVarTransformer.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprVarTransformerReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformerFilter.hh"

#include <unistd.h>
#include <stdio.h>
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

using namespace std;


struct SIAResponse {
  int cls;
  double weight;
  double response;

  ~SIAResponse() {}

  SIAResponse(int c, double w, double r) 
    : cls(c), weight(w), response(r) {}

  SIAResponse(const SIAResponse& other)
    : cls(other.cls), weight(other.weight), response(other.response) {}
};

// sorts by greater, not less!!!
struct SIACmpPairDDFirst
  : public binary_function<pair<double,double>,pair<double,double>,bool> {
  bool operator()(const pair<double,double>& l, const pair<double,double>& r)
    const {
    return (l.first > r.first);
  }
};


void help(const char* prog) 
{
  cout << "Usage:  " << prog << " training_data_file " << endl;
  cout << "\t Options: " << endl;
  cout << "\t-h --- help                                        " << endl;
  cout << "\t-p path to temporary cache files (default=\"./\")  " << endl;
  cout << "\t-o output Tuple file                               " << endl;
  cout << "\t-a input ascii file mode (see SprSimpleReader.hh)  " << endl;
  cout << "\t-A save output data in ascii instead of Root       " << endl;
  cout << "\t-y list of input classes (see SprAbsFilter.hh)     " << endl;
  cout << "\t-Q apply variable transformation saved in file     " << endl;
  cout << "\t-K keep this fraction in training set and          " << endl;
  cout << "\t\t put the rest into validation set                " << endl;
  cout << "\t-D randomize training set split-up                 " << endl;
  cout << "\t-t read validation/test data from a file           " << endl;
  cout << "\t\t (must be in same format as input data!!!        " << endl;
  cout << "\t-v verbose level (0=silent default,1,2)            " << endl;
  cout << "\t-V include only these input variables              " << endl;
  cout << "\t-z exclude input variables from the list           " << endl;
  cout << "\t\t Variables must be listed in quotes and separated by commas." 
       << endl;
}


int prepareExit(const map<string,SprAbsClassifier*>& classifiers,
		const vector<const SprAbsClassifier*>& cToClean,
		const vector<const SprIntegerBootstrap*>& bootstraps,
		int status)
{
  for( map<string,SprAbsClassifier*>::const_iterator 
	 i=classifiers.begin();i!=classifiers.end();i++ )
    delete i->second;
  for( unsigned int i=0;i<cToClean.size();i++ )
    delete cToClean[i];
  for( unsigned int i=0;i<bootstraps.size();i++ )
    delete bootstraps[i];
  return status;
}


bool answerYN(const char* question)
{
  cout << question << " y/n [y] ";
  char yn [2];
  cin.getline(yn,2,'\n');
  if( yn[0]=='\0' || yn[0]=='y' ) return true;
  return false;
}


unsigned answerHowMany(unsigned defaultN, const char* question)
{
  cout << question << " [" << defaultN << "] ";
  string line;
  getline(cin,line,'\n');
  unsigned n = defaultN;
  if( !line.empty() )
    n = atoi(line.c_str());
  return n;
}


string answerName(const char* question)
{
  cout << question << " ";
  string line;
  getline(cin,line,'\n');
  if( !line.empty() ) {
    line.erase( line.find_last_not_of(' ') + 1 );
    line.erase( 0, line.find_first_not_of(' ') );
  }
  return line;
}


bool findCache(const string& prefix, const char* cacheName, ifstream& file)
{
  string fname = prefix;
  fname += cacheName;
  file.open(fname.c_str());
  if( !file ) return false;
  return true;
}


bool checkOldCache(const string& prefix)
{
  string cmd = "ls -a ";
  cmd += prefix;
  cmd += "*";
  if( system(cmd.c_str()) == 0 )
    return true;
  return false;
}


bool eraseOldCache(const string& prefix)
{
  string cmd = "rm -f ";
  cmd += prefix;
  cmd += "*";
  if( system(cmd.c_str()) == 0 )
    return true;
  return false;
}


bool moveNewOntoOld(const string& prefix, const char* name)
{
  // get file name
  string oldfile = prefix;
  oldfile += name;
  string newfile = oldfile;
  newfile += ".new";
  
  // move new cache to old cache
  struct stat buf;
  if( stat(newfile.c_str(),&buf) == 0 ) {
    string cmd = "mv -f";
    cmd += " "; cmd += newfile;
    cmd += " "; cmd += oldfile;
    if( system(cmd.c_str()) != 0 ) {
      cerr << "Attempt to move new cache " << newfile.c_str() 
	   << " onto old cache " << oldfile.c_str()
	   << " terminated with error " << errno << endl;
      return false;
    }
    cout << "Moved " << newfile.c_str() << " =====> " 
	 << oldfile.c_str() << endl;
  }

  // exit
  return true;
}


bool makeNewCache(const string& prefix, const char* cacheName, ofstream& file)
{
  // get file name
  string fname = prefix;
  fname += cacheName;
  fname += ".new";

  // check file existence
  struct stat buf;
  if( stat(fname.c_str(),&buf) == 0 ) {
    cerr << "Warning: file " << fname.c_str() << " will be deleted." << endl;
    string cmd = "rm -f ";
    cmd += fname.c_str();
    if( system(cmd.c_str()) != 0 ) {
      cerr << "Attempt to delete file " << fname.c_str() 
           << " terminated with error " << errno << endl;
      return false;
    }
  }
 
  // open a new file
  file.open(fname.c_str());
  if( !file ) {
    cerr << "Unable to open file " << fname.c_str() << endl;
    return false;
  }

  // exit
  return true;
}


bool storeNewCache(const string& prefix, const char* cacheName,
		   const vector<SIAResponse>& v)
{
  // get file name
  string fname = prefix;
  fname += cacheName;
  fname += ".new";

  // open the file
  ofstream file(fname.c_str(),ios_base::app);
  if( !file ) {
    cerr << "Unable to open file " << fname.c_str() << endl;
    return false;
  }

  // store values
  int j = 0;
  for( unsigned int i=0;i<v.size();i++ ) {
    file << v[i].cls << " " << v[i].weight << " " << v[i].response << " #   ";
    if( ++j == 10 ) {
      j = 0;
      file << endl;
    }
  }

  // exit
  return true;
}


bool storeEffCache(const string& prefix, const char* cacheName,
		   const vector<double>& v)
{
  // get file name
  string fname = prefix;
  fname += cacheName;

  // check file existence
  struct stat buf;
  if( stat(fname.c_str(),&buf) == 0 ) {
    cerr << "Warning: file " << fname.c_str() << " will be deleted." << endl;
    string cmd = "rm -f ";
    cmd += fname.c_str();
    if( system(cmd.c_str()) != 0 ) {
      cerr << "Attempt to delete file " << fname.c_str() 
           << " terminated with error " << errno << endl;
      return false;
    }
  }
 
  // open a new file
  ofstream file(fname.c_str());
  if( !file ) {
    cerr << "Unable to open file " << fname.c_str() << endl;
    return false;
  }

  // store values
  for( unsigned int i=0;i<v.size();i++ )
    file << v[i] << " ";
  file << endl;

  // exit
  return true;
}


unsigned readCache(ifstream& file, vector<SIAResponse>& v)
{
  v.clear();
  int cls(0); double weight(0), resp(0);
  string line;
  char dummy;
  while( getline(file,line) ) {
    istringstream str(line);
    unsigned n = 0;
    for( unsigned int i=0;i<line.size();i++ )
      if( line[i]=='#' ) n++;
    for( unsigned int i=0;i<n;i++ ) {
      str >> cls >> weight >> resp >> dummy;
      v.push_back(SIAResponse(cls,weight,resp));
    }
  }
  return v.size();
}


bool resetValidation(const char* cacheName, 
		     map<string,vector<SIAResponse> >& validated, 
		     const SprAbsFilter* filter)
{
  // find element
  map<string,vector<SIAResponse> >::iterator iter = validated.find(cacheName);

  // if it exists
  if( iter == validated.end() ) {
    pair<map<string,vector<SIAResponse> >::iterator,bool> inserted =
      validated.insert(pair<const string,vector<SIAResponse> >(cacheName,
						      vector<SIAResponse>()));
    assert( inserted.second );
    return true;
  }

  // if it exists with the wrong size
  if( filter->size() != iter->second.size() ) {
    iter->second.clear();
    return true;
  }

  // exit
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
  string tupleFile;
  string pathToCache;
  int readMode = 0;
  SprRWFactory::DataType writeMode = SprRWFactory::Root;
  int verbose = 0;
  string includeList, excludeList;
  string inputClassesString;
  string valFile;
  bool split = false;
  double splitFactor = 0;
  bool splitRandomize = false; 
  string transformerFile;

  // decode command line
  int c;
  extern char* optarg;
  //  extern int optind;
  while( (c = getopt(argc,argv,"hp:o:a:Ay:Q:K:Dt:v:V:z:")) 
	 != EOF ) {
    switch( c )
      {
      case 'h' :
	help(argv[0]);
	return 1;
      case 'p' :
	pathToCache = optarg;
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
      case 'y' :
	inputClassesString = optarg;
	break;
      case 'Q' :
        transformerFile = optarg;
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
      case 'v' :
	verbose = (optarg==0 ? 0 : atoi(optarg));
	break;
      case 'V' :
	includeList = optarg;
	break;
      case 'z' :
	excludeList = optarg;
	break;
      }
  }

  // There have to be 1 argument after all options.
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

  // read validation data from file
  auto_ptr<SprAbsFilter> valFilter;
  if( !split && valFile.empty() ) {
    cout << "No test data specified. Will use training data." << endl;
    vector<double> weights;
    filter->weights(weights);
    bool ownData = false;
    valFilter.reset(new SprEmptyFilter(filter->data(),weights,ownData));
  }
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

  // determine path to cache
  if( !pathToCache.empty() ) {
    pathToCache.erase( pathToCache.find_last_not_of(' ') + 1 );
    pathToCache.erase( 0, pathToCache.find_first_not_of(' ') );
  }
  if( !pathToCache.empty() && *(pathToCache.rbegin())!='/' )
    pathToCache += "/";
  cout << "Will use directory \"" << pathToCache.c_str() 
       << "\" for cache." << endl;
  string prefix = pathToCache;
  prefix += ".cache_";
  
  // check for presence of old cache
  if( checkOldCache(prefix) ) {
    cout << "Warning!!! Some old cache files .cache* are found." << endl;
    if( answerYN("Erase old cache?") ) {
      if( !eraseOldCache(prefix) ) return 2;
    }
  }

  // make criteria
  SprTwoClassIDFraction idfrac;
  SprTwoClassGiniIndex gini;

  //
  // put everything in a big loop
  //
  bool go = true;
  while( go ) {

    //
    // read classifier configurations
    //
    map<string,SprAbsClassifier*> classifiers;
    map<string,vector<SIAResponse> > validated;
    map<string,string> message;
    vector<const SprAbsClassifier*> cToClean;
    vector<const SprIntegerBootstrap*> bootstraps;
    
    //
    // LDA
    //
    if( answerYN("Use Linear Discriminant Analysis?") ) {
      classifiers.insert(pair<const string,
			 SprAbsClassifier*>("LDA",
					    new SprFisher(filter.get(),1)));
      ifstream input;
      if( findCache(prefix,"LDA",input) ) {
	if( resetValidation("LDA",validated,valFilter.get()) ) {
	  if( readCache(input,validated.find("LDA")->second) 
	      != valFilter->size() ) {
	    cerr << "Cannot read cached LDA values." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	}
      }
      ostringstream ost;
      ost << "LDA = Linear Discriminant Analysis";
      message.insert(pair<const string,string>("LDA",ost.str()));
    }
    
    //
    // QDA
    //
    if( answerYN("Use Quadratic Discriminant Analysis?") ) {
      classifiers.insert(pair<const string,
			 SprAbsClassifier*>("QDA",
					    new SprFisher(filter.get(),2)));
      ifstream input;
      if( findCache(prefix,"QDA",input) ) {
	if( resetValidation("QDA",validated,valFilter.get()) ) {
	  if( readCache(input,validated.find("QDA")->second) 
	      != valFilter->size() ) {
	    cerr << "Cannot read cached QDA values." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	}
      }
      ostringstream ost;
      ost << "QDA = Quadratic Discriminant Analysis";
      message.insert(pair<const string,string>("QDA",ost.str()));
    }

    //
    // LogitR
    //
    if( answerYN("Use Logistic Regression?") ) {
      double eps = 0.001;
      double updateFactor = 1;
      int initToZero = 1;
      bool repeat = false;
      ifstream input;
      string line;
      if( findCache(prefix,"LR",input) ) {
	repeat = true;
	if( !getline(input,line,'\n') ) {
	  cerr << "Cannot read top line from LR cache." << endl;
	  return prepareExit(classifiers,cToClean,bootstraps,3);
	}
	istringstream str(line);
	str >> eps >> updateFactor >> initToZero;
      }
      cout << "Input accuracy, update factor, and init flag "
	   << "(0 if initialize from 0; 1 if initialize from LDA) ["
	   << eps << " " << updateFactor << " " << initToZero << "] ";
      getline(cin,line,'\n');
      if( !line.empty() ) {
	repeat = false;
	istringstream str(line);
	str >> eps >> updateFactor >> initToZero;
      }
      if( repeat ) {
	if( resetValidation("LR",validated,valFilter.get()) ) {
	  if( readCache(input,validated.find("LR")->second) 
	      != valFilter->size() ) {
	    cerr << "Cannot read cached LR values." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	}
      }
      else {
	// save LR parameters to file
	ofstream output;
	if( makeNewCache(prefix,"LR",output) ) {
	  output << eps << " " << updateFactor << " " << initToZero << endl;
	}
	else {
	  cerr << "Unable to make output file for LR." << endl;
	  return prepareExit(classifiers,cToClean,bootstraps,3);
	}
	
	// make LR
	double beta0 = 0;
	SprVector beta;
	if( initToZero == 0 ) {
	  SprVector dummy(filter->dim());
	  beta = dummy;
	  for( unsigned int i=0;i<filter->dim();i++ ) beta[i] = 0;
	}
	SprLogitR* logit = new SprLogitR(filter.get(),beta0,beta,
					 eps,updateFactor);
	classifiers.insert(pair<const string,
			   SprAbsClassifier*>("LR",logit));
      }
      ostringstream ost;
      ost << "LR = Logistic Regression with:"
	  << " Eps=" << eps
	  << " updateFactor=" << updateFactor 
	  << " initFlag=" << initToZero;
      message.insert(pair<const string,string>("LR",ost.str()));
    }

    //
    // single decision tree
    //
    if( answerYN("Use single decision tree?") ) {
      
      // get instances
      unsigned ninstance = 0;
      ifstream iinst;
      if( findCache(prefix,"DT_instances",iinst) ) {
	string line;
	if( !getline(iinst,line,'\n') ) {
	  cerr << "Cannot read top line from DT cache." << endl;
	  return prepareExit(classifiers,cToClean,bootstraps,3);
	}
	istringstream str(line);
	str >> ninstance;
      }
      ninstance = 
	answerHowMany(ninstance,
	      "How many instances of the classifier would you like to run?");
      ofstream oinst;
      if( makeNewCache(prefix,"DT_instances",oinst) ) {
	oinst << ninstance << endl;
      }
      else {
	cerr << "Unable to make output file for DT." << endl;
	return prepareExit(classifiers,cToClean,bootstraps,3);
      }
      if( !moveNewOntoOld(prefix,"DT_instances") )
	return prepareExit(classifiers,cToClean,bootstraps,3);
      
      // loop over instances
      for( unsigned int instance=0;instance<ninstance;instance++ ) {
	string name = "DT_";
	char s [200];
	sprintf(s,"%i",instance);
	name += s;
	unsigned nleaf = 0;
	bool repeat = false;
	ifstream input;
	string line;
	if( findCache(prefix,name.c_str(),input) ) {
	  repeat = true;
	  if( !getline(input,line,'\n') ) {
	    cerr << "Cannot read top line from DT cache." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	  istringstream str(line);
	  str >> nleaf;
	}
	cout << "Input minimal tree leaf size "
	     << "for DT instance " << instance << " [" << nleaf << "] ";
	getline(cin,line,'\n');
	if( !line.empty() ) {
	  repeat = false;
	  istringstream str(line);
	  str >> nleaf;
	}
	if( repeat ) {
	  if( resetValidation(name.c_str(),validated,valFilter.get()) ) {
	    if( readCache(input,validated.find(name)->second) 
		!= valFilter->size() ) {
	      cerr << "Cannot read cached DT values." << endl;
	      return prepareExit(classifiers,cToClean,bootstraps,3);
	    }
	  }
	}
	else {
	  // save params to file
	  ofstream output;
	  if( makeNewCache(prefix,name.c_str(),output) ) {
	    output << nleaf << endl;
	  }
	  else {
	    cerr << "Unable to make output file for DT." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	  
	  // make decision tree
	  bool discrete = false;
	  SprTopdownTree* tree = new SprTopdownTree(filter.get(),&gini,
						    nleaf,discrete);
	  classifiers.insert(pair<const string,SprAbsClassifier*>(name,
								  tree));
	}
	ostringstream ost;
	ost << name.c_str() 
	    << " = Decision Tree with:"
	    << " FOM=Gini"
	    << " nEventsPerLeaf=" << nleaf;
	message.insert(pair<const string,string>(name,ost.str()));
      }
    }

    //
    // single neural net
    //
    if( answerYN("Use neural net?") ) {

      // get instances
      unsigned ninstance = 0;
      ifstream iinst;
      if( findCache(prefix,"STDNN_instances",iinst) ) {
	string line;
	if( !getline(iinst,line,'\n') ) {
	  cerr << "Cannot read top line from STDNN cache." << endl;
	  return prepareExit(classifiers,cToClean,bootstraps,3);
	}
	istringstream str(line);
	str >> ninstance;
      }
      ninstance = 
	answerHowMany(ninstance,
	     "How many instances of the classifier would you like to run?");
      ofstream oinst;
      if( makeNewCache(prefix,"STDNN_instances",oinst) ) {
	oinst << ninstance << endl;
      }
      else {
	cerr << "Unable to make output file for STDNN." << endl;
	return prepareExit(classifiers,cToClean,bootstraps,3);
      }
      if( !moveNewOntoOld(prefix,"STDNN_instances") )
	return prepareExit(classifiers,cToClean,bootstraps,3);

      // loop over instances
      for( unsigned int instance=0;instance<ninstance;instance++ ) {
	string name = "STDNN_";
	char s [200];
	sprintf(s,"%i",instance);
	name += s;
	string structure = "I:H:H:O";
	unsigned ncycles(0), initPoints(0);
	double eta(0.1), initEta(0.1);
	bool repeat = false;
	ifstream input;
	string line;
	if( findCache(prefix,name.c_str(),input) ) {
	  repeat = true;
	  if( !getline(input,line,'\n') ) {
	    cerr << "Cannot read top line from STDNN cache." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	  istringstream str(line);
	  str >> structure >> ncycles >> eta >> initPoints >> initEta;
	}
	cout << "Input StdBackprop NN structure (see README), "
	     << "number of training cycles, "
	     << "learning rate, "
	     << "number of points for initialization (0 if use all), "
	     << "and learning rate for initialization "
	     << "for STDNN instance " << instance 
	     << " [" << structure.c_str() << " " << ncycles
	     << " " << eta << " " << initPoints << " " << initEta << "] ";
	getline(cin,line,'\n');
	if( !line.empty() ) {
	  repeat = false;
	  istringstream str(line);
	  str >> structure >> ncycles >> eta >> initPoints >> initEta;
	}
	if( repeat ) {
	  if( resetValidation(name.c_str(),validated,valFilter.get()) ) {
	    if( readCache(input,validated.find(name)->second) 
		!= valFilter->size() ) {
	      cerr << "Cannot read cached STDNN values." << endl;
	      return prepareExit(classifiers,cToClean,bootstraps,3);
	    }
	  }
	}
	else {
	  // save params to file
	  ofstream output;
	  if( makeNewCache(prefix,name.c_str(),output) ) {
	    output << structure.c_str() << " " << ncycles << " " 
		   << eta << " " << initPoints << " " << initEta << endl;
	  }
	  else {
	    cerr << "Unable to make output file for STDNN." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }

	  // make NN
	  SprStdBackprop* stdnn = 
	    new SprStdBackprop(filter.get(),structure.c_str(),ncycles,eta);
	  stdnn->init(initEta,initPoints);
	  classifiers.insert(pair<const string,SprAbsClassifier*>(name,stdnn));
	}
	
	// prepare message
	ostringstream ost;
	ost << name.c_str() 
	    << " = StdBackprop Neural Net with:"
	    << " Structure=" << structure.c_str()
	    << " nCycles=" << ncycles
	    << " LearnRate=" << eta
	    << " nInitPoints=" << initPoints
	    << " InitLearnRate=" << initEta;
	message.insert(pair<const string,string>(name,ost.str()));
      }
    }

    //
    // boosted neural nets
    //
    if( answerYN("Use boosted neural nets?") ) {
      
      // get instances
      unsigned ninstance = 0;
      ifstream iinst;
      if( findCache(prefix,"BNN_instances",iinst) ) {
	string line;
	if( !getline(iinst,line,'\n') ) {
	  cerr << "Cannot read top line from BNN cache." << endl;
	  return prepareExit(classifiers,cToClean,bootstraps,3);
	}
	istringstream str(line);
	str >> ninstance;
      }
      ninstance = 
	answerHowMany(ninstance,
	    "How many instances of the classifier would you like to run?");
      ofstream oinst;
      if( makeNewCache(prefix,"BNN_instances",oinst) ) {
	oinst << ninstance << endl;
      }
      else {
	cerr << "Unable to make output file for BNN." << endl;
	return prepareExit(classifiers,cToClean,bootstraps,3);
      }
      if( !moveNewOntoOld(prefix,"BNN_instances") )
	return prepareExit(classifiers,cToClean,bootstraps,3);

      // loop over instances
      for( unsigned int instance=0;instance<ninstance;instance++ ) {
	string name = "BNN_";
	char s [200];
	sprintf(s,"%i",instance);
	name += s;
	int abMode = 0;
	double epsilon = 0;
	string structure = "I:H:H:O";
	unsigned adaCycles(0), nnCycles(0), initPoints(0);
	double eta(0.1), initEta(0.1);
	bool repeat = false;
	ifstream input;
	string line;
	if( findCache(prefix,name.c_str(),input) ) {
	  repeat = true;
	  if( !getline(input,line,'\n') ) {
	    cerr << "Cannot read top line from BNN cache." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	  istringstream str(line);
	  str >> abMode >> epsilon >> adaCycles 
	      >> structure >> nnCycles >> eta >> initPoints >> initEta;
	}
	cout << "Input AdaBoost mode (Discrete=1 Real=2 Epsilon=3), "
	     << "epsilon (only has effect for Epsilon and Real boost), "
	     << "number of neural nets, "
	     << "neural net structure, number of NN training cycles, "
	     << "learning rate, " 
	     << "number of points for NN initialization (0 if use all), "
	     << "and learning rate for NN initialization "
	     << "for BNN instance " << instance 
	     << " [" << abMode << " " << epsilon << " " 
	     << adaCycles << " " 
	     << structure.c_str() << " " << nnCycles << " "
	     << eta << " " << initPoints << " " << initEta << "] ";
	getline(cin,line,'\n');
	if( !line.empty() ) {
	  repeat = false;
	  istringstream str(line);
	  str >> abMode >> epsilon >> adaCycles 
	      >> structure >> nnCycles >> eta >> initPoints >> initEta;
	}
	if( repeat ) {
	  if( resetValidation(name.c_str(),validated,valFilter.get()) ) {
	    if( readCache(input,validated.find(name)->second) 
		!= valFilter->size() ) {
	      cerr << "Cannot read cached BNN values." << endl;
	      return prepareExit(classifiers,cToClean,bootstraps,3);
	    }
	  }
	}
	else {
	  // save params to file
	  ofstream output;
	  if( makeNewCache(prefix,name.c_str(),output) ) {
	    output << " " << abMode << " " << epsilon << " " << adaCycles 
		   << " " << structure << " " << nnCycles << " " << eta 
		   << " " << initPoints << " " << initEta << endl;
	  }
	  else {
	    cerr << "Unable to make output file for BNN." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }

	  // set AdaBoost mode
	  SprTrainedAdaBoost::AdaBoostMode mode = SprTrainedAdaBoost::Discrete;
	  switch( abMode )
	    {
	    case 1 :
	      mode = SprTrainedAdaBoost::Discrete;
	      break;
	    case 2 :
	      mode = SprTrainedAdaBoost::Real;
	      break;
	    case 3 :
	      mode = SprTrainedAdaBoost::Epsilon;
	      break;
	    default :
	      cerr << "Unknown mode for AdaBoost." << endl;
	      return prepareExit(classifiers,cToClean,bootstraps,3);
	    }
	  
	  // make neural net
	  SprStdBackprop* stdnn = 
	    new SprStdBackprop(filter.get(),structure.c_str(),nnCycles,eta);
	  cToClean.push_back(stdnn);
	  if( !stdnn->init(initEta,initPoints) ) {
	    cerr << "Unable to initialize neural net." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	  
	  // make AdaBoost
	  bool useStandardAB = true;
	  bool bagInput = false;
	  SprAdaBoost* ab = new SprAdaBoost(filter.get(),adaCycles,
					    useStandardAB,mode,bagInput);
	  ab->setEpsilon(epsilon);
	  classifiers.insert(pair<const string,SprAbsClassifier*>(name,ab));
	  if( !ab->addTrainable(stdnn) ) {
	    cerr << "Unable to add a neural net to AdaBoost." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	}
	
	// prepare message
	ostringstream ost;
	string sMode;
	switch( abMode )
	  {
	  case 1 :
	    sMode = "Discrete";
	    break;
	  case 2 :
	    sMode = "Real";
	    break;
	  case 3 :
	    sMode = "Epsilon";
	    break;
	  default :
	    cerr << "Unknown mode for AdaBoost." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	ost << name.c_str() 
	    << " = Boosted Neural Nets with:"
	    << " Mode=" << sMode.c_str()
	    << " Epsilon=" << epsilon
	    << " nNets=" << adaCycles 
	    << " structure=" << structure.c_str()
	    << " nCyclesPerNet=" << nnCycles
	    << " LearnRate=" << eta
	    << " nInitPoints=" << initPoints
	    << " InitLearnRate=" << initEta;
	message.insert(pair<const string,string>(name,ost.str()));
      }
    }

    //    
    // boosted splits
    //
    if( answerYN("Use boosted splits?") ) {
      unsigned ncycles = 0;
      bool repeat = false;
      ifstream input;
      string line;
      if( findCache(prefix,"BDS",input) ) {
	repeat = true;
	if( !getline(input,line,'\n') ) {
	  cerr << "Cannot read top line from BDS cache." << endl;
	  return prepareExit(classifiers,cToClean,bootstraps,3);
	}
	istringstream str(line);
	str >> ncycles;
      }
      cout << "Input number of AdaBoost splits per dimension [" 
	   << ncycles << "] ";
      getline(cin,line,'\n');
      if( !line.empty() ) {
	repeat = false;
	istringstream str(line);
	str >> ncycles;
      }
      if( repeat ) {
	if( resetValidation("BDS",validated,valFilter.get()) ) {
	  if( readCache(input,validated.find("BDS")->second) 
	      != valFilter->size() ) {
	    cerr << "Cannot read cached BDS values." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	}
      }
      else {
	// save ncycles to file
	ofstream output;
	if( makeNewCache(prefix,"BDS",output) ) {
	  output << ncycles << endl;
	}
	else {
	  cerr << "Unable to make output file for BDS." << endl;
	  return prepareExit(classifiers,cToClean,bootstraps,3);
	}
	
	// make AdaBoost
	bool useStandardAB = true;
	ncycles *= filter->dim();
	SprAdaBoost* ab = new SprAdaBoost(filter.get(),
					  ncycles,
					  useStandardAB,
					  SprTrainedAdaBoost::Discrete);
	classifiers.insert(pair<const string,SprAbsClassifier*>("BDS",ab));
	for( unsigned int i=0;i<filter->dim();i++ ) {
	  SprBinarySplit* s = new SprBinarySplit(filter.get(),&idfrac,i);
	  cToClean.push_back(s);
	  if( !ab->addTrainable(s,SprUtils::lowerBound(0.5)) ) {
	    cerr << "Unable to add binary split to AdaBoost." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	}
      }
      ostringstream ost;
      ost << "BDS = Boosted Decision Splits with: nSplits=" 
	  << ncycles;
      message.insert(pair<const string,string>("BDS",ost.str()));
    }
    
    //
    // boosted trees
    //
    if( answerYN("Use boosted trees?") ) {
      
      // get instances
      unsigned ninstance = 0;
      ifstream iinst;
      if( findCache(prefix,"BDT_instances",iinst) ) {
	string line;
	if( !getline(iinst,line,'\n') ) {
	  cerr << "Cannot read top line from BDT cache." << endl;
	  return prepareExit(classifiers,cToClean,bootstraps,3);
	}
	istringstream str(line);
	str >> ninstance;
      }
      ninstance = 
	answerHowMany(ninstance,
	    "How many instances of the classifier would you like to run?");
      ofstream oinst;
      if( makeNewCache(prefix,"BDT_instances",oinst) ) {
	oinst << ninstance << endl;
      }
      else {
	cerr << "Unable to make output file for BDT." << endl;
	return prepareExit(classifiers,cToClean,bootstraps,3);
      }
      if( !moveNewOntoOld(prefix,"BDT_instances") )
	return prepareExit(classifiers,cToClean,bootstraps,3);

      // loop over instances
      for( unsigned int instance=0;instance<ninstance;instance++ ) {
	string name = "BDT_";
	char s [200];
	sprintf(s,"%i",instance);
	name += s;
	int abMode = 0;
	double epsilon = 0;
	int iBagInput = 0;
	unsigned ncycles(0), nleaf(0);
	bool repeat = false;
	ifstream input;
	string line;
	if( findCache(prefix,name.c_str(),input) ) {
	  repeat = true;
	  if( !getline(input,line,'\n') ) {
	    cerr << "Cannot read top line from BDT cache." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	  istringstream str(line);
	  str >> abMode >> epsilon >> iBagInput >> ncycles >> nleaf;
	}
	cout << "Input AdaBoost mode (Discrete=1 Real=2 Epsilon=3), "
	     << "epsilon (only has effect for Epsilon and Real boost), "
	     << "bag training events flag (No=0 Yes=1), "
	     << "number of trees, and minimal tree leaf size "
	     << "for BDT instance " << instance 
	     << " [" << abMode << " " << epsilon << " " << iBagInput << " " 
	     << ncycles << " " << nleaf << "] ";
	getline(cin,line,'\n');
	if( !line.empty() ) {
	  repeat = false;
	  istringstream str(line);
	  str >> abMode >> epsilon >> iBagInput >> ncycles >> nleaf;
	}
	if( repeat ) {
	  if( resetValidation(name.c_str(),validated,valFilter.get()) ) {
	    if( readCache(input,validated.find(name)->second) 
		!= valFilter->size() ) {
	      cerr << "Cannot read cached BDT values." << endl;
	      return prepareExit(classifiers,cToClean,bootstraps,3);
	    }
	  }
	}
	else {
	  // save params to file
	  ofstream output;
	  if( makeNewCache(prefix,name.c_str(),output) ) {
	    output << abMode << " " << epsilon << " " << iBagInput << " "
		   << ncycles << " " << nleaf << endl;
	  }
	  else {
	    cerr << "Unable to make output file for BDT." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }

	  // set AdaBoost mode
	  SprTrainedAdaBoost::AdaBoostMode mode = SprTrainedAdaBoost::Discrete;
	  switch( abMode )
	    {
	    case 1 :
	      mode = SprTrainedAdaBoost::Discrete;
	      break;
	    case 2 :
	      mode = SprTrainedAdaBoost::Real;
	      break;
	    case 3 :
	      mode = SprTrainedAdaBoost::Epsilon;
	      break;
	    default :
	      cerr << "Unknown mode for AdaBoost." << endl;
	      return prepareExit(classifiers,cToClean,bootstraps,3);
	    }
	  
	  // make decision tree
	  bool discrete = (mode!=SprTrainedAdaBoost::Real);
	  SprTopdownTree* tree = new SprTopdownTree(filter.get(),&gini,
						    nleaf,discrete,0);
	  if( mode == SprTrainedAdaBoost::Real ) tree->forceMixedNodes();
	  tree->useFastSort();
	  cToClean.push_back(tree);
	  
	  // make AdaBoost
	  bool useStandardAB = true;
	  bool bagInput = (iBagInput==1);
	  SprAdaBoost* ab = new SprAdaBoost(filter.get(),ncycles,
					    useStandardAB,mode,bagInput);
	  ab->setEpsilon(epsilon);
	  classifiers.insert(pair<const string,SprAbsClassifier*>(name,ab));
	  if( !ab->addTrainable(tree,SprUtils::lowerBound(0.5)) ) {
	    cerr << "Unable to add a decision tree to AdaBoost." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	}
	
	// prepare message
	ostringstream ost;
	string sMode;
	switch( abMode )
	  {
	  case 1 :
	    sMode = "Discrete";
	    break;
	  case 2 :
	    sMode = "Real";
	    break;
	  case 3 :
	    sMode = "Epsilon";
	    break;
	  default :
	    cerr << "Unknown mode for AdaBoost." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	string sBag;
	if( iBagInput == 1 )
	  sBag = "Yes";
	else
	  sBag = "No";
	ost << name.c_str() 
	    << " = Boosted Decision Trees with:"
	    << " Mode=" << sMode.c_str()
	    << " Epsilon=" << epsilon
	    << " BagTrainingEvents=" << sBag.c_str()
	    << " nTrees=" << ncycles 
	    << " nEventsPerLeaf=" << nleaf;
	message.insert(pair<const string,string>(name,ost.str()));
      }
    }
    
    //
    // random forest
    //
    if( answerYN("Use random forest?") ) {
      
      // get instances
      unsigned ninstance = 0;
      ifstream iinst;
      if( findCache(prefix,"RF_instances",iinst) ) {
	string line;
	if( !getline(iinst,line,'\n') ) {
	  cerr << "Cannot read top line from RF cache." << endl;
	  return prepareExit(classifiers,cToClean,bootstraps,3);
	}
	istringstream str(line);
	str >> ninstance;
      }
      ninstance = 
	answerHowMany(ninstance,
	     "How many instances of the classifier would you like to run?");
      ofstream oinst;
      if( makeNewCache(prefix,"RF_instances",oinst) ) {
	oinst << ninstance << endl;
      }
      else {
	cerr << "Unable to make output file for RF." << endl;
	return prepareExit(classifiers,cToClean,bootstraps,3);
      }
      if( !moveNewOntoOld(prefix,"RF_instances") )
	return prepareExit(classifiers,cToClean,bootstraps,3);
      
      // loop over instances
      for( unsigned int instance=0;instance<ninstance;instance++ ) {
	string name = "RF_";
	char s [200];
	sprintf(s,"%i",instance);
	name += s;
	unsigned ncycles(0), nleaf(0), nfeatures(filter->dim());
	bool repeat = false;
	ifstream input;
	string line;
	if( findCache(prefix,name.c_str(),input) ) {
	  repeat = true;
	  if( !getline(input,line,'\n') ) {
	    cerr << "Cannot read top line from RF cache." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	  istringstream str(line);
	  str >> ncycles >> nleaf >> nfeatures;
	}
	cout << "Input number of trees, minimal tree leaf size, "
	     << "and number of input features for sampling "
	     << " for RF instance " << instance << " [" 
	     << ncycles << " " << nleaf << " " << nfeatures << "] ";
	getline(cin,line,'\n');
	if( !line.empty() ) {
	  repeat = false;
	  istringstream str(line);
	  str >> ncycles >> nleaf >> nfeatures;
	}
	if( repeat ) {
	  if( resetValidation(name.c_str(),validated,valFilter.get()) ) {
	    if( readCache(input,validated.find(name)->second) 
		!= valFilter->size() ) {
	      cerr << "Cannot read cached RF values." << endl;
	      return prepareExit(classifiers,cToClean,bootstraps,3);
	    }
	  }
	}
	else {
	  // save params to file
	  ofstream output;
	  if( makeNewCache(prefix,name.c_str(),output) ) {
	    output << ncycles << " " << nleaf << " " << nfeatures << endl;
	  }
	  else {
	    cerr << "Unable to make output file for RF." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	  
	  // make bootstrap object
	  SprIntegerBootstrap* bootstrap 
	    = new SprIntegerBootstrap(filter->dim(),nfeatures);
	  bootstraps.push_back(bootstrap);
	
	  // make decision tree
	  bool discrete = false;
	  SprTopdownTree* tree = new SprTopdownTree(filter.get(),&gini,
						    nleaf,discrete,bootstrap);
	  tree->useFastSort();
	  cToClean.push_back(tree);
	  
	  // make bagger
	  SprBagger* bagger = new SprBagger(filter.get(),ncycles,discrete);
	  classifiers.insert(pair<const string,SprAbsClassifier*>(name,
								  bagger));
	  if( !bagger->addTrainable(tree) ) {
	    cerr << "Unable to add a decision tree to Random Forest." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	}
	ostringstream ost;
	ost << name.c_str() 
	    << " = Random Forest with: nTrees=" << ncycles 
	    << " nEventsPerLeaf=" << nleaf 
	    << " nFeaturesToSample=" << nfeatures;
	message.insert(pair<const string,string>(name,ost.str()));
      }
    }
    
    //
    // arc-x4
    //
    if( answerYN("Use arc-x4?") ) {
      
      // get instances
      unsigned ninstance = 0;
      ifstream iinst;
      if( findCache(prefix,"AX4instances_",iinst) ) {
	string line;
	if( !getline(iinst,line,'\n') ) {
	  cerr << "Cannot read top line from AX4 cache." << endl;
	  return prepareExit(classifiers,cToClean,bootstraps,3);
	}
	istringstream str(line);
	str >> ninstance;
      }
      ninstance = 
	answerHowMany(ninstance,
	    "How many instances of the classifier would you like to run?");
      ofstream oinst;
      if( makeNewCache(prefix,"AX4instances_",oinst) ) {
	oinst << ninstance << endl;
      }
      else {
	cerr << "Unable to make output file for AX4." << endl;
	return prepareExit(classifiers,cToClean,bootstraps,3);
      }
      if( !moveNewOntoOld(prefix,"AX4instances_") )
	return prepareExit(classifiers,cToClean,bootstraps,3);
      
      // loop over instances
      for( unsigned int instance=0;instance<ninstance;instance++ ) {
	string name = "AX4_";
	char s [200];
	sprintf(s,"%i",instance);
	name += s;
	unsigned ncycles(0), nleaf(0), nfeatures(filter->dim());
	bool repeat = false;
	ifstream input;
	string line;
	if( findCache(prefix,name.c_str(),input) ) {
	  repeat = true;
	  if( !getline(input,line,'\n') ) {
	    cerr << "Cannot read top line from AX4 cache." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	  istringstream str(line);
	  str >> ncycles >> nleaf >> nfeatures;
	}
	cout << "Input number of trees, minimal tree leaf size, "
	     << "and number of input features for sampling "
	     << " for AX4 instance " << instance << " [" 
	     << ncycles << " " << nleaf << " " << nfeatures << "] ";
	getline(cin,line,'\n');
	if( !line.empty() ) {
	  repeat = false;
	  istringstream str(line);
	  str >> ncycles >> nleaf >> nfeatures;
	}
	if( repeat ) {
	  if( resetValidation(name.c_str(),validated,valFilter.get()) ) {
	    if( readCache(input,validated.find(name)->second) 
		!= valFilter->size() ) {
	      cerr << "Cannot read cached AX4 values." << endl;
	      return prepareExit(classifiers,cToClean,bootstraps,3);
	    }
	  }
	}
	else {
	  // save params to file
	  ofstream output;
	  if( makeNewCache(prefix,name.c_str(),output) ) {
	    output << ncycles << " " << nleaf << " " << nfeatures << endl;
	  }
	  else {
	    cerr << "Unable to make output file for AX4." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	  
	  // make bootstrap object
	  SprIntegerBootstrap* bootstrap 
	    = new SprIntegerBootstrap(filter->dim(),nfeatures);
	  bootstraps.push_back(bootstrap);
	  
	  // make decision tree
	  bool discrete = false;
	  SprTopdownTree* tree = new SprTopdownTree(filter.get(),&gini,
						    nleaf,discrete,bootstrap);
	  tree->useFastSort();
	  cToClean.push_back(tree);
	  
	  // make bagger
	  SprArcE4* arcer = new SprArcE4(filter.get(),ncycles,discrete);
	  classifiers.insert(pair<const string,SprAbsClassifier*>(name,arcer));
	  if( !arcer->addTrainable(tree) ) {
	    cerr << "Unable to add a decision tree to arc-x4." << endl;
	    return prepareExit(classifiers,cToClean,bootstraps,3);
	  }
	}
	ostringstream ost;
	ost << name.c_str() 
	    << " = arc-x4 with: nTrees=" << ncycles 
	    << " nEventsPerLeaf=" << nleaf 
	    << " nFeaturesToSample=" << nfeatures;
	message.insert(pair<const string,string>(name,ost.str()));
      }
    }
    
    //
    // Train all classifiers.
    //
    for( map<string,SprAbsClassifier*>::const_iterator 
	   iter=classifiers.begin();iter!=classifiers.end();iter++ ) {
      cout << "Training classifier " << iter->first.c_str() << endl;
      
      // train
      if( !iter->second->train(verbose) ) {
	cerr << "Classifier " << iter->first.c_str() << " cannot be trained. " 
	     << "Skipping..." << endl;
	continue;
      }
      
      // fill out vector of responses for validation data
      if( resetValidation(iter->first.c_str(),validated,valFilter.get()) ) {
	cout << "Saving responses for validation data for classifier "
	     << iter->first.c_str() << endl;
	
	// make a trained classifier
	SprAbsTrainedClassifier* t = iter->second->makeTrained();
	if( t == 0 ) {
	  cerr << "Unable to make trained classifier for " 
	       << iter->first.c_str() << endl;
	  continue;
	}

	// compute responses
	map<string,vector<SIAResponse> >::iterator found =
	  validated.find(iter->first);
	assert( found != validated.end() );
	for( unsigned int i=0;i<valFilter->size();i++ ) {
	  const SprPoint* p = (*(valFilter.get()))[i];
	  found->second.push_back(SIAResponse(int(inputClasses[1]==p->class_),
					   valFilter->w(i),
					   t->response(p)));
	}
	
	// clean up
	delete t;

	// store the vector into cache
	if( !storeNewCache(prefix,iter->first.c_str(),found->second) ) {
	  cerr << "Unable to save new cache for classifier " 
	       << iter->first.c_str() << endl;
	  return prepareExit(classifiers,cToClean,bootstraps,4);
	}
      }
      else {
	cerr << "Vector of responses for classifier " << iter->first.c_str()
	     << " already exists. Skipping..."<< endl;
      }
    }
    
    //
    // replace old cache with new cache
    //
    for( map<string,SprAbsClassifier*>::const_iterator 
	   iter=classifiers.begin();iter!=classifiers.end();iter++ ) {
      if( !moveNewOntoOld(prefix,iter->first.c_str()) )
	return prepareExit(classifiers,cToClean,bootstraps,5);
    }
    
    //
    // ask the user to specify signal efficiency points
    //
    vector<double> effS;
    ifstream effInput;
    if( findCache(prefix,"eff",effInput) ) {
      string line;
      if( getline(effInput,line) ) {
	istringstream str(line);
	double dummy = 0;
	while( str >> dummy ) effS.push_back(dummy);
      }
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
    effInput.close();
    cout << "Input signal efficiency values for which background "
	 << "will be estimated [ ";
    for( unsigned int i=0;i<effS.size();i++ ) 
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
      return prepareExit(classifiers,cToClean,bootstraps,7);
    }
    stable_sort(effS.begin(),effS.end());
    if( !storeEffCache(prefix,"eff",effS) ) {
      cerr << "Unable to store efficiency values." << endl;
      return prepareExit(classifiers,cToClean,bootstraps,6);
    }
    
    //
    // Estimate background fractions for these signal efficiencies.
    //
    map<string,vector<double> > effB;
    const double wsig = valFilter->weightInClass(inputClasses[1]);
    const double wbgr = valFilter->weightInClass(inputClasses[0]);
    assert( wsig > 0 );
    assert( wbgr > 0 );
    
    for( map<string,vector<SIAResponse> >::const_iterator 
	   iter=validated.begin();iter!=validated.end();iter++ ) {
      
      // prepare vectors
      vector<pair<double,double> > signal;
      vector<pair<double,double> > bgrnd;
      
      // fill them
      for( unsigned int i=0;i<iter->second.size();i++ ) {
	if(      iter->second[i].cls == 0 ) {
	  bgrnd.push_back(pair<double,double>(iter->second[i].response,
					      iter->second[i].weight));
	}
	else if( iter->second[i].cls == 1 ) {
	  signal.push_back(pair<double,double>(iter->second[i].response,
					       iter->second[i].weight));
	}
      }
      
      // sort
      stable_sort(bgrnd.begin(),bgrnd.end(),SIACmpPairDDFirst());
      stable_sort(signal.begin(),signal.end(),SIACmpPairDDFirst());
      
      // find dividing point in classifier response
      vector<double> cuts(effS.size());
      double w = 0;
      unsigned int divider = 0;
      unsigned int i = 0;
      while( i<signal.size() && divider<effS.size() ) {
	w += signal[i].second;
	if( (w/wsig) > effS[divider] ) {
	  if( i == 0 )
	    cuts[divider] = signal[0].first;
	  else
	    cuts[divider] = 0.5 * (signal[i].first + signal[i-1].first); 
	  divider++;
	}
	i++;
      }
      
      // find background fractions
      pair<map<string,vector<double> >::iterator,bool> inserted = 
	effB.insert(pair<const string,vector<double> >(iter->first,
					    vector<double>(effS.size(),1)));
      assert( inserted.second );
      w = 0;
      divider = 0;
      i = 0;
      while( i<bgrnd.size() && divider<effS.size() ) {
	if( bgrnd[i].first < cuts[divider] )
	  inserted.first->second[divider++] = w/wbgr;
	w += bgrnd[i].second;
	i++;
      }
    }
    
    //
    // make a table of signal and background efficiencies
    //
    cout << "===========================================" << endl;
    for( map<string,string>::const_iterator 
	   iter=message.begin();iter!=message.end();iter++ )
      cout << iter->second.c_str() << endl;
    cout << "===========================================" << endl;
    cout << "Table of surviving background fractions" 
	 << " (* shows minimal value in a row)" << endl;
    cout << "===========================================" << endl;
    char s[200];
    sprintf(s,"Signal Eff \\ Classifiers |");
    cout << s;
    string temp = "--------------------------";
    for( map<string,vector<double> >::const_iterator 
	   iter=effB.begin();iter!=effB.end();iter++ ) {
      sprintf(s," %8s |",iter->first.c_str());
      cout << s;
      temp += "-----------";
    }
    cout << endl;
    cout << temp.c_str() << endl;
    for( unsigned int i=0;i<effS.size();i++ ) {
      sprintf(s,"          %6.4f         |",effS[i]);
      cout << s;
      vector<string> names;
      vector<double> values;
      for( map<string,vector<double> >::const_iterator 
	     iter=effB.begin();iter!=effB.end();iter++ ) {
	names.push_back(iter->first);
	values.push_back(iter->second[i]);
      }
      unsigned int foundMin = min_element(values.begin(),values.end()) - values.begin();
      for( unsigned int j=0;j<names.size();j++ ) {
	if( j == foundMin )
	  sprintf(s," *%7.5f |",values[j]);
	else
	  sprintf(s,"  %7.5f |",values[j]);
	cout << s;
      }
      cout << endl;
    }
    cout << "===========================================" << endl;

    // save output to n-tuple
    if( answerYN("Save classifier output to an ntuple?") ) {

      // open file
      string ntupleFile 
	= answerName("Give name of the output ntuple file -->");
      auto_ptr<SprAbsWriter> 
	writer(SprRWFactory::makeWriter(writeMode,"ClassifierComparison"));
      if( !writer->init(ntupleFile.c_str()) ) {
	cerr << "Unable to open ntuple file " << ntupleFile.c_str() << endl;
	return prepareExit(classifiers,cToClean,bootstraps,7);
      }

      // make axes
      vector<string> axes;
      valFilter->vars(axes);
      if( !writer->setAxes(axes) ) {
	cerr << "Unable to set axes for the ntuple writer." << endl;
	return prepareExit(classifiers,cToClean,bootstraps,8);
      }
      for( map<string,vector<SIAResponse> >::const_iterator
	     iter=validated.begin();iter!=validated.end();iter++ ) {
	if( !writer->addAxis(iter->first.c_str()) ) {
	  cerr << "Unable to add a classifier axis for " 
	       << iter->first.c_str() << endl;
	  return prepareExit(classifiers,cToClean,bootstraps,8);
	}
      }

      // feed
      for( unsigned int i=0;i<valFilter->size();i++ ) {
	const SprPoint* p = (*(valFilter.get()))[i];
	double w = valFilter->w(i);
	vector<double> f;
	for( map<string,vector<SIAResponse> >::const_iterator
	     iter=validated.begin();iter!=validated.end();iter++ ) {
	  f.push_back((iter->second)[i].response);
	}
	if( !writer->write(p->class_,p->index_,w,p->x_,f) ) {
	  cerr << "Unable to write event " << p->index_ << endl;
	  return prepareExit(classifiers,cToClean,bootstraps,8);
	}
      }
      cout << "Writing to ntuple done." << endl;

      // clean up
      writer->close();
    }// end saving to file

    // clean up
    prepareExit(classifiers,cToClean,bootstraps,0);

    // exit?
    if( !answerYN("Continue?") ) break;
  }// end of the big loop
  
  // exit
  return 0;
}
  
