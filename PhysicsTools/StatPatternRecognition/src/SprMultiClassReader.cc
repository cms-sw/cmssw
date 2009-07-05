//$Id: SprMultiClassReader.cc,v 1.2 2007/09/21 22:32:10 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprMultiClassReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprMultiClassLearner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedMultiClassLearner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierReader.hh"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;


SprMultiClassReader::~SprMultiClassReader()
{
  for( unsigned int i=0;i<classifiers_.size();i++ ) {
    if( classifiers_[i].second )
      delete classifiers_[i].first;
  }
}


bool SprMultiClassReader::read(const char* filename)
{
  // sanity check
  if( !classifiers_.empty() ) {
    cerr << "You are attempting to re-read the saved multi class learner "
	 << "configuration without using the previous one." << endl;
    return false;
  }

  // open file
  string fname = filename;
  ifstream file(fname.c_str());
  if( !file ) {
    cerr << "Unable to open file " << fname.c_str() << endl;
    return false;
  }
  cout << "Reading MultiClassLearner from file " << fname.c_str() << endl;

  // read
  return this->read(file);
}


bool SprMultiClassReader::read(std::istream& input)
{
  // read indicator matrix
  string line;
  unsigned nLine = 0;
  for( unsigned int i=0;i<2;i++ ) {
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Cannot read from line " << nLine << endl;
      return false;
    }
  }
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line " << nLine << endl;
    return false;
  }
  if( line.find(':') != string::npos )
    line.erase(0,line.find_first_of(':')+1);
  else {
    cerr << "Cannot read from line " << nLine << endl;
    return false;
  }
  istringstream ist(line);
  unsigned nClasses(0), nClassifiers(0);
  ist >> nClasses >> nClassifiers;
  if( nClasses == 0 ) {
    cerr << "No classes found." << endl;
    return false;
  }
  if( nClassifiers == 0 ) {
    cerr << "No classifiers found." << endl;
    return false;
  }
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line " << nLine << endl;
    return false;
  }
  mapper_.clear();
  mapper_.resize(nClasses);
  SprMatrix mat(nClasses,nClassifiers,0);
  indicator_ = mat;
  for( unsigned int i=0;i<nClasses;i++ ) {
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Cannot read from line " << nLine << endl;
      return false;
    }
    string sclass, srow;
    if( line.find(':') != string::npos ) {
      sclass = line.substr(0,line.find_first_of(':'));
      srow = line.substr(line.find_first_of(':')+1);
    }
    else {
      cerr << "Cannot read from line " << nLine << endl;
      return false;
    }
    if( sclass.empty() ) {
      cerr << "Cannot read class on line " << nLine << endl;
      return false;
    }
    if( srow.empty() ) {
      cerr << "Cannot read matrix row on line " << nLine << endl;
      return false;
    }
    istringstream istclass(sclass), istrow(srow);
    istclass >> mapper_[i];
    for( unsigned int j=0;j<nClassifiers;j++ )
      istrow >> indicator_[i][j];
  }
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line " << nLine << endl;
    return false;
  }
  
  // read trained classifiers
  classifiers_.clear();
  classifiers_.resize(nClassifiers);
  for( unsigned int n=0;n<nClassifiers;n++ ) {
    // read index of the current classifier
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Cannot read from line " << nLine << endl;
      return false;
    }
    if( line.find(':') != string::npos )
      line.erase(0,line.find_first_of(':')+1);
    else {
      cerr << "Cannot read from line " << nLine << endl;
      return false;
    }
    istringstream istc(line);
    unsigned iClassifiers = 0;
    istc >> iClassifiers;
    if( iClassifiers != n ) {
      cerr << "Wrong classifier index on line " << nLine << endl;
      return false;
    }

    // read each classifier
    string requested;
    SprAbsTrainedClassifier* trained =
      SprClassifierReader::readTrainedFromStream(input,requested,nLine);
    if( trained == 0 ) {
      cerr << "Unable to read trained classifier " << n << endl;
      return false;
    }

    // add classifier to the list
    classifiers_[n] = pair<const SprAbsTrainedClassifier*,bool>(trained,true);
  }// end of loop over classifiers

  // read variables
  if( !SprClassifierReader::readVars(input,vars_,nLine) ) {
    cerr << "Unable to read variables." << endl;
    return false;
  }

  // exit
  return true;
}


void SprMultiClassReader::setTrainable(SprMultiClassLearner* multi)
{
  if( classifiers_.empty() ) {
    cerr << "Classifier list is empty in multi class reader." << endl;
    return;
  }
  assert( multi != 0 );
  multi->reset();
  multi->setTrained(indicator_,mapper_,classifiers_);
  classifiers_.clear();
}


SprTrainedMultiClassLearner* SprMultiClassReader::makeTrained()
{
  if( classifiers_.empty() ) {
    cerr << "Classifier list is empty in multi class reader." << endl;
    return 0;
  }
  SprTrainedMultiClassLearner* t 
    = new SprTrainedMultiClassLearner(indicator_,mapper_,classifiers_);
  classifiers_.clear();
  t->setVars(vars_);
  return t;
}


bool SprMultiClassReader::readIndicatorMatrix(const char* filename, 
					      SprMatrix& indicator)
{
  // open file
  string fname = filename;
  ifstream input(fname.c_str());
  if( !input ) {
    cerr << "Unable to open file " << fname.c_str() << endl;
    return false;
  }
  cout << "Reading indicator matrix from file " << fname.c_str() << endl;

  // read indicator matrix dimensionality
  unsigned N(0), M(0);
  string line;
  unsigned nLine = 0;
  while( getline(input,line) ) {
    // update line counter
    nLine++;

    // remove comments
    if( line.find('#') != string::npos )
      line.erase( line.find_first_of('#') );

    // skip empty line
    if( line.find_first_not_of(' ') == string::npos ) continue;

    // make stream
    istringstream ist(line);

    // read matrix dimensions
    ist >> N >> M;
    break;
  }
  if( N==0 || M==0 ) {
    cerr << "Unable to read indicator matrix dimensionality: " 
	 << N << " " << M << "    on line " << nLine << endl;
    return false;
  }

  // read the matrix itself
  SprMatrix temp(N,M,0);
  for( unsigned int n=0;n<N;n++ ) {
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Unable to read line " << nLine << endl;
      return false;
    }
    istringstream ist(line);
    for( unsigned int m=0;m<M;m++ ) ist >> temp[n][m];
  }

  // check columns of indicator matrix
  for( unsigned int m=0;m<M;m++ ) {
    unsigned countPlus(0), countMinus(0);
    for( unsigned int n=0;n<N;n++ ) {
      int elem = int(temp[n][m]);
      if(      elem == -1 )
	countMinus++;
      else if( elem == +1 )
	countPlus++;
      else if( elem != 0 ) {
	cerr << "Invalid indicator matrix element [" << n+1 << "]" 
	     << "[" << m+1 << "]=" << elem << endl;
	return false;
      }
    }
    if( countPlus==0 || countMinus==0 ) {
      cerr << "Column " << m+1 << " of the indicator matrix does not " 
	   << "have background and signal labels present." << endl;
      return false;
    }
  }

  // check rows
  for( unsigned int n=0;n<N;n++ ) {
    unsigned sum = 0;
    for( unsigned int m=0;m<M;m++ )
      sum += abs(int(temp[n][m]));
    if( sum == 0 ) {
      cerr << "Row " << n+1 << " of the indicator matrix has nothing "
	   << "but zeros." << endl;
      return false;
    }
  }

  // exit
  indicator = temp;
  return true;
}
