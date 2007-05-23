//$Id: SprMultiClassReader.cc,v 1.1 2007/02/05 21:49:46 narsky Exp $

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
  for( int i=0;i<classifiers_.size();i++ ) {
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

  // read indicator matrix
  string line;
  unsigned nLine = 0;
  for( int i=0;i<2;i++ ) {
    nLine++;
    if( !getline(file,line) ) {
      cerr << "Cannot read from " << fname.c_str() 
	   << " line " << nLine << endl;
      return false;
    }
  }
  nLine++;
  if( !getline(file,line) ) {
    cerr << "Cannot read from " << fname.c_str() 
	 << " line " << nLine << endl;
    return false;
  }
  if( line.find(':') != string::npos )
    line.erase(0,line.find_first_of(':')+1);
  else {
    cerr << "Cannot read from " << fname.c_str() 
	 << " line " << nLine << endl;
    return false;
  }
  istringstream ist(line);
  unsigned nClasses(0), nClassifiers(0);
  ist >> nClasses >> nClassifiers;
  if( nClasses == 0 ) {
    cerr << "No classes found in " << fname.c_str() << endl;
    return false;
  }
  if( nClassifiers == 0 ) {
    cerr << "No classifiers found in " << fname.c_str() << endl;
    return false;
  }
  nLine++;
  if( !getline(file,line) ) {
    cerr << "Cannot read from " << fname.c_str() 
	 << " line " << nLine << endl;
    return false;
  }
  mapper_.clear();
  mapper_.resize(nClasses);
  SprMatrix mat(nClasses,nClassifiers,0);
  indicator_ = mat;
  for( int i=0;i<nClasses;i++ ) {
    nLine++;
    if( !getline(file,line) ) {
      cerr << "Cannot read from " << fname.c_str() 
	   << " line " << nLine << endl;
      return false;
    }
    string sclass, srow;
    if( line.find(':') != string::npos ) {
      sclass = line.substr(0,line.find_first_of(':'));
      srow = line.substr(line.find_first_of(':')+1);
    }
    else {
      cerr << "Cannot read from " << fname.c_str() 
	   << " line " << nLine << endl;
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
    for( int j=0;j<nClassifiers;j++ )
      istrow >> indicator_[i][j];
  }
  nLine++;
  if( !getline(file,line) ) {
    cerr << "Cannot read from " << fname.c_str() 
	 << " line " << nLine << endl;
    return false;
  }
  
  // read trained classifiers
  classifiers_.clear();
  classifiers_.resize(nClassifiers);
  for( int n=0;n<nClassifiers;n++ ) {
    // read index of the current classifier
    nLine++;
    if( !getline(file,line) ) {
      cerr << "Cannot read from " << fname.c_str() 
	   << " line " << nLine << endl;
      return false;
    }
    if( line.find(':') != string::npos )
      line.erase(0,line.find_first_of(':')+1);
    else {
      cerr << "Cannot read from " << fname.c_str() 
	   << " line " << nLine << endl;
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
      SprClassifierReader::readTrainedFromFile(file,requested,nLine);
    if( trained == 0 ) {
      cerr << "Unable to read trained classifier " << n 
	   << " from file " << fname.c_str() << endl;
      return false;
    }

    // add classifier to the list
    classifiers_[n] = pair<const SprAbsTrainedClassifier*,bool>(trained,true);
  }// end of loop over classifiers

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
  return t;
}
