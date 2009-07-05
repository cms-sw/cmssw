// $Id: SprVarTransformerReader.cc,v 1.1 2007/11/12 06:19:18 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprVarTransformerReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsVarTransformer.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPCATransformer.hh"
#include "PhysicsTools/StatPatternRecognition/src/SprMatrix.hh"

#include <fstream>
#include <sstream>
#include <utility>
#include <cassert>

using namespace std;


SprAbsVarTransformer* SprVarTransformerReader::read(const char* filename)
{
  // open file
  string fname = filename;
  ifstream is(fname.c_str());
  if( !is ) {
    cerr << "Unable to open file " << fname.c_str() << endl;
    return 0;
  }

  // exit
  return SprVarTransformerReader::read(is);
}


SprAbsVarTransformer* SprVarTransformerReader::read(std::istream& is)
{
  // init
  string line;
  unsigned nLine = 0;

  // read transformer name
  nLine++;
  if( !getline(is,line) ) {
    cerr << "Unable to read VarTransformer from line " << nLine << endl;
    return 0;
  }
  istringstream ist(line);
  string dummy, transformerName, version;
  ist >> dummy >> transformerName >> version;

  // decode name
  if( transformerName.empty() ) {
    cerr << "Unable to read VarTransformer name on line " << nLine << endl;
    return false;
  }
  SprAbsVarTransformer* t = 0;
  if( transformerName == "PCA" )
    t = SprVarTransformerReader::readPCATransformer(is,nLine);
  else {
    cerr << "Unknown VarTransformer name specified on line " << nLine << endl;
    return 0;
  }
  if( t == 0 ) return 0;
 
  // read vars
  vector<string> oldVars, newVars;
  if( !SprVarTransformerReader::readVars(is,nLine,oldVars,newVars) || 
      oldVars.empty() || newVars.empty() ) {
    cerr << "Unable to read VarTransformer variables." << endl;
    return 0;
  }
  t->setOldVars(oldVars);
  t->setNewVars(newVars);
  
  // exit
  return t;
}


bool SprVarTransformerReader::readVars(std::istream& is, unsigned& nLine,
				       std::vector<std::string>& oldVars,
				       std::vector<std::string>& newVars)
{
  // read old variables
  oldVars.clear();

  // skip 2 lines
  string line;
  for( int i=0;i<2;i++ ) {
    nLine++;
    if( !getline(is,line) ) {
      cerr << "Unable to read VarTransformer from line " << nLine << endl;
      return false;
    }
  }

  // read all lines skipping those that have nothing but =
  while( getline(is,line) ) {
    nLine++;

    // get rid of spaces
    line.erase( 0, line.find_first_not_of(' ') );
    line.erase( line.find_last_not_of(' ')+1 );

    // get rid of '='
    line.erase( 0, line.find_first_not_of('=') );
    line.erase( line.find_last_not_of('=')+1 );

    // if empty, do nothing
    if( line.empty() ) break;

    // add var
    istringstream ist(line);
    int index = -1;
    string var;
    ist >> index >> var;
    if( index != static_cast<int>(oldVars.size()) ) {
      cerr << "Incorrect VarTransformer variable index on line " 
	   << nLine << endl;
      return false;
    }
    oldVars.push_back(var);
  }

  // read old variables
  newVars.clear();

  // skip 2 lines
  for( int i=0;i<2;i++ ) {
    nLine++;
    if( !getline(is,line) ) {
      cerr << "Unable to read VarTransformer from line " << nLine << endl;
      return false;
    }
  }

  // read all lines skipping those that have nothing but =
  while( getline(is,line) ) {
    nLine++;

    // get rid of spaces
    line.erase( 0, line.find_first_not_of(' ') );
    line.erase( line.find_last_not_of(' ')+1 );

    // get rid of '='
    line.erase( 0, line.find_first_not_of('=') );
    line.erase( line.find_last_not_of('=')+1 );

    // if empty, do nothing
    if( line.empty() ) break;

    // add var
    istringstream ist(line);
    int index = -1;
    string var;
    ist >> index >> var;
    if( index != static_cast<int>(newVars.size()) ) {
      cerr << "Incorrect VarTransformer variable index on line " 
	   << nLine << endl;
      return false;
    }
    newVars.push_back(var);
  }

  // exit
  return true;
}


SprPCATransformer* SprVarTransformerReader::readPCATransformer(
					    std::istream& is, unsigned& nLine)
{
  // read dimensionality
  string line;
  nLine++;
  if( !getline(is,line) ) {
    cerr << "Unable to read VarTransformer from line " << nLine << endl;
    return 0;
  }
  istringstream ist(line);
  string dummy;
  int dim = -1;
  ist >> dummy >> dim;
  if( dim <= 0 ) {
    cerr << "Unable to read dimensionality from VarTransformer line " 
	 << nLine << endl;
    return 0;
  }
  
  // read eigenvalues
  vector<pair<double,int> > eigenValues(dim);
  nLine++;
  if( !getline(is,line) ) {
    cerr << "Unable to read VarTransformer from line " << nLine << endl;
    return 0;
  }
  istringstream ist_eigen(line);
  ist_eigen >> dummy;
  for( int d=0;d<dim;d++ )
    ist_eigen >> eigenValues[d].first;
  nLine++;
  if( !getline(is,line) ) {
    cerr << "Unable to read VarTransformer from line " << nLine << endl;
    return 0;
  }
  istringstream ist_index(line);
  ist_index >> dummy;
  for( int d=0;d<dim;d++ ) {
    ist_index >> eigenValues[d].second;
    assert( eigenValues[d].second >= 0 );
  }

  // read transformation matrix
  SprMatrix U(dim,dim);
  for( int i=0;i<dim;i++ ) {
    nLine++;
    if( !getline(is,line) ) {
      cerr << "Unable to read VarTransformer from line " << nLine << endl;
      return 0;
    }
    istringstream istU(line);
    int d = -1;
    istU >> d;
    if( d != i ) {
      cerr << "Dimension of VarTransformer does not macth on line " 
	   << nLine << endl;
      return 0;
    }
    for( int j=0;j<dim;j++ )
      istU >> dummy >> dummy >> U[i][j];
  }

  // make PCA transformer
  SprPCATransformer* t = new SprPCATransformer(U,eigenValues);

  // exit
  return t;
}
