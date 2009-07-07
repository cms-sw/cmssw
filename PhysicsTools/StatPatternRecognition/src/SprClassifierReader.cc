//$Id: SprClassifierReader.cc,v 1.3 2007/10/30 18:56:14 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClassifierReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprNNDefs.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerBootstrap.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassIDFraction.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassGiniIndex.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTwoClassCrossEntropy.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprCoordinateMapper.hh"

#include "PhysicsTools/StatPatternRecognition/interface/SprStdBackprop.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedStdBackprop.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTopdownTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedTopdownTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDecisionTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedDecisionTree.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedNode.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprFisher.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedFisher.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprLogitR.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedLogitR.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBinarySplit.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedBinarySplit.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBagger.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprArcE4.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedBagger.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprCombiner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedCombiner.hh"

#include "PhysicsTools/StatPatternRecognition/src/SprVector.hh"
#include "PhysicsTools/StatPatternRecognition/src/SprSymMatrix.hh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <cassert>

using namespace std;

//
// Template declarations.
//
/*
template SprTrainedStdBackprop* 
SprClassifierReader::readTrained<SprTrainedStdBackprop>(const char* filename, 
							const char* classifier,
							int verbose);
template SprTrainedTopdownTree* 
SprClassifierReader::readTrained<SprTrainedTopdownTree>(const char* filename, 
							const char* classifier,
							int verbose);
template SprTrainedDecisionTree* 
SprClassifierReader::readTrained<SprTrainedDecisionTree>(const char* filename, 
						       const char* classifier,
							 int verbose);
template SprTrainedFisher* 
SprClassifierReader::readTrained<SprTrainedFisher>(const char* filename, 
						   const char* classifier,
						   int verbose);
template SprTrainedLogitR* 
SprClassifierReader::readTrained<SprTrainedLogitR>(const char* filename, 
						   const char* classifier,
						   int verbose);
template SprTrainedBinarySplit* 
SprClassifierReader::readTrained<SprTrainedBinarySplit>(const char* filename, 
							const char* classifier,
							int verbose);
template SprTrainedAdaBoost* 
SprClassifierReader::readTrained<SprTrainedAdaBoost>(const char* filename, 
						     const char* classifier,
						     int verbose);
template SprTrainedBagger* 
SprClassifierReader::readTrained<SprTrainedBagger>(const char* filename, 
						   const char* classifier,
						   int verbose);
template SprTrainedCombiner* 
SprClassifierReader::readTrained<SprTrainedCombiner>(const char* filename, 
						     const char* classifier,
						     int verbose);

template SprTrainedStdBackprop* 
SprClassifierReader::readTrained<SprTrainedStdBackprop>(std::istream& input, 
							const char* classifier,
							int verbose);
template SprTrainedTopdownTree* 
SprClassifierReader::readTrained<SprTrainedTopdownTree>(std::istream& input, 
							const char* classifier,
							int verbose);
template SprTrainedDecisionTree* 
SprClassifierReader::readTrained<SprTrainedDecisionTree>(std::istream& input, 
						       const char* classifier,
							 int verbose);
template SprTrainedFisher* 
SprClassifierReader::readTrained<SprTrainedFisher>(std::istream& input, 
						   const char* classifier,
						   int verbose);
template SprTrainedLogitR* 
SprClassifierReader::readTrained<SprTrainedLogitR>(std::istream& input, 
						   const char* classifier,
						   int verbose);
template SprTrainedBinarySplit* 
SprClassifierReader::readTrained<SprTrainedBinarySplit>(std::istream& input, 
							const char* classifier,
							int verbose);
template SprTrainedAdaBoost* 
SprClassifierReader::readTrained<SprTrainedAdaBoost>(std::istream& input, 
						     const char* classifier,
						     int verbose);
template SprTrainedBagger* 
SprClassifierReader::readTrained<SprTrainedBagger>(std::istream& input, 
						   const char* classifier,
						   int verbose);
template SprTrainedCombiner* 
SprClassifierReader::readTrained<SprTrainedCombiner>(std::istream& input, 
						     const char* classifier,
						     int verbose);
*/


bool SprClassifierReader::readTrainable(const char* filename, 
					SprAbsClassifier* trainable,
					int verbose)
{
  // open file
  string fname = filename;
  ifstream file(fname.c_str());
  if( !file ) {
    cerr << "Unable to open file " << fname.c_str() << endl;
    return false;
  }
  if( verbose > 0 ) {
    cout << "Reading classifier configuration from file " 
	 << fname.c_str() << endl;
  }

  // read
  return SprClassifierReader::readTrainable(file,trainable,verbose);
}


bool SprClassifierReader::readTrainable(std::istream& input, 
					SprAbsClassifier* trainable,
					int verbose)
{
  // set line counter
  unsigned nLine = 1;

  // read clasifier name
  string found = SprClassifierReader::readClassifierName(input);
  if( found.empty() ) {
    cerr << "Unable to read classifier name on line " << nLine << endl;
    return false;
  }
  if( verbose > 0 )
    cout << "Found classifier " << found.c_str() << endl;

  // if requested classifier is supplied, make sure it matches
  string requested = trainable->name();
  if( !requested.empty() && (requested!=found) ) {
    cerr << "Requested classifier " << requested.c_str() 
	 << " does not macth to the actual stored classifier " 
	 << found.c_str() << " on line " << nLine << endl;
    return false;
  }

  // switch between classifier types
  if(      requested == "StdBackprop" ) {
    if( !SprClassifierReader::readStdBackprop(input,
				   static_cast<SprStdBackprop*>(trainable),
					      nLine) ) {
      cerr << "Unable to read classifier " << requested.c_str() << endl;
      return false;
    }
  }
  else if( requested == "AdaBoost" ) {
    if( !SprClassifierReader::readAdaBoost(input,
				   static_cast<SprAdaBoost*>(trainable),
					   nLine) ) {
      cerr << "Unable to read classifier " << requested.c_str() << endl;
      return false;
    }
  }
  else if( requested=="Bagger" || requested=="ArcE4" ) {
    if( !SprClassifierReader::readBagger(input,
				   static_cast<SprBagger*>(trainable),
					 nLine) ) {
      cerr << "Unable to read classifier " << requested.c_str() << endl;
      return false;
    }
  }
  else if( requested == "TopdownTree" ) {
    cerr << "Readout of trainable TopdownTree not implemented." << endl;
    return false;
  }
  else if( requested == "DecisionTree" ) {
    cerr << "Readout of trainable DecisionTree not implemented." << endl;
    return false;
  }
  else if( requested == "Fisher" ) {
    cerr << "Readout of trainable Fisher not implemented." << endl;
    return false;
  }
  else if( requested == "LogitR" ) {
    cerr << "Readout of trainable LogitR not implemented." << endl;
    return false;
  }
   else if( requested == "BinarySplit" ) {
    cerr << "Readout of trainable BinarySplit not implemented." << endl;
    return false;
  }
   else if( requested == "Combiner" ) {
    cerr << "Readout of trainable Combiner not implemented." << endl;
    return false;
  }
 else {
    cerr << "Unknown classifier requested." << endl;
    return false;
  }

  // exit
  return true;
}


SprAbsTrainedClassifier* SprClassifierReader::readTrained(
					      const char* filename,
					      int verbose)
{
  // open file
  string fname = filename;
  ifstream file(fname.c_str());
  if( !file ) {
    cerr << "Unable to open file " << fname.c_str() << endl;
    return 0;
  }
  if( verbose > 0 ) {
    cout << "Reading classifier configuration from file " 
	 << fname.c_str() << endl;
  }

  // read
  return SprClassifierReader::readTrained(file,verbose);
}


SprAbsTrainedClassifier* SprClassifierReader::readTrained(std::istream& input,
							  int verbose)
{
  // make empty classifier name
  string requested;

  // start line counter
  unsigned nLine = 0;

  // read
  SprAbsTrainedClassifier* t 
    = SprClassifierReader::readTrainedFromStream(input,requested,nLine);
  if( t == 0 ) return 0;

  // set vars
  vector<string> vars;
  if( !SprClassifierReader::readVars(input,vars,nLine) ) {
    cerr << "Unable to read variables in SprClassifierReader::readTrained." 
	 << endl;
    return 0;
  }
  t->setVars(vars);

  // exit
  return t;
}


template<class T> T* SprClassifierReader::readTrained(const char* filename, 
						      const char* classifier,
						      int verbose)
{
  // open file
  string fname = filename;
  ifstream file(fname.c_str());
  if( !file ) {
    cerr << "Unable to open file " << fname.c_str() << endl;
    return 0;
  }
  if( verbose > 0 ) {
    cout << "Reading classifier configuration from file " 
	 << fname.c_str() << endl;
  }

  // read
  return SprClassifierReader::readTrained<T>(file,classifier,verbose);
}


template<class T> T* SprClassifierReader::readTrained(std::istream& input, 
						      const char* classifier,
						      int verbose)
{
  // request a specific classifier
  string requested = classifier;

  // start line counter
  unsigned nLine = 0;

  // read specific classifier
  SprAbsTrainedClassifier* t
    = SprClassifierReader::readTrainedFromStream(input,requested,nLine);
  if( t == 0 ) return 0;

  // set vars
  vector<string> vars;
  if( !SprClassifierReader::readVars(input,vars,nLine) ) {
    cerr << "Unable to read variables in SprClassifierReader::readTrained." 
	 << endl;
    return 0;
  }
  t->setVars(vars);

  // exit
  return static_cast<T*>(t);
}


SprAbsTrainedClassifier* SprClassifierReader::readTrainedFromStream(
					       std::istream& input,
					       const std::string& requested,
					       unsigned& nLine)
{
  // read clasifier name
  nLine++;
  string found = SprClassifierReader::readClassifierName(input);
  if( found.empty() ) {
    cerr << "Unable to read classifier name on line " << nLine << endl;
    return 0;
  }

  // if requested classifier is supplied, make sure it matches
  if( !requested.empty() && (requested!=found) ) {
    cerr << "Requested classifier " << requested.c_str() 
	 << " does not match to the actual stored classifier " 
	 << found.c_str() << " on line " << nLine << endl;
    return 0;
  }

  // read specific classifier
  return SprClassifierReader::readSelectedTrained(input,found,nLine);
}


std::string SprClassifierReader::readClassifierName(std::istream& input)
{
  // read current line
  string line;
  if( !getline(input,line) ) {
    cerr << "Cannot read from input." << endl;
    return "";
  }

  // get 2nd field
  istringstream ist(line);
  string dummy;
  string classifierName;
  ist >> dummy >> classifierName;

  // remove ":"
  if( classifierName.find(':') != string::npos )
    classifierName.erase(classifierName.find_first_of(':'));

  // exit
  return classifierName;
}


SprAbsTrainedClassifier* SprClassifierReader::readSelectedTrained(
					       std::istream& input,
					       const std::string& requested,
					       unsigned& nLine)
{
  // init
  SprAbsTrainedClassifier* trained = 0;
  SprData data;
  SprEmptyFilter filter(&data);

  // switch between classifier types
  if(      requested == "StdBackprop" ) {
    auto_ptr<SprStdBackprop> trainable(new SprStdBackprop(&filter));
    if( !SprClassifierReader::readStdBackprop(input,trainable.get(),nLine) ) {
      cerr << "Unable to read classifier " << requested.c_str() << endl;
      return 0;
    }
    trained = trainable->makeTrained();
  }
  else if( requested == "AdaBoost" ) {
    auto_ptr<SprAdaBoost> trainable(new SprAdaBoost(&filter));
    if( !SprClassifierReader::readAdaBoost(input,trainable.get(),nLine) ) {
      cerr << "Unable to read classifier " << requested.c_str() << endl;
      return 0;
    }
    trained = trainable->makeTrained();
  }
  else if( requested=="Bagger" || requested=="ArcE4" ) {
    auto_ptr<SprBagger> trainable(new SprBagger(&filter));
    if( !SprClassifierReader::readBagger(input,trainable.get(),nLine) ) {
      cerr << "Unable to read classifier " << requested.c_str() << endl;
      return 0;
    }
    trained = trainable->makeTrained();
  }
  else if( requested == "TopdownTree" ) {
    return SprClassifierReader::readTopdownTree(input,nLine);
  }
  else if( requested == "DecisionTree" ) {
    return SprClassifierReader::readDecisionTree(input,nLine);
  }
  else if( requested == "Fisher" ) {
    return SprClassifierReader::readFisher(input,nLine);
  }
  else if( requested == "LogitR" ) {
    return SprClassifierReader::readLogitR(input,nLine);
  }
  else if( requested == "BinarySplit" ) {
    return SprClassifierReader::readBinarySplit(input,nLine);
  }
  else if( requested == "Combiner" ) {
    return SprClassifierReader::readCombiner(input,nLine);
  }
  else {
    cerr << "Unknown classifier requested." << endl;
    return 0;
  }

  // exit
  return trained;
}


bool SprClassifierReader::readStdBackprop(std::istream& input, 
					  SprStdBackprop* trainable,
					  unsigned& nLine)
{
  // sanity check
  assert( trainable != 0 );

  // init
  string structure = "Unknown";
  //bool configured = false;
  //bool initialized = false;

  // read header
  string line;
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Unable to read line " << nLine << endl;
    return false;
  }

  // read the cut
  string dummy;
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Unable to read line " << nLine << endl;
    return false;
  }
  istringstream istcut(line);
  istcut >> dummy;
  unsigned int nCut = 0;
  istcut >> nCut;
  SprCut cut;
  double low(0), high(0);
  for( unsigned int i=0;i<nCut;i++ ) {
    istcut >> low >> high;
    cut.push_back(pair<double,double>(low,high));
  }

  // read number of nodes
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Unable to read line " << nLine << endl;
    return false;
  }
  istringstream istNnodes(line);
  int nNodes = 0;
  istNnodes >> dummy >> nNodes;
  if( nNodes <= 0 ) {
    cerr << "Rean an invalid number of NN nodes: " << nNodes 
	 << " on line " << nLine << endl;
    return false;
  }
  
  // init nodes
  vector<SprNNDefs::NodeType>   nodeType(nNodes,SprNNDefs::INPUT);
  vector<SprNNDefs::ActFun>     nodeActFun(nNodes,SprNNDefs::ID);
  vector<double>                nodeAct(nNodes,0);
  vector<double>                nodeOut(nNodes,0);
  vector<int>                   nodeNInputLinks(nNodes,0);
  vector<int>                   nodeFirstInputLink(nNodes,-1);
  vector<double>                nodeBias(nNodes,0);

  // read nodes
  for( int node=0;node<nNodes;node++ ) {
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Unable to read line " << nLine << endl;
      return false;
    }
    istringstream istnode(line);
    int index = -1;
    istnode >> index;
    if( index != node ) {
      cerr << "Incorrect node number on line " << nLine
	   << ": Expect " << node << " Actual " << index << endl;
      return false;
    }
    istnode >> dummy;
    char readNodeType;
    istnode >> readNodeType;
    switch( readNodeType )
      {
      case 'I' :
	nodeType[node] = SprNNDefs::INPUT;
	break;
      case 'H' :
	nodeType[node] = SprNNDefs::HIDDEN;
	break;
      case 'O' :
	nodeType[node] = SprNNDefs::OUTPUT;
	break;
      default :
	cerr << "Unknown node type on line " << nLine << endl;
	return false;
      }
    istnode >> dummy;
    int actFun = 0;
    istnode >> actFun;
    switch( actFun )
      {
      case 1 :
	nodeActFun[node] = SprNNDefs::ID;
	break;
      case 2 :
	nodeActFun[node] = SprNNDefs::LOGISTIC;
	break;
      default :
	cerr << "Unknown activation function on line " << nLine << endl;
	return false;
      }
    istnode >> dummy;
    istnode >> nodeNInputLinks[node];
    istnode >> dummy;
    istnode >> nodeFirstInputLink[node];
    istnode >> dummy;
    istnode >> nodeBias[node];
  }// nodes done

  // read number of links
  int nLinks = 0;
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Unable to read line " << nLine << endl;
    return false;
  }
  istringstream istNlinks(line);
  istNlinks >> dummy >> nLinks;
  if( nLinks <= 0 ) {
    cerr << "Rean an invalid number of NN links: " << nLinks 
	 << " on line " << nLine << endl;
    return false;
  }
  
  // init links
  vector<int>                   linkSource(nLinks,0);
  vector<double>                linkWeight(nLinks,0);

  // read links
  for( int link=0;link<nLinks;link++ ) {
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Unable to read line " << nLine << endl;
      return false;
    }
    istringstream istlink(line);
    int index = -1;
    istlink >> index;
    if( index != link ) {
      cerr << "Incorrect link number on line " << nLine
	   << ": Expect " << link << " Actual " << index << endl;
      return false;
    }
    istlink >> dummy;
    istlink >> linkSource[link];
    istlink >> dummy;
    istlink >> linkWeight[link];
  }// links done

  // set params for the supplied NN
  trainable->structure_               = structure;
  trainable->configured_              = true;
  trainable->initialized_             = true;
  trainable->nNodes_                  = nNodes;
  trainable->nLinks_                  = nLinks;
  trainable->nodeType_                = nodeType;
  trainable->nodeActFun_              = nodeActFun;
  trainable->nodeAct_                 = nodeAct;
  trainable->nodeOut_                 = nodeOut;
  trainable->nodeNInputLinks_         = nodeNInputLinks;
  trainable->nodeFirstInputLink_      = nodeFirstInputLink;
  trainable->linkSource_              = linkSource;
  trainable->nodeBias_                = nodeBias;
  trainable->linkWeight_              = linkWeight;
  trainable->cut_                     = cut;

  // exit
  return true;
}


bool SprClassifierReader::readAdaBoost(std::istream& input, 
				       SprAdaBoost* trainable,
				       unsigned& nLine)
{
  // sanity check
  assert( trainable != 0 );

  // read number of weak classifiers
  string line;
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read line " << nLine << endl;
    return false;
  }
  if( line.find(':') != string::npos )
    line.erase(0,line.find_first_of(':')+1);
  else {
    cerr << "Cannot read line " << nLine << endl;
    return false;
  }
  istringstream ist(line);
  unsigned nClassifiers = 0;
  ist >> nClassifiers;
  if( nClassifiers == 0 ) {
    cerr << "No classifiers found." << endl;
    return false;
  }
      
  // read AdaBoost mode
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read line " << nLine << endl;
    return false;
  }
  if( line.find(':') != string::npos )
    line.erase(0,line.find_first_of(':')+1);
  else {
    cerr << "Cannot read from line " << nLine << endl;
    return false;
  }
  istringstream istmode(line);
  int abMode = 0;
  istmode >> abMode;
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
      cerr << "Unknown mode for AdaBoost " << abMode << endl;
      return false;
  }
  double epsilon = 0.01;
  if( mode == SprTrainedAdaBoost::Real ) {
    if( line.find(':') != string::npos ) {
      line.erase(0,line.find_first_of(':')+1);
      istringstream isteps(line);
      isteps >> epsilon;
    }
    else {
      cout << "Epsilon not provided for Real AdaBoost on line " 
	   << nLine << endl;
      cout << "Will assume default value." << endl;
    }
  }

  // read betas
  vector<double> beta(nClassifiers,0);
  string dummy;
  unsigned index;
  for( unsigned int i=0;i<nClassifiers;i++ ) {
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Cannot read line " << nLine << endl;
      return false;
    }
    istringstream istindex(line);
    istindex >> dummy;
    istindex >> index;
    if( index != i ) {
      cerr << "Wrong classifier index for beta " << index 
	   << " " << i << " line " << nLine << endl;
      return false;
    }
    if( line.find(':') != string::npos )
      line.erase(0,line.find_first_of(':')+1);
    else {
      cerr << "Cannot read from line " << nLine << endl;
      return false;
    }
    istringstream istbeta(line);
    istbeta >> beta[i];
  }

  // skip "Classifiers" line
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read line " << nLine << endl;
    return false;
  }  
 
  // read weak classifiers
  vector<pair<const SprAbsTrainedClassifier*,bool> > weak(nClassifiers);
  for( unsigned int i=0;i<nClassifiers;i++ ) {
    // read index
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Cannot read from line " << nLine << endl;
      return false;
    }
    int index = -1;
    istringstream istindex(line);
    string dummy;
    istindex >> dummy;
    istindex >> index;
    if( index != (int)i ) {
      cerr << "Wrong classifier index " << index 
	   << " " << i << " line " << nLine << endl;
      return false;
    }

    // read classifier name
    string weakClassifier;
    istindex >> weakClassifier;

    // read trained classifier
    const SprAbsTrainedClassifier* trained
      = SprClassifierReader::readTrainedFromStream(input,weakClassifier,nLine);
    if( trained == 0 ) {
      cerr << "Unable to read weak classifier " 
	   << weakClassifier.c_str() << endl;
      return false;
    }
    weak[i].first = trained;
    weak[i].second = true;
  } // end of reading loop

  // set AdaBoost
  if( weak.empty() || beta.empty() ) {
    cerr << "Classifier list is empty in AdaBoost reader." << endl;
    return false;
  }
  trainable->reset();
  trainable->setTrained(weak,beta);
  trainable->setMode(mode);
  trainable->setEpsilon(epsilon);

  // exit
  return true;
}


SprTrainedTopdownTree* SprClassifierReader::readTopdownTree(
                                          std::istream& input, 
					  unsigned& nLine)
{
  // read number of nodes
  nLine++;
  string line;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  if( line.find(':') != string::npos )
    line.erase(0,line.find_first_of(':')+1);
  else {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  istringstream istn(line);
  unsigned nNodes(0);
  istn >> nNodes;
  if( nNodes == 0 ) {
    cerr << "Tree has no nodes at line " << nLine << endl;
    return 0;
  }

  // loop over nodes
  SprTopdownNodeMap mapped;
  string dummy;
  for( unsigned int j=0;j<nNodes;j++ ) {
    // reset
    dummy.clear();

    // read node index and dimensionality
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Cannot read from line " << nLine << endl;
      return 0;
    }
    istringstream istnode(line);
    int dau1(-1), dau2(-1);
    SprTrainedNode* node = new SprTrainedNode;
    istnode >> dummy;
    istnode >> node->id_;
    istnode >> dummy;
    istnode >> node->score_;
    istnode >> dummy;
    istnode >> node->d_;
    istnode >> dummy;
    istnode >> node->cut_;
    istnode >> dummy;
    istnode >> dau1 >> dau2;
    if( node->id_ < 0 ) {
      cerr << "Node id is negative on line " << nLine << endl;
      SprClassifierReader::prepareTopdownTreeExit(mapped);
      return 0;
    }
    if( node->d_<0 && (dau1>=0 || dau2>=0)) {
      cerr << "Node dimension is negative on line " << nLine << endl;
      SprClassifierReader::prepareTopdownTreeExit(mapped);
      return 0;
    }
      if( !SprClassifierReader::addTopdownTreeNode(node,dau1,dau2,mapped) ) {
	cerr << "Unable to add node on line " << nLine << endl;
	SprClassifierReader::prepareTopdownTreeExit(mapped);
	return 0;
      }
    }// end nodes loop
  
  // make nodes
  vector<const SprTrainedNode*> nodes;
  if( !SprClassifierReader::makeTopdownTreeNodeList(mapped,nodes) ) {
    cerr << "Unable to make a list of nodes." << endl;
    SprClassifierReader::prepareTopdownTreeExit(mapped);
    return 0;
  }

  // make decision tree
  return new SprTrainedTopdownTree(nodes,true);
}


bool SprClassifierReader::addTopdownTreeNode(SprTrainedNode* node, 
					     int dau1, int dau2,
					     SprTopdownNodeMap& mapped)
{
  pair<int,int> daus(dau1,dau2);
  pair<SprTrainedNode*,pair<int,int> > nodeWithDaus(node,daus);
  pair<const int,pair<SprTrainedNode*,pair<int,int> > > 
    nodeElement(node->id_,nodeWithDaus);
  return mapped.insert(nodeElement).second;
}


bool SprClassifierReader::makeTopdownTreeNodeList(const SprTopdownNodeMap& 
						  mapped,
						  std::vector<
						  const SprTrainedNode*>& 
						  nodes) 
{
  // init
  nodes.clear();

  // resolve mother/daughter references
  for( SprTopdownNodeMap::const_iterator 
	 i=mapped.begin();i!=mapped.end();i++ ) {
    SprTrainedNode* node = i->second.first;
    int dau1 = i->second.second.first;
    int dau2 = i->second.second.second;
    if( (dau1<0 && dau2>=0) || (dau1>=0 && dau2<0) ) {
      cerr << "Daughters are set incorrectly: " << dau1 << " " << dau2 << endl;
      return false;
    }
    if( dau1 >= 0 ) {
      SprTopdownNodeMap::const_iterator iter = mapped.find(dau1);
      node->toDau1_ = iter->second.first;
      iter->second.first->toParent_ = node;
    }
    if( dau2 >= 0 ) {
      SprTopdownNodeMap::const_iterator iter = mapped.find(dau2);
      node->toDau2_ = iter->second.first;
      iter->second.first->toParent_ = node;
    }
  }

  // convert the map into a plain vector
  nodes.clear();
  for( SprTopdownNodeMap::const_iterator i=mapped.begin();i!=mapped.end();i++ )
    nodes.push_back(i->second.first);

  // sanity check
  if( nodes.empty() || nodes[0]->id_!=0 ) {
    cerr << "Root node of the tree is misconfigured." << endl;
    return false;
  }

  // exit
  return true;
}


void SprClassifierReader::prepareTopdownTreeExit(
				      const SprTopdownNodeMap& mapped)
{
  for( SprTopdownNodeMap::const_iterator 
	 iter=mapped.begin();iter!=mapped.end();iter++ )
    delete iter->second.first;
}


SprTrainedDecisionTree* SprClassifierReader::readDecisionTree(
					  std::istream& input, 
					  unsigned& nLine)
{
  // read number of nodes
  nLine++;
  string line;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  if( line.find(':') != string::npos )
    line.erase(0,line.find_first_of(':')+1);
  else {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  istringstream istn(line);
  unsigned nNodes(0);
  istn >> nNodes;
  if( nNodes == 0 ) {
    cerr << "Tree has no nodes at line " << nLine << endl;
    return 0;
  }
  vector<SprBox> nodes1(nNodes);

  // loop over nodes
  string dummy;
  for( unsigned int j=0;j<nNodes;j++ ) {
    dummy.clear();
    // read node index and dimensionality
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Cannot read from line " << nLine << endl;
      return 0;
    }
    istringstream istnode(line);
    istnode >> dummy;
    int nodeIndex = 0;
    istnode >> nodeIndex;
    if( nodeIndex != (int)j ) {
      cerr << "Wrong node index " << nodeIndex << " at line " << nLine 
	   << ".  Must be " << j << endl;
      return 0;
    }
    istnode >> dummy;
    unsigned dim = 0;
    istnode >> dim;
    if( dim == 0 ) continue;
    SprBox node;
    // read node bounds
    for( unsigned int k=0;k<dim;k++ ) {
      nLine++;
      if( !getline(input,line) ) {
	cerr << "Cannot read from line " << nLine << endl;
	return 0;
      }
      unsigned d = 0;
      istringstream istlimits(line);
      istlimits >> dummy;
      istlimits >> d;
      istlimits >> dummy;
      double xa(0), xb(0);
      istlimits >> xa;
      istlimits >> xb;
      if( xa > xb ) {
	cerr << "Incorrect node limits at line " << nLine << endl;
	return 0;
      }
      node.insert(pair<const unsigned,
		  pair<double,double> >(d,pair<double,double>(xa,xb)));
    }// end node bounds loop
    
    // add a new node
    nodes1[j] = node;
  }// end nodes loop

  // make decision tree
  return new SprTrainedDecisionTree(nodes1);
}


SprTrainedFisher* SprClassifierReader::readFisher(std::istream& input, 
						  unsigned& nLine)
{
  string line;

  // read dimensionality
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  if( line.find(':') != string::npos )
    line.erase(0,line.find_first_of(':')+1);
  else {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  istringstream istdim(line);
  unsigned dim = 0;
  istdim >> dim;
  if( dim == 0 ) {
    cerr << "Fisher dimensionality cannot be zero." << endl;
    return 0;
  }

  // skip 2 lines
  for( unsigned int i=0;i<2;i++ ) {
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Cannot read from line " << nLine << endl;
      return 0;
    }
  }

  // read order
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  if( line.find(':') != string::npos )
    line.erase(0,line.find_first_of(':')+1);
  else {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  istringstream istorder(line);
  unsigned order = 0;
  istorder >> order;
  if( order!=1 && order!=2 ) {
    cerr << "Fisher can only handle order 1 or 2: " << order << endl;
    return 0;
  }

  // read const term
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  if( line.find(':') != string::npos )
    line.erase(0,line.find_first_of(':')+1);
  else {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  istringstream istcterm(line);
  double cterm = 0;
  istcterm >> cterm;

  // skip 1 line
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }

  // read linear part
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  istringstream istlinear(line);
  SprVector linear(dim);
  for( unsigned int d=0;d<dim;d++ )
    istlinear >> linear[d];

  // if LDA, make Fisher
  if( order == 1 )
    return new SprTrainedFisher(linear,cterm);

  // if QDA, read quadratic part
  if( order == 2 ) {
    // skip one line
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Cannot read from line " << nLine << endl;
      return 0;
    }

    // read quadratic part
    SprSymMatrix quadr(dim);
    for( unsigned int i=0;i<dim;i++ ) {
      nLine++;
      if( !getline(input,line) ) {
	cerr << "Cannot read from line " << nLine << endl;
	return 0;
      }
      istringstream istrow(line);
      for( unsigned int j=0;j<dim;j++ )
	istrow >> quadr[i][j];
    }

    // make Fisher
    return new SprTrainedFisher(linear,quadr,cterm);
  }

  // if we come that far, something must be wrong
  return 0;
}


SprTrainedLogitR* SprClassifierReader::readLogitR(std::istream& input, 
						  unsigned& nLine)
{
  string line;

  // read dimensionality
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  if( line.find(':') != string::npos )
    line.erase(0,line.find_first_of(':')+1);
  else {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  istringstream istdim(line);
  unsigned dim = 0;
  istdim >> dim;
  if( dim == 0 ) {
    cerr << "LogitR dimensionality cannot be zero." << endl;
    return 0;
  }

  // skip 2 lines
  for( unsigned int i=0;i<2;i++ ) {
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Cannot read from line " << nLine << endl;
      return 0;
    }
  }

  // read const term
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  if( line.find(':') != string::npos )
    line.erase(0,line.find_first_of(':')+1);
  else {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  istringstream istbeta0(line);
  double beta0 = 0;
  istbeta0 >> beta0;

  // skip one line
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }

  // read beta coefficients  
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  istringstream istbeta(line);
  SprVector beta(dim);
  for( unsigned int i=0;i<dim;i++ ) istbeta >> beta[i];

  // exit
  return new SprTrainedLogitR(beta0,beta);
}


bool SprClassifierReader::readBagger(std::istream& input, 
				     SprBagger* trainable,
				     unsigned& nLine)
{
  // sanity check
  assert( trainable != 0 );

  // read number of weak classifiers
  string line;
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read line " << nLine << endl;
    return false;
  }
  if( line.find(':') != string::npos )
    line.erase(0,line.find_first_of(':')+1);
  else {
    cerr << "Cannot read line " << nLine << endl;
    return false;
  }
  istringstream ist(line);
  unsigned nClassifiers = 0;
  ist >> nClassifiers;
  if( nClassifiers == 0 ) {
    cerr << "No classifiers found." << endl;
    return false;
  }
      
  // read weak classifiers
  vector<pair<const SprAbsTrainedClassifier*,bool> > weak(nClassifiers);
  for( unsigned int i=0;i<nClassifiers;i++ ) {
    // read index
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Cannot read from line " << nLine << endl;
      return false;
    }
    int index = -1;
    istringstream istindex(line);
    string dummy;
    istindex >> dummy;
    istindex >> index;
    if( index != (int)i ) {
      cerr << "Wrong classifier index " << index 
	   << " " << i << " line " << nLine << endl;
      return false;
    }

    // read classifier name
    string weakClassifier;
    istindex >> weakClassifier;

    // read trained classifier
    const SprAbsTrainedClassifier* trained
      = SprClassifierReader::readTrainedFromStream(input,weakClassifier,nLine);
    if( trained == 0 ) {
      cerr << "Unable to read weak classifier " 
	   << weakClassifier.c_str() << endl;
      return false;
    }
    weak[i].first = trained;
    weak[i].second = true;
  } // end of reading loop

  // set Bagger
  if( weak.empty() ) {
    cerr << "Classifier list is empty in Bagger reader." << endl;
    return false;
  }
  trainable->reset();
  trainable->setTrained(weak);

  // exit
  return true;
}


SprTrainedBinarySplit* SprClassifierReader::readBinarySplit(
				       std::istream& input, 
				       unsigned& nLine)
{
  string line;

  // read dimension
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line "<< nLine << endl;
    return 0;
  }
  if( line.find(':') != string::npos )
    line.erase(0,line.find_first_of(':')+1);
  else {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  unsigned dim = 0;
  istringstream istdim(line);
  istdim >> dim;

  // read cut
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  if( line.find(':') != string::npos )
    line.erase(0,line.find_first_of(':')+1);
  else {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  istringstream istcut(line);
  unsigned nCut = 0;
  istcut >> nCut;
  SprCut cut(nCut);
  for( unsigned int j=0;j<nCut;j++ ) {
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Cannot read from line " << nLine << endl;
      return 0;
    }
    istringstream istpair(line);
    double low(0), high(0);
    istpair >> low >> high;
    cut[j] = pair<double,double>(low,high);
  }

  // exit
  return new SprTrainedBinarySplit(dim,cut);
}


SprTrainedCombiner* SprClassifierReader::readCombiner(std::istream& input,
						      unsigned& nLine)
{
  // read the number of sub-classifiers
  string line;
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  istringstream istnsub(line);
  string dummy;
  unsigned nSub = 0;
  istnsub >> dummy >> nSub;
  if( nSub == 0 ) {
    cerr << "No subclassifiers found on line " << nLine << endl;
    return 0;
  }

  // init sub-classifiers containers
  vector<pair<const SprAbsTrainedClassifier*,bool> > trained(nSub);
  vector<string> labels(nSub);
  vector<map<unsigned,SprCut> > constraints(nSub);
  vector<SprCoordinateMapper*> inputDataMappers(nSub);
  vector<double> defaultValues(nSub);

  // read sub-classifiers
  for( unsigned int is=0;is<nSub;is++ ) {
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Cannot read from line " << nLine << endl;
      return 0;
    }
    istringstream istsub(line);
    int index = -1;
    istsub >> dummy >> index >> dummy 
	   >> labels[is] >> dummy >> defaultValues[is];
    if( index != (int)is ) {
      cerr << "Wrong classifier index on line " << nLine
	   << " : " << index << " Expected: " << is << endl;
      return 0;
    }
    if( labels[is].empty() ) {
      cerr << "Cannot read classifier name on line " << nLine << endl;
      return 0;
    }

    // read vars
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Cannot read from line " << nLine << endl;
      return 0;
    }
    istringstream istnvar(line);
    unsigned nVars = 0;
    istnvar >> dummy >> nVars;
    if( nVars == 0 ) {
      cerr << "No variables found on line " << nLine << endl;
      return 0;
    }
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Cannot read from line " << nLine << endl;
      return 0;
    }
    istringstream istvar(line);
    vector<string> vars(nVars);
    for( unsigned int iv=0;iv<nVars;iv++ ) {
      istvar >> vars[iv];
      if( vars[iv].empty() ) {
	cerr << "Cannot read variable name " << iv 
	     << " on line " << nLine << endl;
      }
    }

    // read mappers
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Cannot read from line " << nLine << endl;
      return 0;
    }
    istringstream istnmap(line);
    unsigned nMap = 0;
    istnmap >> dummy >> nMap;
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Cannot read from line " << nLine << endl;
      return 0;
    }
    istringstream istmap(line);
    vector<unsigned> mapper(nMap);
    for( unsigned int im=0;im<nMap;im++ )
      istmap >> mapper[im];
    inputDataMappers[is] = SprCoordinateMapper::createMapper(mapper);
    if( inputDataMappers[is] == 0 ) {
      cerr << "Cannot read coordinate mapper." << endl;
      return 0;
    }

    // read constraints
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Cannot read from line " << nLine << endl;
      return 0;
    }
    istringstream istnconstr(line);
    unsigned nConstr = 0;
    istnconstr >> dummy >> nConstr;
    for( unsigned int j=0;j<nConstr;j++ ) {
      nLine++;
      if( !getline(input,line) ) {
	cerr << "Cannot read from line " << nLine << endl;
	return 0;
      }
      istringstream istconstr(line);
      int ivar = -1;
      unsigned nCut = 0;
      SprCut cut;
      istconstr >> ivar >> nCut;
      if( ivar<0 || ivar>=static_cast<int>(vars.size()) ) {
	cerr << "Wrong variable index on line " << nLine 
	     << " : " << ivar << endl;
	return 0;
      }
      double xa(0), xb(0);
      for( unsigned int k=0;k<nCut;k++ ) {
	istconstr >> xa >> xb;
	cut.push_back(SprInterval(xa,xb));
      }
      constraints[is].insert(pair<const unsigned,SprCut>(ivar,cut));
    }

    // read trained classifier
    string requested;
    SprAbsTrainedClassifier* t =
      SprClassifierReader::readTrainedFromStream(input,requested,nLine);
    if( t == 0 ) {
      cerr << "Unable to read trained classifier " << is << endl;
      return 0;
    }
      
    // add trained classifier
    bool ownTrained = true;
    trained[is] = pair<const SprAbsTrainedClassifier*,bool>(t,ownTrained);
  }// end of sub-classifier loop

  // read overall classifier
  string requested;
  SprAbsTrainedClassifier* overall = 
    SprClassifierReader::readTrainedFromStream(input,requested,nLine);
  if( overall == 0 ) {
    cerr << "Unable to read overall trained classifier." << endl;
    return 0;
  }

  // read features of the overall classifier
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  istringstream istnfeat(line);
  unsigned nFeat = 0;
  istnfeat >> dummy >> nFeat;
  if( nFeat == 0 ) {
    cerr << "No features found on line " << nLine << endl;
    return 0;
  }
  nLine++;
  if( !getline(input,line) ) {
    cerr << "Cannot read from line " << nLine << endl;
    return 0;
  }
  istringstream istfeat(line);
  vector<string> fVars(nFeat);
  for( unsigned int d=0;d<nFeat;d++ )
    istfeat >> fVars[d];
  overall->setVars(fVars);

  // make trained combiner
  bool ownOverall = true;
  return new SprTrainedCombiner(overall,trained,labels,constraints,
				inputDataMappers,defaultValues,ownOverall);
}


bool SprClassifierReader::readTrainableConfig(
			      std::istream& input,
			      unsigned& nLine,
			      SprAbsFilter* data,
			      bool discreteTree, 
			      bool mixedNodesTree,
			      bool fastSortTree,
			      std::vector<SprAbsTwoClassCriterion*>& criteria,
			      std::vector<SprIntegerBootstrap*>& bstraps,
			      std::vector<SprAbsClassifier*>& classifiers,
			      std::vector<SprCCPair>& ccPairs,
			      bool readOneEntry)
{
  // read classifier params
  string line;
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

    // read classifier name
    string classifierName;
    ist >> classifierName;
    cout << "Reading configuration for classifier " 
	 << classifierName.c_str() << endl;

    // 
    // topdown tree
    //
    if(      classifierName == "TopdownTree" ) {
      int iCrit = 0;// criterion for decision tree optimization
      unsigned nFeatures = 0;// features to try for each decision split
      int nLeaf = 0;// events per leaf
      int iRand = 0;// if negative, generate seed for bstrap from time of day
      ist >> iCrit >> nFeatures >> nLeaf >> iRand;

      // make criterion
      SprAbsTwoClassCriterion* crit = 0;
      switch( iCrit )
	{
	case 1 :
	  crit = new SprTwoClassIDFraction;
	  cout << "Optimization criterion set to "
	       << "Fraction of correctly classified events " << endl;
	  break;
	case 5 :
	  crit = new SprTwoClassGiniIndex;
	  cout << "Optimization criterion set to "
	       << "Gini index  -1+p^2+q^2 " << endl;
	  break;
	case 6 :
	  crit = new SprTwoClassCrossEntropy;
	  cout << "Optimization criterion set to "
	       << "Cross-entropy p*log(p)+q*log(q) " << endl;
	  break;
	default :
	  cerr << "Unable to make initialization criterion." << endl;
	  return false;
	}
      criteria.push_back(crit);
      
      // make bootstrap for feature sampling
      SprIntegerBootstrap* boot = 0;
      if( nFeatures > 0 ) {
	boot = new SprIntegerBootstrap(data->dim(),nFeatures,iRand);
	bstraps.push_back(boot);
      }
      
      // make decision tree
      bool discrete = discreteTree;
      SprTopdownTree* tree = new SprTopdownTree(data,crit,
						nLeaf,discrete,boot);
      if( mixedNodesTree ) tree->forceMixedNodes();
      if( fastSortTree ) tree->useFastSort();
      classifiers.push_back(tree);

      // add decision tree
      cout << "Adding topdown tree with: iCrit=" << iCrit
	   << " nFeaturesToSample=" << nFeatures
	   << " nEventsPerLeaf=" << nLeaf << endl;
      ccPairs.push_back(SprCCPair(tree,SprUtils::lowerBound(0.5)));
      if( readOneEntry ) return true;
    }// end of decision tree
    //
    // AdaBoost
    //
    else if( classifierName == "AdaBoost" ) {
      unsigned cycles = 0;
      int abMode = 0;
      int iBagInput = 0;
      double epsilon = 0;
      ist >> cycles >> abMode >> iBagInput >> epsilon;

      // check cycles
      if( cycles == 0 ) {
	cerr << "No training cycles requested for AdaBoost." << endl;
	return false;
      }

      // decode AdaBoost mode
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
	  return false;
	}

      // check epsilon
      if( mode!=SprTrainedAdaBoost::Discrete && epsilon<SprUtils::eps() ) {
	cerr << "Epsilon set to small for AdBoost: " << epsilon << endl;
	return false;
      }

      // make AdaBoost
      bool useStandardAB = false;
      bool bagInput = (iBagInput == 1);
      SprAdaBoost* ab = new SprAdaBoost(data,cycles,
					useStandardAB,mode,bagInput);
      ab->setEpsilon(epsilon);
      classifiers.push_back(ab);

      // collect trainable classifiers
      bool discrete = (mode!=SprTrainedAdaBoost::Real);
      bool mixedNodes = (mode==SprTrainedAdaBoost::Real);
      bool fastSort = true;
      vector<SprCCPair> abTrainablePairs;
      if( !SprClassifierReader::readTrainableConfig(input,nLine,data,
						    discrete,mixedNodes,
						    fastSort,criteria,
						    bstraps,classifiers,
						    abTrainablePairs,
						    true) ) {
	cerr << "Unable to read weak classifier for AdaBoost " 
	     << "on line " << nLine << endl;
	return false;
      }

      // add trainable classifiers to AdaBoost
      for( unsigned int i=0;i<abTrainablePairs.size();i++ ) {
	if( !ab->addTrainable(abTrainablePairs[i].first,
			      abTrainablePairs[i].second) ) {
	  cerr << "Unable to add classifier " << i << " of type " 
	       << abTrainablePairs[i].first->name() << " to AdaBoost." << endl;
	  return false;
	}
      }

      // add AdaBoost
      cout << "Adding AdaBoost with: nCycle=" << cycles
	   << " AdaBoostMode=" << abMode
	   << " BagInput=" << iBagInput
	   << " Epsilon=" << epsilon << endl;
      ccPairs.push_back(SprCCPair(ab,SprCut()));
      if( readOneEntry ) return true;
    }// end AdaBoost
    //
    // bagger
    //
    else if( classifierName=="Bagger" || classifierName=="ArcE4" ) {
      unsigned cycles = 0;
      int iRand = 0;
      ist >> cycles >> iRand;

      // check cycles
      if( cycles == 0 ) {
	cerr << "No training cycles requested for Bagger." << endl;
	return false;
      }

      // make bagger
      bool discrete = false;
      SprBagger* bagger = 0;
      if( classifierName == "Bagger" )
	bagger = new SprBagger(data,cycles,discrete);
      else
	bagger = new SprArcE4(data,cycles,discrete);
      classifiers.push_back(bagger);

      // collect trainable classifiers
      bool mixedNodes = false;
      bool fastSort = true;
      vector<SprCCPair> baggerTrainablePairs;
      if( !SprClassifierReader::readTrainableConfig(input,nLine,data,
						    discrete,mixedNodes,
						    fastSort,criteria,
						    bstraps,classifiers,
						    baggerTrainablePairs,
						    true) ) {
	cerr << "Unable to read weak classifier for Bagger " 
	     << "on line " << nLine << endl;
	return false;
      }

      // add trainable classifiers to Bagger
      for( unsigned int i=0;i<baggerTrainablePairs.size();i++ ) {
	if( !bagger->addTrainable(baggerTrainablePairs[i].first) ) {
	  cerr << "Unable to add classifier " << i << " of type " 
	       << baggerTrainablePairs[i].first->name() 
	       << " to Bagger." << endl;
	  return false;
	}
      }

      // add bagger
      cout << "Adding Bagger with: nCycle=" << cycles << endl;
      ccPairs.push_back(SprCCPair(bagger,SprCut()));
      if( readOneEntry ) return true;
    }
    //
    // neural net
    //
    else if( classifierName == "StdBackprop" ) {
      string structure;// NN structure
      unsigned nnCycles(0);// training cycles
      unsigned initPoints(0);// number of points for initialization
      double eta(0);// learning rate
      double initEta(0);// learning rate for initialization
      ist >> structure >> nnCycles >> eta >> initPoints >> initEta;

      // make neural net
      SprStdBackprop* stdnn =
	new SprStdBackprop(data,structure.c_str(),nnCycles,eta);
      classifiers.push_back(stdnn);
      if( !stdnn->init(initEta,initPoints) ) {
	cerr << "Unable to initialize neural net." << endl;
	return false;
      }

      // add neural net
      cout << "Adding a neural net with:"
	   << " structure=" << structure.c_str()
            << " nCyclesPerNet=" << nnCycles
            << " LearnRate=" << eta
            << " nInitPoints=" << initPoints
	   << " InitLearnRate=" << initEta << endl;
      ccPairs.push_back(SprCCPair(stdnn,SprCut()));
      if( readOneEntry ) return true;
    }// end of std backprop
    //
    // Fisher
    //
    else if( classifierName == "Fisher" ) {
      int order = 0;// Fisher order
      ist >> order;
      if( order!=1 && order!=2 ) {
	cerr << "Invalid order for Fisher: " << order << endl;
	return false;
      }
      SprFisher* fisher = new SprFisher(data,order);
      classifiers.push_back(fisher);
      cout << "Adding Fisher with: Order=" << order << endl;
      ccPairs.push_back(SprCCPair(fisher,SprCut()));
      if( readOneEntry ) return true;
    }
    //
    // LogitR
    //
    else if( classifierName == "LogitR" ) {
      double eps = 0;// accuracy
      double updateFactor = 0;// update factor
      int initToZero = 0;// initialization flag
      ist >> eps >> updateFactor >> initToZero;

      // check params
      if( eps < SprUtils::eps() ) {
	cerr << "Accuracy for LogitR set too small: " << eps 
	     << "   Recommended 0.001" << endl;
	return false;
      }
      if( updateFactor < SprUtils::eps() ) {
	cerr << "Update factor for LogitR set too small: " 
	     << updateFactor << "    Recommended 0.5-1" << endl;
	return false;
      }
      if( initToZero!=0 && initToZero!=1 ) {
	cerr << "Invalid initialization flag: " << initToZero 
	     << " Must be 0 or 1." << endl;
	return false;
      }	

      // make LogitR
      double beta0 = 0;
      SprVector beta;
      if( initToZero == 0 ) {
	SprVector dummy(data->dim());
	beta = dummy;
	for( unsigned int i=0;i<data->dim();i++ ) beta[i] = 0;
      }
      SprLogitR* logit = new SprLogitR(data,beta0,beta,
				       eps,updateFactor);
      classifiers.push_back(logit);
      cout << "Adding Logistic Regression with: " 
	   << " Eps=" << eps
	   << " updateFactor=" << updateFactor
	   << " initFlag=" << initToZero << endl;
      ccPairs.push_back(SprCCPair(logit,SprCut()));
      if( readOneEntry ) return true;
    }
    //
    // binary split
    //
    else if( classifierName == "BinarySplit" ) {
      // make criterion
      SprTwoClassIDFraction* crit = new SprTwoClassIDFraction;
      criteria.push_back(crit);

      // make splits
      for( unsigned int d=0;d<data->dim();d++ ) {
	SprBinarySplit* split = new SprBinarySplit(data,crit,d);
	classifiers.push_back(split);
	cout << "Adding binary split on dimension " << d << endl;
	ccPairs.push_back(SprCCPair(split,SprUtils::lowerBound(0.5)));
      }
      if( readOneEntry ) return true;
    }
    //
    // unknown classifier
    //
    else {
      cerr << "Unknown classifier " << classifierName.c_str() << endl;
      return false;
    }
  }// end of config input

  // exit
  return true;
}


bool SprClassifierReader::readVars(std::istream& input, 
				   std::vector<std::string>& vars,
				   unsigned& nLine)
{
  // init
  vars.clear();

  // skip 2 lines
  string line;
  for( unsigned int i=0;i<2;i++ ) {
    nLine++;
    if( !getline(input,line) ) {
      cerr << "Unable to read from line " << nLine << endl;
      return false;
    }
  }

  // read all lines skipping those that have nothing but =
  while( getline(input,line) ) {
    nLine++;

    // get rid of spaces
    line.erase( 0, line.find_first_not_of(' ') );
    line.erase( line.find_last_not_of(' ')+1 );

    // get rid of '='
    line.erase( 0, line.find_first_not_of('=') );
    line.erase( line.find_last_not_of('=')+1 );

    // if empty, do nothing
    if( line.empty() ) continue;

    // add var
    istringstream ist(line);
    int index = -1;
    string var;
    ist >> index >> var;
    if( index != static_cast<int>(vars.size()) ) {
      cerr << "Incorrect variable index on line " << nLine << endl;
      return false;
    }
    vars.push_back(var);
  }

  // exit
  return true;
}
