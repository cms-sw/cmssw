//$Id: SprTrainedStdBackprop.cc,v 1.2 2007/09/21 22:32:10 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedStdBackprop.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformation.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"

#include <cmath>
#include <iomanip>
#include <cassert>

using namespace std;


SprTrainedStdBackprop::SprTrainedStdBackprop()
  : 
  SprAbsTrainedClassifier()
  , nNodes_(0)
  , nLinks_(0)
  , structure_()
  , nodeType_()
  , nodeActFun_()
  , nodeNInputLinks_()
  , nodeFirstInputLink_()
  , linkSource_()
  , nodeBias_()
  , linkWeight_()
{
  this->setCut(SprUtils::lowerBound(0.5));
}


SprTrainedStdBackprop::SprTrainedStdBackprop(
			const char* structure,
                        const std::vector<SprNNDefs::NodeType>& nodeType,
			const std::vector<SprNNDefs::ActFun>& nodeActFun,
			const std::vector<int>& nodeNInputLinks,
			const std::vector<int>& nodeFirstInputLink,
			const std::vector<int>& linkSource,
			const std::vector<double>& nodeBias,
			const std::vector<double>& linkWeight)
  :
  SprAbsTrainedClassifier(),
  nNodes_(0),
  nLinks_(0),
  structure_(structure),
  nodeType_(nodeType),
  nodeActFun_(nodeActFun),
  nodeNInputLinks_(nodeNInputLinks),
  nodeFirstInputLink_(nodeFirstInputLink),
  linkSource_(linkSource),
  nodeBias_(nodeBias),
  linkWeight_(linkWeight)
{
  nNodes_ = nodeType_.size();
  assert( nNodes_ == nodeActFun_.size() );
  assert( nNodes_ == nodeNInputLinks_.size() );
  assert( nNodes_ == nodeFirstInputLink_.size() );
  assert( nNodes_ == nodeBias_.size() );
  nLinks_ = linkSource_.size();
  assert( nLinks_ == linkWeight_.size() );
  this->setCut(SprUtils::lowerBound(0.5));
}


SprTrainedStdBackprop::SprTrainedStdBackprop(
			  const SprTrainedStdBackprop& other)
  :
  SprAbsTrainedClassifier(other)
  , nNodes_(other.nNodes_)
  , nLinks_(other.nLinks_)
  , structure_(other.structure_)
  , nodeType_(other.nodeType_)
  , nodeActFun_(other.nodeActFun_)
  , nodeNInputLinks_(other.nodeNInputLinks_)
  , nodeFirstInputLink_(other.nodeFirstInputLink_)
  , linkSource_(other.linkSource_)
  , nodeBias_(other.nodeBias_)
  , linkWeight_(other.linkWeight_)
{}


double SprTrainedStdBackprop::activate(double x, SprNNDefs::ActFun f) const 
{
  switch (f) 
    {
    case SprNNDefs::ID :
      return x;
      break;
    case SprNNDefs::LOGISTIC :
      return SprTransformation::logit(x);
      break;
    default :
      cerr << "FATAL ERROR: Unknown activation function " 
	   << f << " in SprTrainedStdBackprop::activate" << endl;
      return 0;
    }
  return 0;
}


void SprTrainedStdBackprop::print(std::ostream& os) const 
{
  os << "Trained StdBackprop with configuration " 
     << structure_.c_str() << " " << SprVersion << endl; 
  os << "Activation functions: Identity=1, Logistic=2" << endl;
  os << "Cut: " << cut_.size();
  for( unsigned int i=0;i<cut_.size();i++ )
    os << "      " << cut_[i].first << " " << cut_[i].second;
  os << endl;
  os << "Nodes: " << nNodes_ << endl;
  for( unsigned int i=0;i<nNodes_;i++ ) {
    char nodeType = 0;
    switch( nodeType_[i] )
      {
      case SprNNDefs::INPUT :
	nodeType = 'I';
	break;
      case SprNNDefs::HIDDEN :
	nodeType = 'H';
	break;
      case SprNNDefs::OUTPUT :
	nodeType = 'O';
	break;
      }
    int actFun = 0;
    switch( nodeActFun_[i] )
      {
      case SprNNDefs::ID :
	actFun = 1;
	break;
      case SprNNDefs::LOGISTIC :
	actFun = 2;
	break;
      }
    os << setw(6) << i
       << "    Type: "           << nodeType
       << "    ActFunction: "    << actFun
       << "    NInputLinks: "    << setw(6) << nodeNInputLinks_[i]
       << "    FirstInputLink: " << setw(6) << nodeFirstInputLink_[i]
       << "    Bias: "           << nodeBias_[i]
       << endl;
  }
  os << "Links: " << nLinks_ << endl;
  for( unsigned int i=0;i<nLinks_;i++ ) {
    os << setw(6) << i
       << "    Source: " << setw(6) << linkSource_[i]
       << "    Weight: " << linkWeight_[i]
       << endl;
  }
}


double SprTrainedStdBackprop::response(const std::vector<double>& v) const
{
  // Initialize and process input nodes
  vector<double> nodeOut(nNodes_,0);
  unsigned int d = 0;
  for( unsigned int i=0;i<nNodes_;i++ ) {
    if( nodeType_[i] == SprNNDefs::INPUT ) {
      assert( d < v.size() );
      nodeOut[i] = v[d++];
    }
    else
      break;
  }
  assert( d == v.size() );

  // Process hidden and output nodes
  for( unsigned int i=0;i<nNodes_;i++ ) {
    double nodeAct = 0;
    if( nodeNInputLinks_[i] > 0 ) {
      for( int j=nodeFirstInputLink_[i];
	   j<nodeFirstInputLink_[i]+nodeNInputLinks_[i];j++ ) {
        nodeAct += nodeOut[linkSource_[j]] * linkWeight_[j];
      }
      nodeOut[i] = this->activate(nodeAct+nodeBias_[i],nodeActFun_[i]);
    }
  }

  // Find output node and return result
  return nodeOut[nNodes_-1];
}
