//$Id: SprTrainedBinarySplit.cc,v 1.2 2007/09/21 22:32:10 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedBinarySplit.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"

#include <stdio.h>
#include <utility>
#include <cassert>

using namespace std;


SprTrainedBinarySplit::SprTrainedBinarySplit(unsigned d, 
					     const SprCut& inputCut) 
  :  
  SprAbsTrainedClassifier(), 
  d_(d), 
  inputCut_(inputCut) 
{
  // set cut on classifier output
  this->setCut(SprUtils::lowerBound(0.5));
}

SprTrainedBinarySplit::SprTrainedBinarySplit(const SprTrainedBinarySplit& 
					     other)
  : 
  SprAbsTrainedClassifier(other),
  d_(other.d_),
  inputCut_(other.inputCut_)
{}


double SprTrainedBinarySplit::response(const std::vector<double>& v) const
{
  // sanity check
  assert( d_ < v.size() );

  // cut
  int accept = 1;
  if( !inputCut_.empty() ) {
    for( unsigned int i=0;i<inputCut_.size();i++ ) {
      if( v[d_]<inputCut_[i].first || v[d_]>inputCut_[i].second ) {
	accept = 0;
	break;
      }
    }
  }

  // exit
  return accept;
}


void SprTrainedBinarySplit::print(std::ostream& os) const
{
  os << "Trained BinarySplit " << SprVersion << endl;
  os << "Dimension: " << d_ << endl;
  os << "Cut: " << inputCut_.size() << endl;
  for( unsigned int i=0;i<inputCut_.size();i++ ) {
    char s [200];
    sprintf(s,"%10g %10g",inputCut_[i].first,inputCut_[i].second);
    os << s << endl;
  }
}

