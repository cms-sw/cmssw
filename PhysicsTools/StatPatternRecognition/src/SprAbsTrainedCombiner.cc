//$Id: SprAbsTrainedCombiner.cc,v 1.3 2006/11/13 19:09:41 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedCombiner.hh"

using namespace std;


bool SprAbsTrainedCombiner::features(const std::vector<double>& v,
				     std::vector<double>& features) const
{
  if( classifiers_.empty() ) return false;
  features.clear();
  for( int i=0;i<classifiers_.size();i++ )
    features.push_back(classifiers_[i]->response(v));
  return true;
}


