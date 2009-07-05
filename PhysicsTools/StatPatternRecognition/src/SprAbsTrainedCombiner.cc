//$Id: SprAbsTrainedCombiner.cc,v 1.2 2007/09/21 22:32:08 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedCombiner.hh"

using namespace std;


bool SprAbsTrainedCombiner::features(const std::vector<double>& v,
				     std::vector<double>& features) const
{
  if( classifiers_.empty() ) return false;
  features.clear();
  for( unsigned int i=0;i<classifiers_.size();i++ )
    features.push_back(classifiers_[i]->response(v));
  return true;
}


