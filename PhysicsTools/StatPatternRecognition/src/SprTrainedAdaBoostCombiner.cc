//$Id: SprTrainedAdaBoostCombiner.cc,v 1.3 2006/11/13 19:09:43 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedAdaBoostCombiner.hh"

using namespace std;


double SprTrainedAdaBoostCombiner::response(const std::vector<double>& v) 
  const
{
  vector<double> features;
  if( !this->features(v,features) ) {
    cerr << "Unable to extract features in SprTrainedAdaBoostCombiner." 
	 << endl;
    return 0;
  }
  return ada_->response(features);
}


