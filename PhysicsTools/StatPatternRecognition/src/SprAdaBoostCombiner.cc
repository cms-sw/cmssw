//$Id: SprAdaBoostCombiner.cc,v 1.3 2006/11/13 19:09:41 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAdaBoostCombiner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedAdaBoost.hh"

using namespace std;


bool SprAdaBoostCombiner::closeClassifierList()
{
  // prepare data for training
  if( !this->makeFeatures() ) {
    cerr << "Unable to make features for SprAdaBoostCombiner." << endl;
    return false;
  }

  // make AdaBoost
  ada_ = new SprAdaBoost(features_,cycles_,false,SprTrainedAdaBoost::Discrete);

  // exit
  return true;
}


SprTrainedAdaBoostCombiner* SprAdaBoostCombiner::makeTrained() const
{
  if( ada_ == 0 ) return 0;
  return new SprTrainedAdaBoostCombiner(classifiers_,ada_->makeTrained(),true);
}

