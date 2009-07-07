//$Id: SprAbsCombiner.cc,v 1.1 2007/05/23 04:16:44 rpw Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsCombiner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"

#include <cassert>

using namespace std;


SprAbsCombiner::SprAbsCombiner(SprAbsFilter* data) 
  : 
  SprAbsClassifier(data),
  classifiers_(),
  classifierLabels_(),
  features_(0)
{}


SprAbsCombiner::SprAbsCombiner(SprAbsFilter* data, 
			       const std::vector<
			       const SprAbsTrainedClassifier*>& c,
			       const std::vector<std::string>& cLabels)
  :
  SprAbsClassifier(data),
  classifiers_(c),
  classifierLabels_(cLabels),
  features_(0)
{
  assert( !classifiers_.empty() );
  assert( classifiers_.size() == classifierLabels_.size() );
}


bool SprAbsCombiner::makeFeatures()
{
  // size
  unsigned int nClassifiers = classifiers_.size();
  if( nClassifiers == 0 ) return false;
  assert( nClassifiers == classifierLabels_.size() );

  // make data
  SprData* features = new SprData("features",classifierLabels_);
  vector<double> r(nClassifiers);
  for( unsigned int i=0;i<data_->size();i++ ) {
    const SprPoint* p = (*data_)[i];
    for( unsigned int j=0;j<nClassifiers;j++ )
      r[j] = classifiers_[j]->response(p);
    features->insert(p->class_,r);
  }

  // get weights
  vector<double> weights;
  data_->weights(weights);

  // get classes
  vector<SprClass> classes;
  data_->classes(classes);

  // make filter
  features_ = new SprEmptyFilter(features,classes,weights,true);

  // exit
  return true;
}
