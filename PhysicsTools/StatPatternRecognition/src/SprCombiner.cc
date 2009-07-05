//$Id: SprCombiner.cc,v 1.1 2007/09/21 22:32:09 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprCombiner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprCoordinateMapper.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"

#include <algorithm>

using namespace std;


SprCombiner::~SprCombiner() 
{ 
  delete features_; 
  for( unsigned int i=0;i<trained_.size();i++ ) {
    if( trained_[i].second )
      delete trained_[i].first;
  }
  for( unsigned int i=0;i<inputDataMappers_.size();i++ )
    delete inputDataMappers_[i];
}


SprCombiner::SprCombiner(SprAbsFilter* data) 
  : 
  SprAbsClassifier(data),
  trainable_(0),
  features_(0),
  trained_(),
  labels_(),
  constraints_(),
  inputDataMappers_(),
  defaultValues_()
{}


bool SprCombiner::train(int verbose)
{
  // sanity check
  if( trainable_==0 || trained_.empty() ) {
    cerr << "Cannot train Combiner - not all classifiers defined." << endl;
    return false;
  }
  if( features_ == 0 ) {
    cerr << "Classifier list has not been closed." << endl;
    return false;
  }

  // set data
  if( !trainable_->setData(features_) ) {
    cerr << "Unable to set data for trainable classifier " 
	 << trainable_->name().c_str() << endl;
    return false;
  }

  // train
  return trainable_->train(verbose);
}


bool SprCombiner::reset()
{
  return trainable_->reset();
}


bool SprCombiner::setData(SprAbsFilter* data)
{
  // reset main data
  data_ = data;

  // remake features
  delete features_;

  // size
  unsigned int nClassifiers = trained_.size();
  if( nClassifiers == 0 ) {
    cerr << "No classifiers have been specified for Combiner." << endl;
    return false;
  }
  assert( nClassifiers == labels_.size() );
  assert( nClassifiers == constraints_.size() );
  assert( nClassifiers == inputDataMappers_.size() );
  assert( nClassifiers == defaultValues_.size() );

  // make copies of everything
  vector<pair<const SprAbsTrainedClassifier*,bool> > trained = trained_;
  vector<string> labels = labels_;
  vector<LocalIndexMap> constraints = constraints_;
  vector<double> defaultValues = defaultValues_; 

  // clean up
  trained_.clear();
  labels_.clear();
  constraints_.clear();
  defaultValues_.clear();
  for( unsigned int i=0;i<inputDataMappers_.size();i++ )
    delete inputDataMappers_[i];
  inputDataMappers_.clear();

  // re-make
  for( unsigned int i=0;i<nClassifiers;i++ ) {
    if( !this->addTrained(trained[i].first,
			  labels[i].c_str(),
			  SprAllowedStringMap(),
			  defaultValues[i],
			  trained[i].second) ) {
      cerr << "Unable to re-insert trained classifier " << i 
	   << " with name " << trained[i].first->name().c_str() << endl;
      return false;
    }
  }

  // mapping from user contraints to trained classifier vars does not change
  constraints_ = constraints;

  // make features
  if( !this->closeClassifierList() ) {
    cerr << "Unable to re-make features." << endl;
    return false;
  }

  // reset
  return this->reset();
}


void SprCombiner::print(std::ostream& os) const
{
  os << "Trained Combiner " << SprVersion << endl;
  os << "Sub-classifiers: " << trained_.size() << endl;

  // loop over sub-classfiers
  for( unsigned int i=0;i<trained_.size();i++ ) {
    os << "Sub-classifier: " << i 
       << " Name: " << labels_[i] 
       << " Default: " << defaultValues_[i] << endl;

    // dump variables
    vector<string> vars;
    trained_[i].first->vars(vars);
    os << "Variables: " << vars.size() << endl;
    for( unsigned int j=0;j<vars.size();j++ )
      os << vars[j].c_str() << " ";
    os << endl;

    // dump mappers
    vector<unsigned> mapper;
    inputDataMappers_[i]->mapper(mapper);
    os << "Mappers: " << mapper.size() << endl;
    for( unsigned int j=0;j<mapper.size();j++ )
      os << mapper[j] << " ";
    os << endl;

    // dump constraints
    os << "Constraints: " << constraints_[i].size() << endl;
    for( LocalIndexMap::const_iterator 
	   iter=constraints_[i].begin();iter!=constraints_[i].end();iter++ ) {
      os << iter->first << " " << iter->second.size() << " ";
      for( unsigned int k=0;k<iter->second.size();k++ )
	os << iter->second[k].first << " " << iter->second[k].second << " ";
      os << endl;
    }

    // dump the sub-classifier
    trained_[i].first->print(os);
  }

  // dump the main classifier
  trainable_->print(os);

  // dump features for trainable classifier
  os << "Features: " << features_->dim() << endl;
  vector<string> fVars;
  features_->vars(fVars);
  for( unsigned int d=0;d<features_->dim();d++ )
    os << fVars[d] << " ";
  os << endl;
}


bool SprCombiner::addTrained(const SprAbsTrainedClassifier* c, 
			     const char* label,
			     const SprAllowedStringMap& stringMap,
			     double defaultValue,
			     bool own)
{
  // sanity check
  if( c==0 || label==0 ) {
    cerr << "Unable to add classifier " << c->name().c_str() << endl;
    return false;
  }

  // get classifier vars
  vector<string> trainedVars;
  c->vars(trainedVars);

  // make sure that each trained classifier variable is present in the data
  vector<string> vars;
  data_->vars(vars);
  SprCoordinateMapper* mapper 
    = SprCoordinateMapper::createMapper(trainedVars,vars);
  if( mapper == 0 ) {
    cerr << "Unable to map trained variables for combiner." << endl;
    return false;
  }

  // make sure that each cut-on variable is present in the variable list
  LocalIndexMap indexMap;
  for( SprAllowedStringMap::const_iterator 
	 iter=stringMap.begin();iter!=stringMap.end();iter++ ) {
    vector<string>::const_iterator found 
      = find(trainedVars.begin(),trainedVars.end(),iter->first);
    if( found == trainedVars.end() ) {
      cerr << "Unable to find variable " << iter->first.c_str() 
	   << " from the input map among variables for trained" 
	   << " classifier " << c->name().c_str() << endl;
      return false;
    }
    int d = found - trainedVars.begin();
    indexMap.insert(pair<unsigned,SprCut>(d,iter->second));
  }

  // add
  trained_.push_back(pair<const SprAbsTrainedClassifier*,bool>(c,own));
  labels_.push_back(label);
  constraints_.push_back(indexMap);
  inputDataMappers_.push_back(mapper);
  defaultValues_.push_back(defaultValue);

  // exit
  return true;
}



bool SprCombiner::makeFeatures()
{
  // size
  unsigned int nClassifiers = trained_.size();
  if( nClassifiers == 0 ) {
    cerr << "No classifiers have been specified for Combiner." << endl;
    return false;
  }
  assert( nClassifiers == labels_.size() );
  assert( nClassifiers == constraints_.size() );
  assert( nClassifiers == inputDataMappers_.size() );
  assert( nClassifiers == defaultValues_.size() );

  // make data
  SprData* features = new SprData("features",labels_);

  // loop over data points
  for( unsigned int ip=0;ip<data_->size();ip++ ) {
    const SprPoint* p = (*data_)[ip];

    // loop over classifiers
    vector<double> resp(nClassifiers);
    for( unsigned int ic=0;ic<nClassifiers;ic++ ) {

      // map this point onto classifier variables
      const SprPoint* pResp 
	= ( inputDataMappers_[ic]==0 ? p : inputDataMappers_[ic]->output(p) );

      // does this point satisfy constraints?
      bool overall = true;
      const LocalIndexMap& indexMap = constraints_[ic];
      for( unsigned d=0;d<pResp->dim();d++ ) {
	double x = pResp->x_[d];
	LocalIndexMap::const_iterator found = indexMap.find(d);
	if( found == indexMap.end() ) continue;
	bool accept = true;
	const SprCut& cut = found->second;
	if( !cut.empty() ) accept = false;
	for( unsigned int k=0;k<cut.size();k++ ) {
	  if( x>cut[k].first && x<cut[k].second ) {
	    accept = true;
	    break;
	  }
	}
	if( !accept ) {
	  overall = false;
	  break;
	}
      }

      // compute classifier response
      if( overall )
	resp[ic] = trained_[ic].first->response(pResp);
      else
	resp[ic] = defaultValues_[ic];

      // clean up the new point
      if( inputDataMappers_[ic] != 0 ) inputDataMappers_[ic]->clear();
    }// end of classifier loop
    
    // insert this point into features
    features->insert(p->class_,resp);
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


bool SprCombiner::closeClassifierList()
{
  return this->makeFeatures();
}


SprTrainedCombiner* SprCombiner::makeTrained() const
{
  // make trained classifier
  SprAbsTrainedClassifier* overall = trainable_->makeTrained();
  bool ownOverall = true;

  // clone sub-classifiers
  vector<pair<const SprAbsTrainedClassifier*,bool> > trained;
  for( unsigned int i=0;i<trained_.size();i++ ) {
    SprAbsTrainedClassifier* c = trained_[i].first->clone();
    trained.push_back(pair<const SprAbsTrainedClassifier*,bool>(c,true));
  }

  // clone coordinate mappers
  vector<SprCoordinateMapper*> inputDataMappers(inputDataMappers_.size());
  for( unsigned int i=0;i<inputDataMappers_.size();i++ )
    inputDataMappers.push_back(inputDataMappers_[i]->clone());

  // make trained combiner
  SprTrainedCombiner* t 
    = new SprTrainedCombiner(overall,trained,labels_,constraints_,
			     inputDataMappers,defaultValues_,ownOverall);

  // vars
  vector<string> vars;
  data_->vars(vars);
  t->setVars(vars);

  // exit
  return t;
}


bool SprCombiner::setClasses(const SprClass& cls0, const SprClass& cls1) 
{
  if( !trainable_->setClasses(cls0,cls1) ) {
    cerr << "Combiner unable to reset classes." << endl;
    return false;
  }
  cout << "Classes for Combiner reset to " << cls0 << " " << cls1 << endl;
  return true;
}
