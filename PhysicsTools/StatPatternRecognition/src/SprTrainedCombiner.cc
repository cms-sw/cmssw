//$Id: SprTrainedCombiner.cc,v 1.1 2007/09/21 22:32:10 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedCombiner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprCoordinateMapper.hh"

#include <cassert>

using namespace std;


SprTrainedCombiner::~SprTrainedCombiner()
{
  if( ownOverall_ ) delete overall_;
  for( unsigned int i=0;i<trained_.size();i++ ) {
    if( trained_[i].second ) delete trained_[i].first;
  }
  for( unsigned int i=0;i<inputDataMappers_.size();i++ )
    delete inputDataMappers_[i];
}


SprTrainedCombiner::SprTrainedCombiner(
  const SprAbsTrainedClassifier* overall,
  const std::vector<std::pair<const SprAbsTrainedClassifier*,bool> >& trained, 
  const std::vector<std::string>& labels,
  const std::vector<SprAllowedIndexMap>& constraints,
  const std::vector<SprCoordinateMapper*>& inputDataMappers,
  const std::vector<double>& defaultValues,
  bool ownOverall) 
  :
  overall_(overall),
  trained_(trained),
  labels_(labels),
  constraints_(constraints),
  inputDataMappers_(inputDataMappers),
  defaultValues_(defaultValues),
  ownOverall_(ownOverall)
{
  assert( overall_ != 0 );
  unsigned int nClassifiers = trained_.size();
  assert( nClassifiers == labels_.size() );
  assert( nClassifiers == constraints_.size() );
  assert( nClassifiers == inputDataMappers_.size() );
  assert( nClassifiers == defaultValues_.size() );
  assert( nClassifiers == overall_->dim() );
}


SprTrainedCombiner::SprTrainedCombiner(const SprTrainedCombiner& other)
  :
  overall_(0),
  trained_(),
  labels_(other.labels_),
  constraints_(other.constraints_),
  inputDataMappers_(),
  defaultValues_(other.defaultValues_),
  ownOverall_(false)
{
  overall_ = other.overall_->clone();
  ownOverall_ = true;
  for( unsigned int i=0;i<other.trained_.size();i++ )
    trained_.push_back(pair<const SprAbsTrainedClassifier*,bool>
                       (other.trained_[i].first->clone(),true));
  for( unsigned int i=0;i<other.inputDataMappers_.size();i++ )
    inputDataMappers_.push_back(other.inputDataMappers_[i]->clone());
  unsigned int nClassifiers = trained_.size();
  assert( nClassifiers == labels_.size() );
  assert( nClassifiers == constraints_.size() );
  assert( nClassifiers == inputDataMappers_.size() );
  assert( nClassifiers == defaultValues_.size() );
  assert( nClassifiers == overall_->dim() );
}


double SprTrainedCombiner::response(const std::vector<double>& v) const
{
  // loop over classifiers
  unsigned int nClassifiers = trained_.size();
  vector<double> cResp(nClassifiers);
  vector<double>* casted = 0;
  for( unsigned int ic=0;ic<nClassifiers;ic++ ) {

    // map this point onto classifier variables
    vector<double>* vResp = 0;
    bool deleteV = false;
    if( inputDataMappers_[ic] == 0 ) {
      if( casted == 0 ) 
	casted = const_cast<vector<double>*>(&v);
      vResp = casted;
    }
    else {
      vResp = new vector<double>();
      deleteV = true;
      inputDataMappers_[ic]->map(v,*vResp);
    }

    // does this point satisfy constraints?
    bool overall = true;
    const SprAllowedIndexMap& indexMap = constraints_[ic];
    for( unsigned int d=0;d<vResp->size();d++ ) {
      double x = (*vResp)[d];
      SprAllowedIndexMap::const_iterator found = indexMap.find(d);
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
      cResp[ic] = trained_[ic].first->response(*vResp);
    else
      cResp[ic] = defaultValues_[ic];

    // clean up
    if( deleteV ) delete vResp;
  }

  // apply overall classifier to the computed vector
  return overall_->response(cResp);
}


void SprTrainedCombiner::print(std::ostream& os) const
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
    for( SprAllowedIndexMap::const_iterator 
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
  overall_->print(os);

  // dump features for overall classifier
  vector<string> fVars;
  overall_->vars(fVars);
  os << "Features: " << fVars.size() << endl;
  for( unsigned int d=0;d<fVars.size();d++ )
    os << fVars[d] << " ";
  os << endl;
}
