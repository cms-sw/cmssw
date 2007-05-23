//$Id: SprTrainedMultiClassLearner.cc,v 1.4 2006/11/26 02:04:31 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedMultiClassLearner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"

#include <algorithm>
#include <functional>
#include <iomanip>
#include <cassert>

using namespace std;


struct STMCLCmpPairSecond
  : public binary_function<pair<const int,double>,
			   pair<const int,double>,bool> {
  bool operator()(const pair<const int,double>& l, 
		  const pair<const int,double>& r) const {
    return (l.second < r.second);
  }
};


SprTrainedMultiClassLearner::SprTrainedMultiClassLearner(
       const SprMatrix& indicator,
       const std::vector<int>& mapper,
       const std::vector<std::pair<const SprAbsTrainedClassifier*,bool> >& 
       classifiers)
  :
  indicator_(indicator),
  mapper_(mapper),
  classifiers_(classifiers),
  loss_(0),
  trans_(0)
{
  assert( !classifiers_.empty() );
  assert( indicator_.num_row() > 0 );
  assert( indicator_.num_col() == classifiers_.size() );
  if( mapper_.empty() ) {
    unsigned n = indicator_.num_row();
    mapper_.resize(n);
    for( int i=0;i<n;i++ ) mapper_[i] = i;
  }
  assert( mapper_.size() == indicator_.num_row() );
}


SprTrainedMultiClassLearner::SprTrainedMultiClassLearner(
                             const SprTrainedMultiClassLearner& other)
  :
  indicator_(other.indicator_),
  mapper_(other.mapper_),
  classifiers_(),
  loss_(other.loss_),
  trans_(other.trans_)
{
  for( int i=0;i<other.classifiers_.size();i++ ) {
    const SprAbsTrainedClassifier* t = other.classifiers_[i].first->clone();
    classifiers_.push_back(pair<const SprAbsTrainedClassifier*,bool>(t,true));
  }
  assert( indicator_.num_col() == classifiers_.size() );
}


void SprTrainedMultiClassLearner::destroy()
{
  for( int i=0;i<classifiers_.size();i++ ) {
    if( classifiers_[i].second )
      delete classifiers_[i].first;
  }
}
 

int SprTrainedMultiClassLearner::response(const std::vector<double>& input,
					  std::map<int,double>& output) const
{
  // sanity check
  assert( loss_ != 0 );

  // compute vector of responses
  vector<double> response(classifiers_.size());
  for( int i=0;i<classifiers_.size();i++ ) {
    double r = classifiers_[i].first->response(input);
    response[i] = ( trans_==0 ? r : trans_(r) );
  }

  // evaluate consistency with each row
  output.clear();
  unsigned ncol = indicator_.num_col();
  for( int i=0;i<indicator_.num_row();i++ ) {
    double rowLoss = 0;
    for( int j=0;j<ncol;j++ )
      rowLoss += loss_(int(indicator_[i][j]),response[j]);
    rowLoss /= ncol;
    output.insert(pair<const int,double>(mapper_[i],rowLoss));
  }

  // find minimal loss
  map<int,double>::const_iterator iter 
    = min_element(output.begin(),output.end(),STMCLCmpPairSecond());
  return iter->first;
}


void SprTrainedMultiClassLearner::print(std::ostream& os) const 
{
  os << "Trained Multi Class Learner" << endl;

  // print matrix
  this->printIndicatorMatrix(os);

  // print classifiers
  assert( indicator_.num_col() == classifiers_.size() );
  for( int j=0;j<classifiers_.size();j++ ) {
    os << "Multi class learner subclassifier: " << j << endl;
    classifiers_[j].first->print(os);
  }
}


void SprTrainedMultiClassLearner::classes(std::vector<int>& classes) const 
{ 
  classes = mapper_; 
  stable_sort(classes.begin(),classes.end());
}


void SprTrainedMultiClassLearner::printIndicatorMatrix(std::ostream& os) const
{
  os << "Indicator matrix:" << endl;
  os << setw(20) << "Classes/Classifiers" << " : " 
     << mapper_.size() << " " << classifiers_.size() << endl;
  os << "=========================================================" << endl;
  for( int i=0;i<indicator_.num_row();i++ ) {
    os << setw(20) << mapper_[i] << " : ";
    for( int j=0;j<indicator_.num_col();j++ ) 
      os << setw(2) << indicator_[i][j] << " ";
    os << endl;
  }
  os << "=========================================================" << endl;
}
