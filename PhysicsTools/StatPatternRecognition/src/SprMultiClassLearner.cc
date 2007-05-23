//$Id: SprMultiClassLearner.cc,v 1.3 2006/11/13 19:09:43 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprMultiClassLearner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedMultiClassLearner.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"

#include <fstream>
#include <string>
#include <iomanip>

using namespace std;


SprMultiClassLearner::~SprMultiClassLearner()
{
  this->destroy();
}


SprMultiClassLearner::SprMultiClassLearner(SprAbsFilter* data, 
					   SprAbsClassifier* c,
					   const std::vector<int>& classes,
					   const SprMatrix& indicator,
					   MultiClassMode mode)
  :
  data_(data),
  mode_(mode),
  indicator_(indicator),
  mapper_(classes),
  trainable_(c),
  trained_()
{
  assert( data_ != 0 );
  assert( trainable_ != 0 );
  assert( mode_!=User || indicator_.num_col()!=0 );
  bool status = this->setClasses();
  assert( status );
}


bool SprMultiClassLearner::setClasses()
{
  // sanity check
  if( mapper_.size() < 2 ) {
    cerr << "Less than 2 classes are specified." << endl;
    return false;
  }

  // set up classes for the trainable classifier
  if( !trainable_->setClasses(0,1) ) {
    cerr << "Unable to set up classes for the trainable classifier." << endl;
    return false;
  }

  // make sure all classes are distinct
  for( int i=0;i<mapper_.size();i++ ) {
    for( int j=i+1;j<mapper_.size();j++ ) {
      if( mapper_[i] == mapper_[j] ) {
	cerr << "Elements " << i << " and " << j 
	     << " of the input vector of classes are equal." << endl;
	return false;
      }
    }
  }

  // check indicator matrix
  if( mode_ == User ) {
    if( indicator_.num_row() != mapper_.size() ) {
      cerr << "Number of rows of the indicator matrix is not equal " 
	   << "to the specified number of classes."<< endl;
      return false;
    }
  }

  // fill out indicator matrix
  if(      mode_ == OneVsAll ) {
    unsigned n = mapper_.size();
    SprMatrix mat(n,n,0);
    indicator_ = mat;
    for( int i=0;i<n;i++ ) { 
      for( int j=0;j<n;j++ ) indicator_[i][j] = -1;
    }
    for( int i=0;i<n;i++ ) indicator_[i][i] = 1;
  }
  else if( mode_ == OneVsOne ) {
    unsigned n = mapper_.size();
    unsigned m = n*(n-1)/2;
    SprMatrix mat(n,m,0);
    indicator_ = mat;
    int jstart = 0;
    int jend = 0;
    for( int i=0;i<n;i++ ) {
      jstart = jend;
      jend += n-1-i;
      for( int j=jstart;j<jend;j++ ) indicator_[i][j] = 1;
      int j = jstart;
      for( int k=i+1;k<n;k++ ) indicator_[k][j++] = -1;
    }
  }

  // show matrix
  this->printIndicatorMatrix(cout);

  // exit
  return true;
}


bool SprMultiClassLearner::train(int verbose)
{
  // reset
  trained_.clear();
  trained_.resize(indicator_.num_col());

  // build a map of classes
  map<int,unsigned> mapper;
  for( int i=0;i<mapper_.size();i++ )
    mapper.insert(pair<const int,unsigned>(mapper_[i],i));

  // loop thru columns of the indicator matrix
  unsigned size = data_->size();
  vector<int> classes(size);
  vector<double> weights;
  for( int j=0;j<indicator_.num_col();j++ ) {
    // memorize classes and weights
    for( int i=0;i<size;i++ )
      classes[i] = (*data_)[i]->class_;
    data_->weights(weights);

    // adjust classes according to the indicator matrix
    for( int i=0;i<size;i++ ) {
      map<int,unsigned>::const_iterator iter 
	= mapper.find((*data_)[i]->class_);
      if( iter == mapper.end() ) {
	(*data_)[i]->class_ = -1;
	data_->setW(i,0);
	continue;
      }
      double flag = indicator_[iter->second][j];
      if(      flag < -0.5 )
	(*data_)[i]->class_ = 0;
      else if( flag > 0.5 )
	(*data_)[i]->class_ = 1;
      else {
	(*data_)[i]->class_ = -1;
	data_->setW(i,0);
      }
    }

    // message
    double w0 = data_->weightInClass(0);
    double w1 = data_->weightInClass(1);
    unsigned n0 = data_->ptsInClass(0);
    unsigned n1 = data_->ptsInClass(1);
    cout << "Training classifier for matrix column " << j << " with "
	 << "     W0=" << w0 << " W1=" << w1
	 << "     N0=" << n0 << " N1=" << n1 << endl;

    // apply a classifier
    if( !trainable_->setData(data_) ) {
      cerr << "Unable to reset classifier for indicator column " << j << endl;
      return false;
    }
    if( !trainable_->train(verbose) ) {
      cerr << "Unable to train classifier for indicator column " << j << endl;
      return false;
    }

    // make trained classifier
    const SprAbsTrainedClassifier* t = trainable_->makeTrained();
    if( t == 0 ) {
      cerr << "Unable to train classifier for column " << j << endl;
      return false;
    }
    trained_[j] = pair<const SprAbsTrainedClassifier*,bool>(t,true);

    // reset classes and weights
    for( int i=0;i<size;i++ )
      (*data_)[i]->class_ = classes[i];
    data_->setWeights(weights);
  }

  // exit
  return true;
}


bool SprMultiClassLearner::reset()
{
  if( !trainable_->reset() ) {
    cerr << "Unable to reset trainable classifier." << endl;
    return false;
  }
  this->destroy();
  return true;
}


void SprMultiClassLearner::destroy()
{
  for( int i=0;i<trained_.size();i++ )
    if( trained_[i].second )
      delete trained_[i].first;
  trained_.clear();
}


bool SprMultiClassLearner::setData(SprAbsFilter* data)
{
  if( !trainable_->setData(data) ) {
    cerr << "Unable to set data for trainable classifier." << endl;
    return false;
  }
  this->destroy();
  return true;
}


void SprMultiClassLearner::print(std::ostream& os) const 
{
  os << "Trained Multi Class Learner" << endl;

  // print matrix
  this->printIndicatorMatrix(os);

  // print classifiers
  for( int i=0;i<trained_.size();i++ ) {
    os << "Multi class learner classifier: " << i << endl;
    trained_[i].first->print(os);
  }

  // print variables
  vector<string> vars;
  data_->vars(vars);
  assert( vars.size() == data_->dim() );
  os << "==================================================" << endl;
  os << "Dimensions:" << endl;
  for( int i=0;i<vars.size();i++ ) {
    char s [200];
    sprintf(s,"%5i %40s",i,vars[i].c_str());
    os << s << endl;
  }
  os << "==================================================" << endl;
}


bool SprMultiClassLearner::store(const char* filename) const
{
  // open file for output
  string fname = filename;
  ofstream fout(fname.c_str());
  if( !fout ) {
    cerr << "Unable to open file " << fname.c_str() << endl;
    return false;
  }
  
  // store into file
  this->print(fout);
 
  // exit
  return true;
}


SprTrainedMultiClassLearner* SprMultiClassLearner::makeTrained() const
{
  vector<pair<const SprAbsTrainedClassifier*,bool> > trained(trained_.size());
  for( int i=0;i<trained_.size();i++ ) {
    trained[i] 
      = pair<const SprAbsTrainedClassifier*,bool>(trained_[i].first->clone(),
						  true);
  }
  return new SprTrainedMultiClassLearner(indicator_,mapper_,trained);
}


void SprMultiClassLearner::printIndicatorMatrix(std::ostream& os) const
{
  os << "Indicator matrix:" << endl;
  os << setw(20) << "Classes/Classifiers" << " : " 
     << mapper_.size() << " " << indicator_.num_col() << endl;
  os << "=========================================================" << endl;
  for( int i=0;i<indicator_.num_row();i++ ) {
    os << setw(20) << mapper_[i] << " : ";
    for( int j=0;j<indicator_.num_col();j++ ) 
      os << setw(2) << indicator_[i][j] << " ";
    os << endl;
  }
  os << "=========================================================" << endl;
}


void SprMultiClassLearner::setTrained(const SprMatrix& indicator, 
				      const std::vector<int>& classes,
				      const std::vector<std::pair<
				      const SprAbsTrainedClassifier*,bool> >& 
				      trained) 
{
  indicator_ = indicator;
  mapper_ = classes;
  trained_ = trained;
  assert( mapper_.size() == indicator_.num_row() );
  assert( trained_.size() == indicator_.num_col() );
  assert( !mapper_.empty() );
  assert( !trained_.empty() );
  mode_ = User;
}
