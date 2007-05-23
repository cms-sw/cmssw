//$Id: SprDataFeeder.cc,v 1.3 2006/11/13 19:09:41 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDataFeeder.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsWriter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedMultiClassLearner.hh"

#include <stdio.h>
#include <iostream>
#include <cassert>
#include <string>

using namespace std;


class CoordinateMapper
{
public:
  ~CoordinateMapper() { this->clear(); }

  CoordinateMapper(const std::vector<unsigned>& mapper)
    : mapper_(mapper), toDelete_() {}

  // map vectors
  const SprPoint* output(const SprPoint* input) {
    // sanity check
    if( mapper_.empty() ) return input;

    // make new point and copy index+class
    SprPoint* p = new SprPoint;
    p->index_ = input->index_;
    p->class_ = input->class_;

    // copy vector elements
    for( int i=0;i<mapper_.size();i++ ) {
      unsigned d = mapper_[i];
      assert( d < input->dim() );
      p->x_.push_back(input->x_[d]);
    }

    // add to the cleanup list
    toDelete_.push_back(p);

    // exit
    return p;
  }

  // clean up
  void clear() {
    for( int i=0;i<toDelete_.size();i++ ) delete toDelete_[i];
    toDelete_.clear();
  }

private:
  vector<unsigned> mapper_;
  vector<const SprPoint*> toDelete_;
};


SprDataFeeder::~SprDataFeeder()
{
  delete mapper_;
}


SprDataFeeder::SprDataFeeder(const SprAbsFilter* data,
			     SprAbsWriter* writer,
			     const std::vector<unsigned>& mapper) 
  : 
  data_(data), 
  writer_(writer), 
  mode_(0),
  classifiers_(),
  multiclass_(),
  mapper_(new CoordinateMapper(mapper))
{
  assert( data_ != 0 );
  assert( writer_ != 0 );
  vector<string> axes;
  data_->vars(axes);
  writer_->setAxes(axes);
}


bool SprDataFeeder::addClassifier(const SprAbsTrainedClassifier* c,
				  const char* name) 
{ 
  if( c != 0 ) {
    // sanity check
    if( mode_ == 2 ) {
      cerr << "Unable to add classifier: " 
	   << "DataFeeder is in the multi class mode." << endl;
      return false;
    }
    else
      mode_ = 1;

    // add classifier
    classifiers_.push_back(c);
    writer_->addAxis(name);
  }

  // exit
  return true;
}


bool SprDataFeeder::addMultiClassLearner(const SprTrainedMultiClassLearner* c,
					 const char* name) 
{ 
  if( c != 0 ) {
    // sanity check
    if( mode_ == 1 ) {
      cerr << "Unable to add multi class learner: " 
	   << "DataFeeder is in the regular classifier mode." << endl;
      return false;
    }
    else
      mode_ = 2;

    // add multi class learner
    multiclass_.push_back(c);
    vector<int> classes;
    c->classes(classes);
    for( int i=0;i<classes.size();i++ ) {
      string axis = name;
      char s [200];
      sprintf(s,"%i",classes[i]);
      axis += s;
      writer_->addAxis(axis.c_str());
    }
    writer_->addAxis(name);
  }

  // exit
  return true;
}


bool SprDataFeeder::feed(int nout) const
{
  // sanity checks
  assert( data_ != 0 );
  assert( writer_ != 0 );
  if( classifiers_.empty() && multiclass_.empty() ) {
    cout << "Warning: no classifiers specified for SprDataFeeder. " 
	 << "Will save data only." << endl;
  }

  // loop through data
  vector<double> f;
  for( int i=0;i<data_->size();i++ ) {
    // message
    if( nout>0 && (i%nout)==0 ) {
      cout << "Feeder storing point " << i 
	   << " out of " << data_->size() << endl;
    }

    // get point
    const SprPoint* pTuple = (*data_)[i];
    const SprPoint* pResp = mapper_->output(pTuple);
    f.clear();

    // add classifiers
    for( int j=0;j<classifiers_.size();j++ )
      f.push_back(classifiers_[j]->response(pResp));

    // add multi class learners
    for( int j=0;j<multiclass_.size();j++ ) {
      map<int,double> resp;
      int cls = multiclass_[j]->response(pResp,resp);
      for( map<int,double>::const_iterator 
	     iter=resp.begin();iter!=resp.end();iter++ ) 
	f.push_back(iter->second);
      f.push_back(double(cls));
    }

    // write
    writer_->write(data_->w(i),pTuple,f);

    // clean up
    mapper_->clear();
  }

  // exit
  bool closed = writer_->close();
  if( closed )
    cout << "Writer successfully closed." << endl;
  else
    cout << "Writer was unable to close successfully." << endl;
  return closed;
}
