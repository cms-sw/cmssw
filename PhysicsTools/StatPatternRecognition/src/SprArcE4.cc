//$Id: SprArcE4.cc,v 1.3 2008/11/26 22:59:20 elmer Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprArcE4.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBootstrap.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTrainedClassifier.hh"

#include <cassert>
#include <cmath>
#include <memory>

using namespace std;


SprArcE4::SprArcE4(SprAbsFilter* data, 
		   unsigned cycles, bool discrete)
  : 
  SprBagger(data,cycles,discrete), 
  initialDataWeights_(),
  response_(data->size(),pair<double,double>(0,0))
{ 
  data_->weights(initialDataWeights_);
  cout << "ArcE4 initialized." << endl;
}


bool SprArcE4::setData(SprAbsFilter* data)
{
  // reset base data
  if( !SprBagger::setData(data) ) {
    cerr << "Unable to set data for ArcE4." << endl;
    return false;
  }

  // copy weights
  data_->weights(initialDataWeights_);
  
  // init responses
  response_.clear();
  response_.resize(data_->size(),pair<double,double>(0,0));

  // exit
  return true;
}


bool SprArcE4::train(int verbose)
{
  // sanity check
  if( cycles_==0 || trainable_.empty() ) {
    cout << "ArcE4 will exit without training." << endl;
    return this->prepareExit(true);
  }

  // if resume training, generate a seed from time of day
  if( cycles_>0 && !trained_.empty() ) {
    delete bootstrap_;
    bootstrap_ = new SprBootstrap(data_,-1);
    assert( bootstrap_ != 0 );
  }

  // update responses
  assert( data_->size() == response_.size() );
  for( unsigned int i=0;i<data_->size();i++ ) {
    const SprPoint* p = (*data_)[i];
    for( unsigned int j=0;j<trained_.size();j++ ) {
      double& resp = response_[i].first;
      double& wresp = response_[i].second;
      resp = wresp*resp + trained_[j].first->response(p);
      wresp += 1.;
      resp /= wresp;
    }
  }

  // after all betas are filled, do an overall validation
  if( valData_ != 0 ) {
    // compute cumulative beta weights for validation points
    valBeta_.clear();
    int vsize = valData_->size();
    valBeta_.resize(vsize,0);
    int tsize = trained_.size();
    for( int i=0;i<vsize;i++ ) {
      const SprPoint* p = (*valData_)[i];
      if( discrete_ ) {
	for( int j=0;j<tsize;j++ )
	  valBeta_[i] += ( trained_[j].first->accept(p) ? 1 : -1 );
      }
      else {
	for( int j=0;j<tsize;j++ )
	  valBeta_[i] += trained_[j].first->response(p);
      }
      if( tsize > 0 ) valBeta_[i] /= tsize;
    }

    // print out
    if( valPrint_ > 0 ) {
      if( !this->printValidation(0) ) {
	cerr << "Unable to print out validation data." << endl;
	return this->prepareExit(false);
      }
    }
  }

  // loop through trainable
  unsigned nCycle = 0;
  unsigned nFailed = 0;
  while( nCycle < cycles_ ) {
    for( unsigned int i=0;i<trainable_.size();i++ ) {
      // check cycles
      if( nCycle++ >= cycles_ ) return this->prepareExit((this->nTrained()>0));

      // generate replica
      auto_ptr<SprEmptyFilter> temp(bootstrap_->weightedReplica());
      if( temp->size() != data_->size() ) {
	cerr << "Failed to generate bootstrap replica." << endl;
	return this->prepareExit(false);
      }

      // get new classifier
      SprAbsClassifier* c = trainable_[i];
      if( !c->setData(temp.get()) ) {
	cerr << "Unable to set data for classifier " << i << endl;
	return this->prepareExit(false);
      }
      if( !c->train(verbose) ) {
	cerr << "ArcE4 failed to train classifier " << i 
	     << ". Continuing..."<< endl;
	if( ++nFailed >= cycles_ ) {
	  cout << "Exiting after failed to train " << nFailed 
	       << " classifiers." << endl;
	  return this->prepareExit((this->nTrained()>0));
	}
	else
	  continue;
      }

      // register new trained classifier
      SprAbsTrainedClassifier* t = c->makeTrained();
      if( t == 0 ) {
	cerr << "ArcE4 failed to train classifier " << i 
	     << ". Continuing..."<< endl;
	if( ++nFailed >= cycles_ ) {
	  cout << "Exiting after failed to train " << nFailed 
	       << " classifiers." << endl;
	  return this->prepareExit((this->nTrained()>0));
	}
	else
	  continue;
      }
      trained_.push_back(pair<const SprAbsTrainedClassifier*,bool>(t,true));

      // reweight events
      this->reweight(t);
      if( verbose > 1 ) {
	cout << "After reweighting:   W1=" << data_->weightInClass(cls1_)
	     << " W0=" << data_->weightInClass(cls0_)
	     << "    N1=" << data_->ptsInClass(cls1_)
	     << " N0=" << data_->ptsInClass(cls0_) << endl;
      }

      // message
      if( verbose>1 || (nCycle%100)==0 ) {
	cout << "Done cycle " << nCycle << endl;
      }

      // validation
      if( valData_ != 0 ) {
	// update votes
	int tsize = trained_.size();
	for( unsigned int i=0;i<valData_->size();i++ ) {
	  const SprPoint* p = (*valData_)[i];
	  if( discrete_ ) {
	    if( t->accept(p) ) 
	      valBeta_[i] = ((tsize-1)*valBeta_[i] + 1)/tsize;
	    else
	      valBeta_[i] = ((tsize-1)*valBeta_[i] - 1)/tsize;
	  }
	  else
	    valBeta_[i] = ((tsize-1)*valBeta_[i] + t->response(p))/tsize;
	}

	// print out
	if( valPrint_!=0 && (nCycle%valPrint_)==0 ) {
	  if( !this->printValidation(nCycle) ) {
	    cerr << "Unable to print out validation data." << endl;
	    return this->prepareExit(false);
	  }
	}
      }
    }
  }

  // normal exit
  return this->prepareExit((this->nTrained()>0));
}


bool SprArcE4::prepareExit(bool status)
{
  // restore weights
  data_->setWeights(initialDataWeights_);

  // do basic restore
  return SprBagger::prepareExit(status);
}


void SprArcE4::reweight(const SprAbsTrainedClassifier* t)
{
  unsigned size = data_->size();
  assert( size == initialDataWeights_.size() );
  assert( size == response_.size() );
  for( unsigned int i=0;i<size;i++ ) {
    const SprPoint* p = (*data_)[i];

    // update response
    double& resp = response_[i].first;
    double& wresp = response_[i].second;
    resp = wresp*resp + t->response(p);
    wresp += 1.;
    resp /= wresp;

    // reweight
    int cls = -1;
    if(      p->class_ == cls0_ ) 
      cls = 0;
    else if( p->class_ == cls1_ )
      cls = 1;
    if( cls > -1 ) {
      double error = wresp * (resp - cls);
      double w = initialDataWeights_[i] * (1.+pow(fabs(error),4));
      data_->setW(i,w);
    }
  }
}
