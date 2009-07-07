//$Id: SprBagger.cc,v 1.4 2008/11/26 22:59:20 elmer Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBagger.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBootstrap.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsTwoClassCriterion.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAverageLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"

#include <stdio.h>
#include <functional>
#include <algorithm>
#include <cmath>
#include <memory>

using namespace std;


struct SBTrainedOwned 
  : public unary_function<pair<const SprAbsTrainedClassifier*,bool>,bool> {
  bool operator()(const pair<const SprAbsTrainedClassifier*,bool>& p) const {
    return p.second;
  }
};


SprBagger::~SprBagger() 
{ 
  this->destroy(); 
  if( ownLoss_ ) {
    delete loss_;
    loss_ = 0;
    ownLoss_ = false;
  }
}


SprBagger::SprBagger(SprAbsFilter* data)
  : 
  SprAbsClassifier(data), 
  crit_(0),
  cls0_(0),
  cls1_(1),
  cycles_(0),
  discrete_(false),
  trained_(), 
  trainable_(), 
  bootstrap_(0),
  valData_(0),
  valBeta_(),
  valPrint_(0),
  loss_(0),
  ownLoss_(false)
{}


SprBagger::SprBagger(SprAbsFilter* data, unsigned cycles, bool discrete)
  : 
  SprAbsClassifier(data), 
  crit_(0),
  cls0_(0),
  cls1_(1),
  cycles_(cycles),
  discrete_(discrete),
  trained_(), 
  trainable_(), 
  bootstrap_(new SprBootstrap(data)),
  valData_(0),
  valBeta_(),
  valPrint_(0),
  loss_(0),
  ownLoss_(false)
{ 
  assert( bootstrap_ != 0 );
  this->setClasses();
  cout << "Bagger initialized with classes " << cls0_ << " " << cls1_
       << " with cycles " << cycles_ << endl;
}


void SprBagger::destroy()
{
  for( unsigned int i=0;i<trained_.size();i++ ) {
    if( trained_[i].second )
      delete trained_[i].first;
  }
  trained_.erase(remove_if(trained_.begin(),trained_.end(),SBTrainedOwned()),
		 trained_.end());
  delete bootstrap_;
  bootstrap_ = 0;
}


void SprBagger::setClasses() 
{
  vector<SprClass> classes;
  data_->classes(classes);
  int size = classes.size();
  if( size > 0 ) cls0_ = classes[0];
  if( size > 1 ) cls1_ = classes[1];
  cout << "Classes for Bagger are set to " << cls0_ << " " << cls1_ << endl;
}


bool SprBagger::reset() 
{
  this->destroy();
  bootstrap_ = new SprBootstrap(data_,-1);
  return true;
}


bool SprBagger::setData(SprAbsFilter* data)
{
  assert( data != 0 );

  // reset base data
  data_ = data;

  // reset data supplied to trainable classifiers
  for( unsigned int i=0;i<trainable_.size();i++ ) {
    if( !trainable_[i]->setData(data_) ) {
      cerr << "Cannot reset data for trainable classifier " << i << endl;
      return false;
    }
  }

  // basic reset
  return this->reset();
}


bool SprBagger::addTrained(const SprAbsTrainedClassifier* c, bool own)
{
  if( c == 0 ) return false;
  trained_.push_back(pair<const SprAbsTrainedClassifier*,bool>(c,own));
  return true;
}


bool SprBagger::addTrainable(SprAbsClassifier* c)
{
  if( c == 0 ) return false;
  trainable_.push_back(c);
  return true;
}


bool SprBagger::train(int verbose)
{
  // sanity check
  if( cycles_==0 || trainable_.empty() ) {
    cout << "Bagger will exit without training." << endl;
    return this->prepareExit(true);
  }

  // if resume training, generate a seed from time of day
  if( cycles_>0 && !trained_.empty() ) {
    delete bootstrap_;
    bootstrap_ = new SprBootstrap(data_,-1);
    assert( bootstrap_ != 0 );
  }

  // after all betas are filled, do an overall validation
  if( valData_ != 0 ) {
    // compute cumulative beta weights for validation points
    valBeta_.clear();
    unsigned int vsize = valData_->size();
    valBeta_.resize(vsize,0);
    unsigned int tsize = trained_.size();
    for( unsigned int i=0;i<vsize;i++ ) {
      const SprPoint* p = (*valData_)[i];
      if( discrete_ ) {
	for( unsigned int j=0;j<tsize;j++ )
	  valBeta_[i] += ( trained_[j].first->accept(p) ? 1 : -1 );
      }
      else {
	for( unsigned int j=0;j<tsize;j++ )
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
      auto_ptr<SprEmptyFilter> temp(bootstrap_->plainReplica());
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
	cerr << "Bagger failed to train classifier " << i 
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
	cerr << "Bagger failed to train classifier " << i 
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

      // message
      if( verbose>0 || (nCycle%100)==0 ) {
	cout << "Finished cycle " << nCycle 
	     << " with classifier " << t->name().c_str() << endl;
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


SprTrainedBagger* SprBagger::makeTrained() const
{
  // sanity check
  if( trained_.empty() ) return 0;

  // prepare a vector of trained classifiers
  vector<pair<const SprAbsTrainedClassifier*,bool> > trained;
  for( unsigned int i=0;i<trained_.size();i++ ) {
    SprAbsTrainedClassifier* c = trained_[i].first->clone();
    trained.push_back(pair<const SprAbsTrainedClassifier*,bool>(c,true));
  }

  // make a trained bagger
  SprTrainedBagger* t = new SprTrainedBagger(trained,discrete_);

  // vars
  vector<string> vars;
  data_->vars(vars);
  t->setVars(vars);

  // exit
  return t;
}


void SprBagger::print(std::ostream& os) const
{
  os << "Trained Bagger " << SprVersion << endl;
  os << "Classifiers: " << trained_.size() << endl;
  for( unsigned int i=0;i<trained_.size();i++ ) {
    os << "Classifier " << i 
       << " " << trained_[i].first->name().c_str() << endl;
    trained_[i].first->print(os);
  }
}


bool SprBagger::setValidation(const SprAbsFilter* valData, 
			      unsigned valPrint,
			      const SprAbsTwoClassCriterion* crit,
			      SprAverageLoss* loss) 
{
  // sanity checks
  if( !valBeta_.empty() ) {
    cerr << "One cannot reset validation data after training has started." 
	 << endl;
    return false;
  }
  assert( valData != 0 );

  // set 
  valData_ = valData;
  valPrint_ = valPrint;
  crit_ = crit;
  loss_ = loss;

  // make default loss if none supplied
  if( crit_==0 && loss_==0 ) {
    loss_ = new SprAverageLoss(&SprLoss::quadratic);
    ownLoss_ = true;
  }

  // check loss and discreteness
  if( loss_==0 && !discrete_ ) {
    cout << "Warning: you requested continuous output for validation,"
	 << " yet you have not supplied average loss appropriate for "
	 << "the continuous output. Do you know what you are doing?" << endl;
  }

  // exit
  return true;
}


bool SprBagger::printValidation(unsigned cycle)
{
  // no print-out for zero training cycle
  if( cycle == 0 ) return true;

  // sanity check
  assert(valBeta_.size() == valData_->size());

  // reset loss
  if( loss_ != 0 ) loss_->reset();

  // loop through validation data
  unsigned int vsize = valData_->size();
  double wcor0(0), wcor1(0), wmis0(0), wmis1(0);
  for( unsigned int i=0;i<vsize;i++ ) {
    const SprPoint* p = (*valData_)[i];
    double w = valData_->w(i);
    if( p->class_!=cls0_ && p->class_!=cls1_ ) w = 0;
    if( loss_ == 0 ) {
      if( valBeta_[i] > 0 ) {
	if(      p->class_ == cls0_ )
	  wmis0 += w;
	else if( p->class_ == cls1_ )
	  wcor1 += w;
      }
      else {
	if(      p->class_ == cls0_ )
	  wcor0 += w;
	else if( p->class_ == cls1_ )
	  wmis1 += w;
      }
    }
    else {
      if(      p->class_ == cls0_ )
	loss_->update(0,valBeta_[i],w);
      else if( p->class_ == cls1_ )
	loss_->update(1,valBeta_[i],w);
    }
  }

  // compute fom
  double fom = 0;
  assert( crit_!=0 || loss_!=0 );
  if( loss_ == 0 )
    fom = crit_->fom(wcor0,wmis0,wcor1,wmis1);
  else
    fom = loss_->value();
  cout << "Validation FOM=" << fom << " at cycle " << cycle << endl;

  // exit
  return true;
}


bool SprBagger::prepareExit(bool status)
{
  // restore the original data supplied to the classifiers
  for( unsigned int i=0;i<trainable_.size();i++ ) {
    SprAbsClassifier* c = trainable_[i];
    if( !c->setData(data_) )
      cerr << "Unable to restore original data for classifier " << i << endl;
  }
   
  // exit
  return status;
}


bool SprBagger::setClasses(const SprClass& cls0, const SprClass& cls1) 
{
  for( unsigned int i=0;i<trainable_.size();i++ ) {
    if( !trainable_[i]->setClasses(cls0,cls1) ) {
      cerr << "Bagger unable to reset classes for classifier " << i << endl;
      return false;
    }
  }
  cls0_ = cls0; cls1_ = cls1;
  cout << "Classes for Bagger reset to " << cls0_ << " " << cls1_ << endl;
  return true;
}


bool SprBagger::initBootstrapFromTimeOfDay()
{
  if( bootstrap_ == 0 ) {
    cerr << "No bootstrap object found for the Bagger." << endl;
    return false;
  }
  bootstrap_->init(-1);
  return true;
}
