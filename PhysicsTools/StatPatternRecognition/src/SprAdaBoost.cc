//$Id: SprAdaBoost.cc,v 1.3 2008/11/26 22:59:20 elmer Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAdaBoost.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprBootstrap.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAverageLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprLoss.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformation.hh"

#include <stdio.h>
#include <functional>
#include <algorithm>
#include <cmath>
#include <memory>

using namespace std;


struct SABTrainedOwned 
  : public unary_function<pair<const SprAbsTrainedClassifier*,bool>,bool> {
  bool operator()(const pair<const SprAbsTrainedClassifier*,bool>& p) const {
    return p.second;
  }
};


struct SABCmpPairDDFirst 
  : public binary_function<pair<double,double>,pair<double,double>,bool> {
  bool operator()(const pair<double,double>& l, const pair<double,double>& r)
    const {
    return (l.first < r.first);
  }
};


struct SABCmpPairDIFirst 
  : public binary_function<pair<double,int>,pair<double,int>,bool> {
  bool operator()(const pair<double,int>& l, const pair<double,int>& r)
    const {
    return (l.first < r.first);
  }
};


struct SABCmpPairDDFirstNumber
  : public binary_function<pair<double,double>,double,bool> {
  bool operator()(const pair<double,double>& l, double r) const {
    return (l.first < r);
  }
};
    

SprAdaBoost::~SprAdaBoost() 
{ 
  this->destroy(); 
  delete bootstrap_;
  if( ownLoss_ ) {
    delete loss_;
    loss_ = 0;
    ownLoss_ = false;
  }
}


SprAdaBoost::SprAdaBoost(SprAbsFilter* data)
  :
  SprAbsClassifier(data), 
  cls0_(0),
  cls1_(1),
  cycles_(0),
  trained_(), 
  trainable_(), 
  beta_(),
  epsilon_(0.01),
  valData_(0),
  valBeta_(),
  valPrint_(0),
  initialDataWeights_(),
  trainedDataWeights_(),
  skipReweighting_(false),
  useStandard_(false),
  mode_(SprTrainedAdaBoost::Discrete),
  bootstrap_(0),
  loss_(0),
  ownLoss_(false)
{}


SprAdaBoost::SprAdaBoost(SprAbsFilter* data, 
			 unsigned cycles,
			 bool useStandard,
			 SprTrainedAdaBoost::AdaBoostMode mode,
			 bool bagInput)
  : 
  SprAbsClassifier(data), 
  cls0_(0),
  cls1_(1),
  cycles_(cycles),
  trained_(), 
  trainable_(), 
  beta_(),
  epsilon_(0.01),
  valData_(0),
  valBeta_(),
  valPrint_(0),
  initialDataWeights_(),
  trainedDataWeights_(),
  skipReweighting_(false),
  useStandard_(useStandard),
  mode_(mode),
  bootstrap_(0),
  loss_(0),
  ownLoss_(false)
{ 
  this->setClasses();
  cout << "AdaBoost initialized with classes " << cls0_ << " " << cls1_
       << " with cycles " << cycles_ << endl;
  if( bagInput ) {
    bootstrap_ = new SprBootstrap(data_);
    cout << "AdaBoost will resample training points." << endl;
  }
}


bool SprAdaBoost::reset()
{
  this->destroy();
  if( bootstrap_ != 0 ) {
    delete bootstrap_;
    bootstrap_ = new SprBootstrap(data_,-1);
  }
  return true;
}


void SprAdaBoost::destroy()
{
  for( unsigned int i=0;i<trained_.size();i++ ) {
    if( trained_[i].second )
      delete trained_[i].first;
  }
  trained_.erase(remove_if(trained_.begin(),trained_.end(),SABTrainedOwned()),
		 trained_.end());
  beta_.clear();
}


void SprAdaBoost::setClasses() 
{
  vector<SprClass> classes;
  data_->classes(classes);
  int size = classes.size();
  if( size > 0 ) cls0_ = classes[0];
  if( size > 1 ) cls1_ = classes[1];
  cout << "Classes for AdaBoost are set to " << cls0_ << " " << cls1_ << endl;
}


bool SprAdaBoost::addTrained(const SprAbsTrainedClassifier* c, bool own)
{
  if( c == 0 ) return false;
  trained_.push_back(pair<const SprAbsTrainedClassifier*,bool>(c,own));
  return true;
}


bool SprAdaBoost::addTrainable(SprAbsClassifier* c, const SprCut& cut)
{
  if( c == 0 ) return false;
  trainable_.push_back(SprCCPair(c,cut));
  return true;
}


bool SprAdaBoost::setData(SprAbsFilter* data)
{
  assert( data != 0 );
  data_ = data;
  for( unsigned int i=0;i<trainable_.size();i++ ) {
    if( !trainable_[i].first->setData(data) ) {
      cerr << "Cannot reset data for trainable classifier " << i << endl;
      return false;
    }
  }
  return this->reset();
}

/*
  -1 - error greater than 1/2; this training cycle should be skipped
   0 - normal exit
  +1 - perfect separation achieved; error = 0
  +2 - abnormal termination
*/
int SprAdaBoost::reweight(const SprAbsTrainedClassifier* c, double& beta, 
			  bool useInputBeta, int verbose)
{
  // sanity check
  assert( c != 0 );

  // message
  if( verbose > 3 ) {
    cout << "AdaBoost processing trained classifier:" << endl;
    c->print(cout);
  }

  // init
  unsigned int size = data_->size();
  double error = 0;

  // reweight events
  if(      mode_ == SprTrainedAdaBoost::Discrete ) {

    // loop through training data and classify points
    unsigned ncor0(0), ncor1(0), nmis0(0), nmis1(0);
    double wmis(0), wcor(0);
    vector<int> out(size);
    for( unsigned int i=0;i<size;i++ ) {
      const SprPoint* p = (*data_)[i];
      double w = data_->w(i);
      if( c->accept(p) ) {
	if(      p->class_ == cls0_ ) {
	  out[i] = 0;
	  wmis += w;
	  nmis0++;
	}
	else if( p->class_ == cls1_ ) {
	  out[i] = 1;
	  wcor += w;
	  ncor1++;
	}
      }
      else {
	if(      p->class_ == cls0_ ) {
	  out[i] = 1;
	  wcor += w;
	  ncor0++;
	}
	else if( p->class_ == cls1_ ) {
	  out[i] = 0;
	  wmis += w;
	  nmis1++;
	}
      }
    }
    
    // compute weighted fraction of misclassified points
    assert( (wcor+wmis) > 0 );
    if( useInputBeta )
      error = SprTransformation::logitDouble(-beta);
    else
      error = wmis/(wcor+wmis);
    if( verbose > 2 ) {
      cout << "Fraction of misclassified events by AdaBoost = " 
	   << error << endl;
      cout << "Counts in categories (correct n0/n1; incorrect n0/n1) "
	   << ncor0 << "/" << ncor1 << " "
	   << nmis0 << "/" << nmis1 << endl;
    }
    double eps = SprUtils::eps();
    if( error > (0.5-eps) ) {
      beta = -1;
      return -1;
    }
    if( error < eps ) {
      beta = SprUtils::max();
      return 1;
    }
    
    // adjust weights
    for( unsigned int i=0;i<size;i++ ) {
      if(      out[i] == 0 )
	data_->uncheckedSetW(i,0.5*(data_->w(i))/error);
      else if( out[i] == 1 )
	data_->uncheckedSetW(i,0.5*(data_->w(i))/(1.-error));
    }

    // return beta coefficient (positive because error<=0.5)
    beta = SprTransformation::logitHalfInverse(1.-error);
    return 0;
  }
  //
  // end of Discrete mode
  //
  else if( mode_ == SprTrainedAdaBoost::Real ) {

    // loop through training data and compute responses
    double eps = SprUtils::eps();
    for( unsigned int i=0;i<size;i++ ) {
      const SprPoint* p = (*data_)[i];
      double r = c->response(p);
      if( r<0. || r>1. ) {
	cerr << "Classifier response out of range for event " << i 
	     << ":  " << r << endl;
	return 2;
      }
      r += (1.-2.*r)*epsilon_;
      double wFactor = 0;
      if(      p->class_ == cls0_ ) {
	if( r > 1.-eps ) {
	  if( verbose > 1 ) {
	    cout << "Classifier response too high for background point " << i 
		 << ":  " << r << endl;
	  }
	  continue;
	}
	wFactor = r/(1.-r);
      }
      else if( p->class_ == cls1_ ) {
	if( r < eps ) {
	  if( verbose > 1 ) {
	    cout << "Classifier response too low for signal point " << i 
		 << ":  " << r << endl;
	  }
	  continue;
	}
	wFactor = (1.-r)/r;
      }
      data_->uncheckedSetW(i,wFactor*data_->w(i));
    }

    // return beta
    beta = 1.;
    return 0;
  }
  //
  // end of Real mode
  //
  else if( mode_ == SprTrainedAdaBoost::Epsilon ) {

    // loop through training data and classify points
    double wFactor = exp(2.*epsilon_);
    for( unsigned int i=0;i<size;i++ ) {
      const SprPoint* p = (*data_)[i];
      double w = data_->w(i);
      if( c->accept(p) ) {
	if(      p->class_ == cls0_ )
	  data_->uncheckedSetW(i,w*wFactor);
      }
      else {
	if( p->class_ == cls1_ )
	  data_->uncheckedSetW(i,w*wFactor);
      }
    }

    // return beta
    beta = epsilon_;
    return 0;
  }
  //
  // end of Epsilon mode
  //

  // if we get up to here, something must be wrong
  return 2;
}


bool SprAdaBoost::train(int verbose)
{
  // sanity check
  if( (cycles_==0 || trainable_.empty()) && beta_.size()==trained_.size() ) {
    cout << "AdaBoost will exit without training." << endl;
    return true;
  }

  // save initial data weights
  data_->weights(initialDataWeights_);

  // loop through trained classifiers first
  double beta = 0;
  unsigned nCycle = 0;
  unsigned int nTrained = trained_.size();
  if( nTrained > 0 ) {
    //
    // fill betas and reweight events
    //
    if( !beta_.empty() ) {
      int nBeta = ( nTrained>beta_.size() ? beta_.size() : nTrained );
      beta_.erase(beta_.begin()+nBeta,beta_.end());
      if( skipReweighting_ ) {
	cout << "Skipping initial event reweighting with trained " 
	     << "classifiers..." << endl;
      }
      else {
	for( unsigned int i=0;i<beta_.size();i++ ) {
	  assert( beta_[i] > 0 );
	  beta = beta_[i];
	  if( this->reweight(trained_[i].first,beta,true,verbose) != 0 ) {
	    cerr << "Unable to compute beta for trained classifier " 
		 << i << endl;
	    return this->prepareExit(false);
	  }
	}
      }
      nCycle += beta_.size();
      cout << "AdaBoost reconstructed " << beta_.size() 
	   << " beta weights." << endl;
    }

    //
    // if betas are not filled for all trained classifiers, fill the rest
    //
    for( unsigned int i=beta_.size();i<nTrained;i++ ) {
      if( this->reweight(trained_[i].first,beta,false,verbose) != 0 ) {
	cerr << "Unable to compute beta for trained classifier " << i << endl;
	return this->prepareExit(false);
      }
      if( verbose>1 || ( (nCycle%100)==0 && valPrint_==0 ) ) {
	cout << "AdaBoost beta= " << beta 
	     << " at cycle " << nCycle << endl;
      }
      if( beta < 0 ) {
	if( verbose > 1 ) {
	  cout << "Beta negative. AdaBoost cannot improve for" 
	       << " trained classifier " << i << endl;
	}
	beta = 0.;
      }
      beta_.push_back(beta);
      nCycle++;
    }
  }// end nTrained>0
  assert( nCycle == nTrained );

  // after all betas are filled, do an overall validation
  if( valData_ != 0 ) {
    // compute cumulative beta weights for validation points
    valBeta_.clear();
    valBeta_.resize(valData_->size(),0);
    for( unsigned int i=0;i<valData_->size();i++ ) {
      const SprPoint* p = (*valData_)[i];
      double resp = 0;
      for( unsigned int j=0;j<trained_.size();j++ ) {
	int out = ( trained_[j].first->accept(p,resp) ? 1 : -1 );
	if(      mode_==SprTrainedAdaBoost::Discrete 
		 || mode_==SprTrainedAdaBoost::Epsilon )
	  valBeta_[i] += out*beta_[j];
	else if( mode_ == SprTrainedAdaBoost::Real ) {
	  resp += (1.-2.*resp)*epsilon_;
	  if(      resp > 1.-SprUtils::eps() ) 
	    valBeta_[i] = SprUtils::max();
	  else if( resp < SprUtils::eps() ) 
	    valBeta_[i] = SprUtils::min();
	  else
	    valBeta_[i] += beta_[j]*SprTransformation::logitHalfInverse(resp);
	}
      }
    }

    // print out
    if( valPrint_ > 0) {
      if( !this->printValidation(nCycle) ) {
	cerr << "Unable to print out validation data." << endl;
	return this->prepareExit(false);
      }
    }
  }

  // if no trainable classifiers, exit
  if( trainable_.empty() ) return this->prepareExit((this->nTrained()>0));

  // loop through trainable
  unsigned nMax = cycles_+nTrained;
  unsigned nFailed = 0;
  while( nCycle < nMax ) {
    for( unsigned int i=0;i<trainable_.size();i++ ) {
      // exit if enough cycles are made
      if( nCycle++ >= nMax ) return this->prepareExit((this->nTrained()>0));

      // get next classifier
      SprAbsClassifier* c = trainable_[i].first;

      // make a bootstrap replica if requested
      auto_ptr<SprEmptyFilter> temp;
      if( bootstrap_ == 0 ) {
	if( !c->reset() ) {
	  cerr << "Unable to reset trainable classifier " << i << endl;
	  return this->prepareExit(false);
	}
      }
      else {
	temp.reset(bootstrap_->plainReplica());
	if( temp->size() != data_->size() ) {
	  cerr << "Failed to generate bootstrap replica." << endl;
	  return this->prepareExit(false);
	}
	if( !c->setData(temp.get()) ) {
	  cerr << "Unable to set data for classifier " << i << endl;
	  return this->prepareExit(false);
	}
      }

      // train
      if( !c->train(verbose-1) ) {
	if( ++nFailed >= trainable_.size() ) {
	  cout << "AdaBoost is exiting after making " 
	       << nFailed << " useless cycles." << endl;
	  return this->prepareExit((this->nTrained()>0));
	}
	else
	  continue;
      }

      // make a trained classifier
      SprAbsTrainedClassifier* t = c->makeTrained();
      if( t == 0 ) {
	if( ++nFailed >= trainable_.size() ) {
	  cout << "AdaBoost is exiting after making " 
	       << nFailed << " useless cycles." << endl;
	  return this->prepareExit((this->nTrained()>0));
	}
	else
	  continue;
      }
      if( trainable_[i].second.empty() )
	t->setCut(this->optimizeCut(t,verbose));
      else
	t->setCut(trainable_[i].second);

      // compute beta
      int status = this->reweight(t,beta,false,verbose);
      if( status == 2 ) {
	cerr << "Unable to compute beta for trainable classifier " 
	     << i << endl;
	return this->prepareExit((this->nTrained()>0));
      }
      if( verbose>0 || ( (nCycle%100)==0 && valPrint_==0 ) ) {
	cout << "AdaBoost beta= " << beta << " at cycle " << nCycle 
	     << " with classifier " << t->name().c_str() << endl;
      }

      // check for perfect separation
      if( status == 1 ) {
	for( unsigned int k=0;k<beta_.size();k++ ) beta_[k] = 0;
	trained_.push_back(pair<const SprAbsTrainedClassifier*,bool>(t,true));
	beta_.push_back(1);
	cout << "AdaBoost exiting since perfect separation is achieved." 
	     << endl;
	return this->prepareExit((this->nTrained()>0));
      }

      // update number of useless cycles if beta negative
      if( status == -1 ) {
	delete t;
	if( ++nFailed >= trainable_.size() ) {
	  cout << "AdaBoost is exiting after making " 
	       << nFailed << " useless cycles." << endl;
	  return this->prepareExit((this->nTrained()>0));
	}
	if( verbose > 1 ) {
	  cout << "Beta negative. AdaBoost cannot improve for" 
	       << " trainable classifier " << i << endl;
	}
      }
      else {
	nFailed = 0;
	trained_.push_back(pair<const SprAbsTrainedClassifier*,bool>(t,true));
	beta_.push_back(beta);
      }

      // validation
      if( valData_ != 0 ) {
	// update beta weights
	if( beta > 0 ) {
	  for( unsigned int j=0;j<valData_->size();j++ ) {
	    const SprPoint* p = (*valData_)[j];
	    double resp = 0;
	    int out = ( t->accept(p,resp) ? 1 : -1 );
	    if(      mode_==SprTrainedAdaBoost::Discrete
		     || mode_==SprTrainedAdaBoost::Epsilon )
	      valBeta_[j] += out*beta;
	    else if( mode_ == SprTrainedAdaBoost::Real ) {
	      resp += (1.-2.*resp)*epsilon_;
	      if(      resp > 1.-SprUtils::eps() ) 
		valBeta_[j] = SprUtils::max();
	      else if( resp < SprUtils::eps() ) 
		valBeta_[j] = -SprUtils::max();
	      else
		valBeta_[j] += beta*SprTransformation::logitHalfInverse(resp);
	    }
	  }
	}

	// print out
	if( valPrint_!=0 && (nCycle%valPrint_)==0 ) {
	  if( !this->printValidation(nCycle) ) {
	    cerr << "Unable to print out validation data." << endl;
	    return this->prepareExit(false);
	  }
	}
      }// end valData_ != 0
    }
  }
   
  // normal exit
  return this->prepareExit((this->nTrained()>0));
}


SprTrainedAdaBoost* SprAdaBoost::makeTrained() const
{
  // sanity check
  if( beta_.empty() || trained_.empty() ) return 0;

  // remove the excessive part of trained_
  int size = beta_.size();
  vector<pair<const SprAbsTrainedClassifier*,bool> > trained;
  for( int i=0;i<size;i++ ) {
    SprAbsTrainedClassifier* c = trained_[i].first->clone();
    trained.push_back(pair<const SprAbsTrainedClassifier*,bool>(c,true));
  }

  // Make a trained AdaBoost and deliver ownership
  // of trained subclassifiers to it.
  SprTrainedAdaBoost* t = new SprTrainedAdaBoost(trained,beta_,
						 useStandard_,mode_);
  t->setEpsilon(epsilon_);

  // vars
  vector<string> vars;
  data_->vars(vars);
  t->setVars(vars);

  // exit
  return t;
}


bool SprAdaBoost::prepareExit(bool status)
{
  // init
  bool success = status;

  // reset input filter
  data_->weights(trainedDataWeights_);
  if( !data_->setWeights(initialDataWeights_) ) success = false;
  initialDataWeights_.clear();

  // restore the original data supplied to the classifiers
  for( unsigned int i=0;i<trainable_.size();i++ ) {
    SprAbsClassifier* c = trainable_[i].first;
    if( !c->setData(data_) )
      cerr << "Unable to restore original data for classifier " << i << endl;
  }
    
  // exit
  return success;
}


void SprAdaBoost::print(std::ostream& os) const
{
  assert( beta_.size() == trained_.size() );
  os << "Trained AdaBoost " << SprVersion << endl;
  os << "Classifiers: " << trained_.size() << endl;
  os << "Mode: " << int(mode_) << "   Epsilon: " << epsilon_ << endl;
  for( unsigned int i=0;i<trained_.size();i++ ) {
    char s [200];
    sprintf(s,"Classifier %6i %s Beta: %12.10f",
	    i,trained_[i].first->name().c_str(),beta_[i]);
    os << s << endl;
  }
  os << "Classifiers:" << endl;
  for( unsigned int i=0;i<trained_.size();i++ ) {
    os << "Classifier " << i 
       << " " << trained_[i].first->name().c_str() << endl;
    trained_[i].first->print(os);
  }
}


SprCut SprAdaBoost::optimizeCut(const SprAbsTrainedClassifier* c,
				int verbose) const
{
  // no optimization for Real AdaBoost because the cut is never used
  // return the default cut in this case
  if( mode_ == SprTrainedAdaBoost::Real ) return c->cut();

  // init cut
  double zcut = 0;

  // store classifier output
  vector<pair<double,double> > r0, r1;
  vector<double> r(data_->size());
  for( unsigned int i=0;i<data_->size();i++ ) {
    const SprPoint* p = (*data_)[i];
    double rsp = c->response(p);
    r[i] = rsp;
    if(      p->class_ == cls0_ )
      r0.push_back(pair<double,double>(rsp,data_->w(i)));
    else if( p->class_ == cls1_ ) 
      r1.push_back(pair<double,double>(rsp,data_->w(i)));
  }
    
  // init weights
  double wcor0(0), wmis1(0);
  double wmis0 = 0;
  for( unsigned int i=0;i<r0.size();i++ )
    wmis0 += r0[i].second;
  double wcor1 = 0;
  for( unsigned int i=0;i<r1.size();i++ )
    wcor1 += r1[i].second;
  assert( wmis0>0 && wcor1>0 );
  double wtot = wmis0+wcor1;
  if( verbose > 2 ) 
    cout << "Optimizing cut for W0=" << wmis0 << " W1=" << wcor1 << endl;
    
  // sort, init min and max, and init iterators
  stable_sort(r.begin(),r.end(),less<double>());
  stable_sort(r0.begin(),r0.end(),SABCmpPairDDFirst());
  stable_sort(r1.begin(),r1.end(),SABCmpPairDDFirst());
  vector<pair<double,double> >::iterator i0start(r0.begin()), 
    i0split(r0.begin());
  vector<pair<double,double> >::iterator i1start(r1.begin()),
    i1split(r1.begin());
  if( verbose > 2 ) {
    if( !r0.empty() ) {
      cout << "Classifier range for 0: " 
	   << r0[0].first << " " << r0[r0.size()-1].first << endl;
    }
    if( !r1.empty() ) {
      cout << "Classifier range for 1: " 
	   << r1[0].first << " " << r1[r1.size()-1].first << endl;
    }
  }
    
  // fill out divisions
  vector<double> division;
  division.push_back(SprUtils::min());
  double xprev = r[0];
  for( unsigned int k=1;k<r.size();k++ ) {
    double xcurr = r[k];
    if( (xcurr-xprev) > SprUtils::eps() ) {
      division.push_back(0.5*(xcurr+xprev));
      xprev = xcurr;
    }
  }
  division.push_back(SprUtils::max());

  // find optimal point
  int ndiv = division.size();
  vector<double> f(ndiv);
  for( int k=0;k<ndiv;k++ ) {
    double z = division[k];
    i0split = find_if(i0start,r0.end(),
		      not1(bind2nd(SABCmpPairDDFirstNumber(),z)));
    i1split = find_if(i1start,r1.end(),
		      not1(bind2nd(SABCmpPairDDFirstNumber(),z)));
    for( vector<pair<double,double> >::iterator iter=i0start;
	 iter!=i0split;iter++ ) {
      wcor0 += iter->second;
      wmis0 -= iter->second;
    }
    for( vector<pair<double,double> >::iterator iter=i1start;
	 iter!=i1split;iter++ ) {
      wmis1 += iter->second;
      wcor1 -= iter->second;
    }
    i0start = i0split;
    i1start = i1split;
    f[k] = (wmis0+wmis1)/wtot;
  }

  // find point giving min misclassification fraction
  vector<double>::iterator imin = min_element(f.begin(),f.end());
  int k = imin - f.begin();
  zcut = division[k];

  // exit
  return SprUtils::lowerBound(zcut);
}


bool SprAdaBoost::setValidation(const SprAbsFilter* valData, 
				unsigned valPrint,
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

  // if no loss specified, use exponential by default
  loss_ = loss;
  ownLoss_ = false;
  if( loss_ == 0 ) {
    loss_ = new SprAverageLoss(&SprLoss::exponential);
    ownLoss_ = true;
  }

  // exit
  return true;
}


bool SprAdaBoost::printValidation(unsigned cycle)
{
  // no print-out for zero training cycle
  if( cycle == 0 ) return true;

  // sanity check
  assert(valBeta_.size() == valData_->size());

  // reset loss
  assert( loss_ != 0 );
  loss_->reset();

  // loop through validation data
  for( unsigned int i=0;i<valData_->size();i++ ) {
    const SprPoint* p = (*valData_)[i];
    double w = valData_->w(i);
    if( p->class_!=cls0_ && p->class_!=cls1_ ) w = 0;
    if(      p->class_ == cls0_ )
      loss_->update(0,valBeta_[i],w);
    else if( p->class_ == cls1_ )
      loss_->update(1,valBeta_[i],w);
  }

  // compute fom
  cout << "Validation Loss=" << loss_->value() 
       << " at cycle " << cycle << endl;

  // exit
  return true;
}


bool SprAdaBoost::storeData(const char* filename) const
{
  // save data weights
  vector<double> weights;
  data_->weights(weights);
  if( !data_->setWeights(trainedDataWeights_) ) return false;
  if( !data_->store(filename) ) return false;
  if( !data_->setWeights(weights) ) return false;

  // exit
  return true;
}


bool SprAdaBoost::setClasses(const SprClass& cls0, const SprClass& cls1) 
{
  for( unsigned int i=0;i<trainable_.size();i++ ) {
    if( !trainable_[i].first->setClasses(cls0,cls1) ) {
      cerr << "AdaBoost unable to reset classes for classifier " << i << endl;
      return false;
    }
  }
  cls0_ = cls0; cls1_ = cls1;
  cout << "Classes for AdaBoost reset to " << cls0_ << " " << cls1_ << endl;
  return true;
}
