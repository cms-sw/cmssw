//$Id: SprLogitR.cc,v 1.2 2007/09/21 22:32:10 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprLogitR.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformation.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprFisher.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"

#include "PhysicsTools/StatPatternRecognition/src/SprMatrix.hh"
#include "PhysicsTools/StatPatternRecognition/src/SprSymMatrix.hh"

#include <stdio.h>
#include <iostream>
#include <cassert>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace std;


SprLogitR::SprLogitR(SprAbsFilter* data, double eps, double updateFactor)
  :
  SprAbsClassifier(data),
  cls0_(0),
  cls1_(1),
  dim_(data->dim()),
  eps_(eps),
  updateFactor_(updateFactor),
  nIterAllowed_(100),
  beta0_(0),
  beta_(dim_),
  beta0Supplied_(0),
  betaSupplied_()
{
  assert( eps_ > 0 );
  this->setClasses();
}


SprLogitR::SprLogitR(SprAbsFilter* data, 
		     double beta0, const SprVector& beta,
		     double eps, double updateFactor)
  :
  SprAbsClassifier(data),
  cls0_(0),
  cls1_(1),
  dim_(data->dim()),
  eps_(eps),
  updateFactor_(updateFactor),
  nIterAllowed_(100),
  beta0_(0),
  beta_(dim_),
  beta0Supplied_(beta0),
  betaSupplied_(beta)
{
  assert( eps_ > 0 );
  assert( updateFactor_ > 0 );
  this->setClasses();
}


bool SprLogitR::setData(SprAbsFilter* data)
{
  assert( data != 0 );
  data_ = data;
  return this->reset();
}


SprTrainedLogitR* SprLogitR::makeTrained() const
{
  // make
  SprTrainedLogitR* t = new SprTrainedLogitR(beta0_,beta_);

  // vars
  vector<string> vars;
  data_->vars(vars);
  t->setVars(vars);

  // exit
  return t;
}


bool SprLogitR::train(int verbose)
{
  // initialize to user-supplied values
  if( (int)dim_ == betaSupplied_.num_row() ) {
    beta0_ = beta0Supplied_;
    beta_ = betaSupplied_;
  }
  else {// obtain initial estimates from LDA
    // message
    if( verbose > 0 ) {
      cout << "Obtaining initial estimates of Logit coefficients " 
	   << "from LDA..." << endl;
    }

    // train LDA
    SprFisher fisher(data_,1);
    if( fisher.train(verbose) ) {
      beta0_ = fisher.cterm();
      fisher.linear(beta_);
      if( verbose > 0 ) {
	cout << "...Obtained estimates of Logit coefficients from LDA." << endl;
      }
    }
    else {
      cout << "Unable to train LDA. Will use zeros for initial estimates of " 
	   << "Logit coefficients." << endl;
      for( int i=0;i<beta_.num_row();i++ ) beta_[i] = 0;
    }

  }// end of LDA
  assert( beta_.num_row() == (int)dim_ );

  //
  // prepare matrices
  //

  // renormalize weights
  unsigned n0 = data_->ptsInClass(cls0_);
  unsigned n1 = data_->ptsInClass(cls1_);
  assert( n0>0 && n1>0 );
  unsigned N = n0 + n1;
  double w0 = data_->weightInClass(cls0_);
  double w1 = data_->weightInClass(cls1_);
  assert( w0>0 && w1>0 );
  double wFactor = double(N)/(w0+w1); 
  SprVector weights(N);
  for( unsigned int i=0;i<N;i++ )
    weights[i] = wFactor*data_->w(i);

  // vector of fitted beta
  SprVector betafit(dim_+1);
  betafit[0] = beta0_;
  for( int i=1;i<betafit.num_row();i++ )
    betafit[i] = beta_[i-1];

  // vector of fitted probabilities
  SprVector prob;

  // input data matrix
  SprMatrix X(N,dim_+1);
  for( unsigned int i=0;i<N;i++ ) {
    X[i][0] = 1;
    const SprPoint* p = (*data_)[i]; 
    for( unsigned int j=1;j<dim_+1;j++ ) X[i][j] = (p->x_)[j-1];
  }

  // vector of classes
  SprVector y(N);
  for( unsigned int i=0;i<N;i++ ) {
    const SprPoint* p = (*data_)[i]; 
    if(      p->class_ == cls0_ )
      y[i] = 0;
    else if( p->class_ == cls1_ )
      y[i] = 1;
  }

  //
  // minimize
  //
  double eps = 1;
  unsigned int iter = 0;
  while( true ) {
    if( ++iter > nIterAllowed_ ) {
      cerr << "Logit exiting because number of alowed iterations exceeded: " 
	   << iter << " " << nIterAllowed_ << endl;
      return false;
    }
    if( !this->iterate(y,X,weights,prob,betafit,eps) ) {
      cerr << "Unable to iterate Logit coefficients at step " << iter << endl;
      return false;
    }
    if( verbose > 0 )
      cout << "Iteration " << iter << " obtains epsilon " << eps << endl;
    if( eps < eps_ ) break;
  }

  // get back optimized betas
  beta0_ = betafit[0];
  for( int i=1;i<betafit.num_row();i++ )
    beta_[i-1] = betafit[i];
  
  // exit
  return true;
}


bool SprLogitR::iterate(const SprVector& y,
			const SprMatrix& X, 
			const SprVector& weights, 
			SprVector& prob, 
			SprVector& betafit, 
			double& eps)
{
  // get sample size
  const unsigned N = X.num_row();
  const unsigned D = X.num_col();

  // compute probabilities
  SprVector pold(N);
  if( prob.num_row() == 0 ) {
    for( unsigned int i=0;i<N;i++ )
      pold[i] = SprTransformation::logit(dot(X.sub(i+1,i+1,1,D).T(),betafit));
  }
  else
    pold = prob;

  // fill out W vector
  SprVector W(N);
  for( unsigned int i=0;i<N;i++ ) {
    W[i] = weights[i]*pold[i]*(1.-pold[i]);
    if( W[i] < 0 ) W[i] = 0;
    if( W[i] > 1 ) W[i] = 1;
  }

  // iterate
  SprSymMatrix XTWX(D);
  for( unsigned int i=0;i<D;i++ ) {
    for( unsigned int j=i;j<D;j++ ) {
      double res = 0;
      for( unsigned int n=0;n<N;n++ )
	res += W[n]*X[n][i]*X[n][j];
      XTWX[i][j] = res;
    }
  }
  int ifail = 0;
  XTWX.invert(ifail);
  if( ifail != 0 ) {
    cerr << "Unable to invert matrix for Logit coefficients." << endl;
    return false;
  }
  betafit += updateFactor_ * (XTWX * (X.T()*(y-pold)));

  // update probabilities
  SprVector pnew(N);
  for( unsigned int i=0;i<N;i++ )
    pnew[i] = SprTransformation::logit(dot(X.sub(i+1,i+1,1,D).T(),betafit));
  
  // compute eps per event
  eps = 0;
  for( unsigned int i=0;i<N;i++ )
    eps += fabs(pnew[i]-pold[i]);
  eps /= N;

  // exit
  prob = pnew;
  return true;
}


void SprLogitR::print(std::ostream& os) const
{
  os << "Trained LogitR " << SprVersion << endl;
  os << "LogitR dimensionality: " << beta_.num_row() << endl;
  os << "LogitR response: L = Beta0 + Beta*X" << endl;  
  os << "By default logit transform is applied: L <- 1/[1+exp(-L)]" << endl;
  os << "Beta0: " << beta0_ << endl;
  os << "Vector of Beta Coefficients:" << endl;
  for( int i=0;i<beta_.num_row();i++ )
    os << setw(10) << beta_[i] << " ";
  os << endl;
}


void SprLogitR::setClasses() 
{
  vector<SprClass> classes;
  data_->classes(classes);
  int size = classes.size();
  if( size > 0 ) cls0_ = classes[0];
  if( size > 1 ) cls1_ = classes[1];
  cout << "Classes for LogitR are set to " << cls0_ << " " << cls1_ << endl;
} 
