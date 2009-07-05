//$Id: SprFisher.cc,v 1.2 2007/09/21 22:32:10 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprFisher.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"

#include <stdio.h>
#include <iostream>
#include <cassert>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace std;


SprFisher::SprFisher(SprAbsFilter* data, int mode) 
  : 
  SprAbsClassifier(data),
  mode_(mode),
  cls0_(0),
  cls1_(1),
  dim_(data->dim()),
  linear_(dim_),
  quadr_(dim_),
  cterm_(0)
{
  this->setClasses();
}


bool SprFisher::setData(SprAbsFilter* data)
{
  assert( data != 0 );
  data_ = data;
  return this->reset();
}


SprTrainedFisher* SprFisher::makeTrained() const
{
  // make
  SprTrainedFisher* t = 0;
  if(      mode_ == 1 )
    t = new SprTrainedFisher(linear_,cterm_);
  else if( mode_ == 2 )
    t = new SprTrainedFisher(linear_,quadr_,cterm_);

  // vars
  vector<string> vars;
  data_->vars(vars);
  t->setVars(vars);

  // exit
  return t;
}


bool SprFisher::train(int verbose)
{
  // init
  SprVector mean0(dim_), mean1(dim_);
  SprSymMatrix cov0(dim_), cov1(dim_);

  // be paranoid and fill with zeros
  for( unsigned int i=0;i<dim_;i++ ) {
    mean0[i] = 0;
    mean1[i] = 0;
    for( unsigned int j=i;j<dim_;j++ ) {
      cov0[i][j] = 0;
      cov1[i][j] = 0;
    }
  }

  // loop through points to compute mean vectors and covariance matrices
  unsigned size = data_->size();
  if( size == 0 ) {
    cerr << "No points in data." << endl;
    return false;
  }
  double size0(0), size1(0);
  double w = 0;
  double r1(0), r2(0);
  for( unsigned int i=0;i<size;i++ ) {
    const SprPoint* p = (*data_)[i];
    unsigned int cls = p->class_;
    if( cls==cls0_ || cls==cls1_ ) {
      w = data_->w(i);
      // increment weights
      if(      cls == cls0_ )
	size0 += w;
      else if( cls == cls1_ )
	size1 += w;
      // loop through dimensions
      for( unsigned int j=0;j<dim_;j++ ) {
	r1 = w * (p->x_)[j];
	if(      cls == cls0_ )
	  mean0[j] += r1;
	else if( cls == cls1_ )
	  mean1[j] += r1;
	for( unsigned int k=j;k<dim_;k++ ) {
	  r2 = r1*((p->x_)[k]);
	  if(      cls == cls0_ )
	    cov0[j][k] += r2;
	  else if( cls == cls1_ )
	    cov1[j][k] += r2;
	}
      }
    }
  }
  double eps = SprUtils::eps();
  if( size0<eps || size1<eps ) {
    cerr << "Cannot find points for a class: " 
	 << size0 << " " << size1 << endl;
    return false;
  }

  // normalize and compute covariances
  mean0 /= size0;
  mean1 /= size1;
  cov0 /= size0;
  cov1 /= size1;
  SprSymMatrix meansq0 = vT_times_v(mean0);
  SprSymMatrix meansq1 = vT_times_v(mean1);
  cov0 -= meansq0;
  cov1 -= meansq1;
  if( mode_ == 1 ) {
    cov0 = (size0*cov0+size1*cov1) / (size0+size1);
  }

  // print out
  if( verbose > 1 ) {
    cout << "Sample means:" << endl;
    for( unsigned int i=0;i<dim_;i++ )
      cout << i << ": " << mean0[i] << " " << mean1[i] << endl;
    cout << "Sample covariance matrices:" << endl;
    if( mode_ == 2 )
      cout << "Class " << cls0_ << endl;
    for( unsigned int i=0;i<dim_;i++ ) {
      for( unsigned int j=0;j<dim_;j++ )
	cout << " " << cov0[i][j];
      cout << endl;
    }
    if( mode_ == 2 ) {
      cout << "Class " << cls1_ << endl;
      for( unsigned int i=0;i<dim_;i++ ) {
	for( unsigned int j=0;j<dim_;j++ )
	  cout << " " << cov1[i][j];
	cout << endl;
      }
    }
  }

  // compute Fisher coefficients
  int ifail = 0;
  cov0.invert(ifail);
  if( ifail != 0 ) {
    cerr << "Unable to invert matrix." << endl;
    return false;
  }
  if( verbose > 1 ) {
    cout << "Inverse matrices:" << endl;
    if( mode_ == 2 )
      cout << "Class " << cls0_ << endl;
    for( unsigned int i=0;i<dim_;i++ ) {
      for( unsigned int j=0;j<dim_;j++ )
	cout << " " << cov0[i][j];
      cout << endl;
    }
  }
  if( mode_ == 2 ) {
    cov1.invert(ifail);
    if( ifail != 0 ) {
      cerr << "Unable to invert matrix." << endl;
      return false;
    }
    if( verbose > 1 ) {
      cout << "Class " << cls1_ << endl;
      for( unsigned int i=0;i<dim_;i++ ) {
	for( unsigned int j=0;j<dim_;j++ )
	  cout << " " << cov1[i][j];
	cout << endl;
      }
    }
  }
  cterm_ = log(size1/size0);
  if(      mode_ == 1 ) {
    linear_ = cov0 * (mean1-mean0);
    cterm_ += -0.5 * (dot(mean1,cov0*mean1) - dot(mean0,cov0*mean0));
  }
  else if( mode_ == 2 ) {
    linear_ = cov1*mean1 - cov0*mean0;
    quadr_ = -0.5 * (cov1-cov0);
    cterm_ += -0.5 * (dot(mean1,cov1*mean1) - dot(mean0,cov0*mean0));
    double d0 = cov0.determinant();
    double d1 = cov1.determinant();
    cterm_ += 0.5*log(d1/d0);//inverted matrices => must have "+", not "-"
  }

  // exit
  return true;
}


void SprFisher::print(std::ostream& os) const
{
  os << "Trained Fisher " << SprVersion << endl;
  os << "Fisher dimensionality: " << linear_.num_row() << endl;
  os << "Fisher response: F = C + T(L)*X + T(X)*Q*X; T is transposition" 
     << endl;
  os << "By default logit transform is applied: F <- 1/[1+exp(-F)]" << endl;
  os << "Fisher order: " << mode_ << endl;
  os << "Const term (C): " << cterm_ << endl;
  os << "Linear Part (L):" << endl;
  for( int i=0;i<linear_.num_row();i++ )
    os << setw(10) << linear_[i] << " ";
  os << endl;
  if( mode_ == 2 ) {
    os << "Quadratic Part (Q):" << endl;
    for( int i=0;i<quadr_.num_row();i++ ) {
      for( int j=0;j<quadr_.num_col();j++ ) {
	os << setw(10) << quadr_[i][j] << " ";
      }
      os << endl;
    }
  }
}


void SprFisher::setClasses() 
{
  vector<SprClass> classes;
  data_->classes(classes);
  int size = classes.size();
  if( size > 0 ) cls0_ = classes[0];
  if( size > 1 ) cls1_ = classes[1];
  cout << "Classes for Fisher are set to " << cls0_ << " " << cls1_ << endl;
} 
