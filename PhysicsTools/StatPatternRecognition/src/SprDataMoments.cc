//$Id: SprDataMoments.cc,v 1.2 2007/09/21 22:32:10 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDataMoments.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/src/SprChiCdf.hh"

#include <iostream>
#include <cassert>
#include <cmath>

using namespace std;

double SprDataMoments::mean(int i) const
{
  if( i >= (int)data_->dim() ) {
    cerr << "Index is out of data dimensions " << i 
	 << " " << data_->dim() << endl;
    return 0;
  }
  double sum = 0;
  double size = 0;
  double w = 0;
  for( unsigned int n=0;n<data_->size();n++ ) {
    w = data_->w(n);
    sum += w * (*((*data_)[n]))[i];
    size += w;
  }
  assert( size > SprUtils::eps() );
  sum /= size;
  return sum;
}


double SprDataMoments::variance(int i, double& mean) const
{
  if( i >= (int)data_->dim() ) {
    cerr << "Index is out of data dimensions " << i 
	 << " " << data_->dim() << endl;
    return 0;
  }
  double sum(0), r(0);
  mean = this->mean(i);
  double size = 0;
  double w = 0;
  for( unsigned int n=0;n<data_->size();n++ ) {
    w = data_->w(n);
    r = (*((*data_)[n]))[i] - mean;
    sum += w * r*r;
    size += w;
  }
  assert( size > SprUtils::eps() );
  sum /= size;
  return (sum>0 ? sum : 0);
}


double SprDataMoments::correl(int i, int j, 
			      double& mean1, double& mean2, 
			      double& var1, double& var2) const
{
  if( i >= (int)data_->dim() ) {
    cerr << "Index is out of data dimensions " << i 
	 << " " << data_->dim() << endl;
    return 0;
  }
  if( j >= (int)data_->dim() ) {
    cerr << "Index is out of data dimensions " << j 
	 << " " << data_->dim() << endl;
    return 0;
  }
  double sum(0), r1(0), r2(0);
  var1 = this->variance(i,mean1);
  var2 = this->variance(j,mean2);
  assert( var1>0 && var2>0 );
  double size = 0;
  double w = 0;
  for( unsigned int n=0;n<data_->size();n++ ) {
    w = data_->w(n);
    r1 = (*((*data_)[n]))[i] - mean1;
    r2 = (*((*data_)[n]))[j] - mean2;
    sum += w * r1*r2;
    size += w;
  }
  assert( size > SprUtils::eps() );
  sum /= double(size)*sqrt(var1)*sqrt(var2);
  return sum;
}


bool SprDataMoments::covariance(SprSymMatrix& cov, SprVector& mean) const
{
  // init
  unsigned dim = data_->dim();
  mean = SprVector(dim);
  cov = SprSymMatrix(dim);

  // be paranoid and fill with zeros
  for( unsigned int i=0;i<dim;i++ ) {
    mean[i] = 0;
    for( unsigned int j=i;j<dim;j++ )
      cov[i][j] = 0;
  }

  // loop through points to compute mean vectors and covariance matrices
  unsigned int size = data_->size();
  double wtot = 0;
  double w = 0;
  double r = 0;

  // mean
  for( unsigned int i=0;i<size;i++ ) {
    const SprPoint* p = (*data_)[i];
    w = data_->w(i);
    wtot += w;
    for( unsigned int j=0;j<dim;j++ )
      mean[j] += w*(p->x_)[j];
  }
  if( wtot < SprUtils::eps() ) {
    cerr << "Unable to compute covariance: Wtot= " << wtot << endl;
    return false;
  }
  mean /= wtot;

  // covariance
  for( unsigned int i=0;i<size;i++ ) {
    const SprPoint* p = (*data_)[i];
    w = data_->w(i);
    for( unsigned int j=0;j<dim;j++ ) {
      r = w * ((p->x_)[j]-mean[j]);
      for( unsigned int k=j;k<dim;k++ )
	cov[j][k] += r * ((p->x_)[k]-mean[k]);
    }
  }
  cov /= wtot;

  // exit
  return true;
}


double SprDataMoments::kurtosis(SprSymMatrix& cov, SprVector& mean) const 
{
  // mean and covariance
  if( !this->covariance(cov,mean) ) {
    cerr << "Unable to compute kurtosis due to covariance." << endl;
    return 0;
  }

  // invert covariance
  int ifail = 0;
  SprSymMatrix invcov = cov.inverse(ifail);
  if( ifail != 0 ) {
    cerr << "Unable to invert covariance matrix for kurtosis." << endl;
    return 0;
  }

  // loop through events
  double kur = 0;
  unsigned dim = data_->dim();
  unsigned int size = data_->size();
  assert( dim>0 && size>0 );
  SprVector v(dim);
  double w(0), wtot(0);
  for( unsigned int i=0;i<size;i++ ) {
    const SprPoint* p = (*data_)[i];
    w = data_->w(i);
    for( unsigned int j=0;j<dim;j++ ) 
      v[j] = (p->x_)[j] - mean[j];
    kur += w * pow(dot(v,invcov*v),2);
    wtot += w;
  }
  if( wtot < SprUtils::eps() ) {
    cerr << "Unable to compute kurtosis: Wtot= " << wtot << endl;
    return 0;
  }
  kur /= wtot;
  kur /= (dim*(dim+2));
  kur -= 1;

  // exit
  return kur;
}


double SprDataMoments::zeroCorrCL(double corrij, double kurtosis) const
{
  if( (kurtosis+1.) < SprUtils::eps() ) {
    cerr << "Kurtosis must be greater than -1!!!" << endl;
    return 0;
  }
  assert( corrij < 1 );
  double x = data_->size()/(kurtosis+1.) * corrij*corrij/(1.-corrij*corrij);
  assert(x >= 0.0);
  double p, q, df = 1.0;
  SprChiCdf::cumchi(&x, &df, &p, &q);
  return q;
}


double SprDataMoments::zeroCorrCL(int i, int j) const
{
  if( i<0 || i>=static_cast<int>(data_->dim()) ) {
    cerr << "Index out of limits: " << i << " " << data_->dim() << endl;
    return 0;
  }
  if( j<0 || j>=static_cast<int>(data_->dim()) ) {
    cerr << "Index out of limits: " << j << " " << data_->dim() << endl;
    return 0;
  }
  SprVector mean;
  SprSymMatrix cov;
  double kur = this->kurtosis(cov,mean);
  assert( cov[i][i]>0 && cov[j][j]>0 );
  double corrij = cov[i][j] / sqrt(cov[i][i]*cov[j][j]);
  return this->zeroCorrCL(corrij,kur);
}


double SprDataMoments::mean(const char* name) const
{
  int i = data_->dimIndex(name);
  if( i < 0 ) {
    cerr << "Unable to find variable " << name << endl;
    return 0;
  }
  return this->mean(i);
}


double SprDataMoments::variance(const char* name, double& mean) const
{
  int i = data_->dimIndex(name);
  if( i < 0 ) {
    cerr << "Unable to find variable " << name << endl;
    return 0;
  }
  return this->variance(i,mean);
}


double SprDataMoments::correl(const char* name1, const char* name2, 
			      double& mean1, double& mean2, 
			      double& var1, double& var2) const
{
  int i = data_->dimIndex(name1);
  if( i < 0 ) {
    cerr << "Unable to find variable " << name1 << endl;
    return 0;
  }
  int j = data_->dimIndex(name2);
  if( j < 0 ) {
    cerr << "Unable to find variable " << name2 << endl;
    return 0;
  }
  return this->correl(i,j,mean1,mean2,var1,var2);
}


double SprDataMoments::correlClassLabel(int d, double& mean, double& var) const
{
  // sanity check
  if( d<0 || d>=static_cast<int>(data_->dim()) ) {
    cerr << "Index out of limits: " << d << " " << data_->dim() << endl;
    return 0;
  }

  // compute mean for the input variable
  mean = this->mean(d);

  // compute class mean
  unsigned int size = data_->size();
  double wtot = 0;
  double cmean(0);
  for( unsigned int i=0;i<size;i++ ) {
    const SprPoint* p = (*data_)[i];
    double w = data_->w(i);
    wtot += w;
    cmean += w*(p->class_);
  }
  if( wtot < SprUtils::eps() ) {
    cerr << "Unable to compute correlation with class label: Wtot= " 
	 << wtot << endl;
    return 0;
  }
  cmean /= wtot;
  
  // compute covariance
  var = 0;
  double cvar = 0;
  double corr = 0;
  for( unsigned int i=0;i<size;i++ ) {
    const SprPoint* p = (*data_)[i];
    double w = data_->w(i);
    var += w * pow((p->x_)[d]-mean,2);
    cvar += w * pow(p->class_-cmean,2);
    corr += w * ((p->x_)[d]-mean) * (p->class_-cmean);
  }
  var /= wtot;
  cvar /= wtot;
  corr /= wtot;

  // compute correlation
  if( cvar < SprUtils::eps() ) {
    cerr << "Unable to compute correlation with class label: Cvar= " 
	 << cvar << endl;
    return 0;
  }
  if( var < SprUtils::eps() ) {
    cerr << "Unable to compute correlation with class label: Var= " 
	 << var << endl;
    return 0;
  }
  corr /= sqrt(var*cvar);

  // exit
  return corr;
}


double SprDataMoments::absMean(int i) const
{
  if( i >= static_cast<int>(data_->dim()) ) {
    cerr << "Index is out of data dimensions " << i 
	 << " " << data_->dim() << endl;
    return 0;
  }
  double sum = 0;
  double size = 0;
  double w = 0;
  for( unsigned int n=0;n<data_->size();n++ ) {
    w = data_->w(n);
    sum += w * fabs((*((*data_)[n]))[i]);
    size += w;
  }
  assert( size > SprUtils::eps() );
  sum /= size;
  return sum;
}


double SprDataMoments::absCorrelClassLabel(int d, double& mean, double& var) 
  const
{
  // sanity check
  if( d<0 || d>=static_cast<int>(data_->dim()) ) {
    cerr << "Index out of limits: " << d << " " << data_->dim() << endl;
    return 0;
  }

  // compute mean for the input variable
  mean = this->absMean(d);

  // compute class mean
  unsigned int size = data_->size();
  double wtot = 0;
  double cmean(0);
  for( unsigned int i=0;i<size;i++ ) {
    const SprPoint* p = (*data_)[i];
    double w = data_->w(i);
    wtot += w;
    cmean += w*(p->class_);
  }
  if( wtot < SprUtils::eps() ) {
    cerr << "Unable to compute correlation with class label: Wtot= " 
	 << wtot << endl;
    return 0;
  }
  cmean /= wtot;
  
  // compute covariance
  var = 0;
  double cvar = 0;
  double corr = 0;
  for( unsigned int i=0;i<size;i++ ) {
    const SprPoint* p = (*data_)[i];
    double w = data_->w(i);
    double r = fabs((p->x_)[d]);
    var += w * pow(r-mean,2);
    cvar += w * pow(p->class_-cmean,2);
    corr += w * (r-mean) * (p->class_-cmean);
  }
  var /= wtot;
  cvar /= wtot;
  corr /= wtot;

  // compute correlation
  if( cvar < SprUtils::eps() ) {
    cerr << "Unable to compute correlation with class label: Cvar= " 
	 << cvar << endl;
    return 0;
  }
  if( var < SprUtils::eps() ) {
    cerr << "Unable to compute correlation with class label: Var= " 
	 << var << endl;
    return 0;
  }
  corr /= sqrt(var*cvar);

  // exit
  return corr;
}

