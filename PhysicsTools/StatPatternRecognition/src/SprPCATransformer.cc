//$Id: SprPCATransformer.cc,v 1.1 2007/11/12 06:19:18 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPCATransformer.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDataMoments.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"
#include "PhysicsTools/StatPatternRecognition/src/SprVector.hh"
#include "PhysicsTools/StatPatternRecognition/src/SprSymMatrix.hh"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <functional>
#include <sstream>

using namespace std;


struct PCACmpPairDIFirst
  : public binary_function<pair<double,int>,pair<double,int>,bool> {
  bool operator()(const pair<double,int>& l, const pair<double,int>& r)
    const {
    return ( fabs(l.first) > fabs(r.first) );
  }
};


SprPCATransformer::SprPCATransformer()
  :
  SprAbsVarTransformer(),
  U_(),
  eigenValues_()
{}


SprPCATransformer::SprPCATransformer(const SprMatrix& U, 
				     const std::vector<
				     std::pair<double,int> >& eigenValues)
  :
  SprAbsVarTransformer(),
  U_(U),
  eigenValues_(eigenValues)
{}


bool SprPCATransformer::train(const SprAbsFilter* data, int verbose)
{
  // sanity check
  assert( data != 0 );

  // init
  eigenValues_.clear();
  U_ = SprMatrix();

  // get dimensionality and a list of vars
  unsigned dim = data->dim();
  data->vars(oldVars_);
  assert( dim == oldVars_.size() );
  newVars_.clear();
  newVars_.resize(dim);
  for( unsigned int d=0;d<dim;d++ ) {
    ostringstream os;
    os << "pc" << d;
    newVars_[d] = os.str();
  }

  // compute covariance matrix
  SprDataMoments moms(data);
  SprVector mean;
  SprSymMatrix cov;
  if( !moms.covariance(cov,mean) ) {
    cerr << "Unable to compute covariance matrix for PCA." << endl;
    return false;
  }

  // diagonalize
  SprMatrix U = cov.diagonalize().T();
  if( U.num_row()!=static_cast<int>(dim) || U.num_col()!=static_cast<int>(dim) ) {
    cerr << "Dimensionality of the PCA transformation matrix does not match." 
	 << endl;
    return false;
  }

  // sort eigenvalues
  eigenValues_.resize(dim);
  for( unsigned int d=0;d<dim;d++ ) 
    eigenValues_[d] = pair<double,int>(cov[d][d],d);
  stable_sort(eigenValues_.begin(),eigenValues_.end(),PCACmpPairDIFirst());
  if( verbose > 0 ) {
    cout << "PCA eigenvalues: ";
    for( unsigned int d=0;d<dim;d++ ) cout << eigenValues_[d].first << " ";
    cout << endl;
  }

  // sort rows of the transformation matrix
  U_ = U;
  for( unsigned int iold=0;iold<dim;iold++ ) {
    unsigned int inew = eigenValues_[iold].second;
    for( unsigned int j=0;j<dim;j++ )
      U_[inew][j] = U[iold][j];
  }

  // exit
  return true;
}


void SprPCATransformer::transform(const std::vector<double>& in, 
				  std::vector<double>& out) const
{
  unsigned int dim = U_.num_row();
  assert( in.size() == dim );
  SprVector v(in);
  out = (U_*v).std();
}


void SprPCATransformer::inverse(const std::vector<double>& in, 
				std::vector<double>& out) const
{
  unsigned int dim = U_.num_row();
  assert( in.size() == dim );
  SprVector v(in);
  out = (U_.T()*v).std();
}


void SprPCATransformer::print(std::ostream& os) const
{
  // save name
  os << "VarTransformer: " << this->name().c_str() 
     << " " << SprVersion.c_str() << endl;

  // init
  int dim = oldVars_.size();
  vector<string> vars(oldVars_);

  // protect againt spaces in var names
  for( int i=0;i<dim;i++ ) {
    if( vars[i].find(' ') != string::npos )
	vars[i].erase(vars[i].find_first_of(' '));
  }

  // store dimensionality
  os << "Dim: " << dim << endl;

  // store eigenvalues
  os << "Eigenvalues:";
  for( int d=0;d<dim;d++ )
    os << " " << eigenValues_[d].first;
  os << endl;

  // store indices
  os << "Indices:";
  for( int d=0;d<dim;d++ )
    os << " " << eigenValues_[d].second;
  os << endl;

  // store transformations
  for( int i=0;i<dim;i++ ) {
    os << i << " " << newVars_[i].c_str() << "=";
    for( int j=0;j<dim;j++ )
      os << " + " << U_[i][j] << " *" << vars[j].c_str();
    os << endl;
  }
}

  
