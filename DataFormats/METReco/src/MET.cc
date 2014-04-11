// -*- C++ -*-

// Package:    METReco
// Class:      MET
//
// Original authors: Michael Schmitt, Richard Cavanaugh The University of Florida
// changes by: Freya Blekman, Cornell University
//

//____________________________________________________________________________||
#include "DataFormats/METReco/interface/MET.h"
#include "TVector.h"

//____________________________________________________________________________||
using namespace std;
using namespace reco;

//____________________________________________________________________________||
MET::MET()
{
  sumet = 0.0;
  elongit = 0.0;
  signif_dxx=signif_dyy=signif_dyx=signif_dxy=0.;
}

// Constructer for the case when only p4_ =  (mEx, mEy, 0, mEt) is known.
// The vertex information is currently not used (but may be in the future)
// and is required by the RecoCandidate constructer.
//____________________________________________________________________________||
MET::MET( const LorentzVector& p4_, const Point& vtx_ ) : 
  RecoCandidate( 0, p4_, vtx_ )
{
  sumet = 0.0;
  elongit = 0.0;
  signif_dxx=signif_dyy=signif_dyx=signif_dxy=0.;
}

// Constructer for the case when the SumET is known in addition to 
// p4_ = (mEx, mEy, 0, mEt).  The vertex information is currently not used
// (but see above).
//____________________________________________________________________________||
MET::MET( double sumet_, const LorentzVector& p4_, const Point& vtx_ ) : 
  RecoCandidate( 0, p4_, vtx_ ) 
{
  sumet = sumet_;
  elongit = 0.0;
  signif_dxx=signif_dyy=signif_dyx=signif_dxy=0.;
}

// Constructor for the case when the SumET, the corrections which
// were applied to the MET, as well the MET itself p4_ = (mEx, mEy, 0, mEt)
// are all known.  See above concerning the vertex information. 
//____________________________________________________________________________||
MET::MET( double sumet_, const std::vector<CorrMETData>& corr_, 
	  const LorentzVector& p4_, const Point& vtx_ ) : 
  RecoCandidate( 0, p4_, vtx_ )
{
  sumet = sumet_;
  elongit = 0.0;
  signif_dxx=signif_dyy=signif_dyx=signif_dxy=0.;
  //-----------------------------------
  // Fill the vector containing the corrections (corr) with vector of 
  // known corrections (corr_) passed in via the constructor.
  std::vector<CorrMETData>::const_iterator i;
  for( i = corr_.begin(); i != corr_.end();  i++ ) 
    {
      corr.push_back( *i );
    }
}

//____________________________________________________________________________||
MET * MET::clone() const {
     return new MET( * this );
}

// function that calculates the MET significance from the vector information.
//____________________________________________________________________________||
double MET::significance() const {
  if(signif_dxx==0 && signif_dyy==0 && signif_dxy==0 && signif_dyx==0)
    return -1;
  TMatrixD metmat = getSignificanceMatrix();
  TVectorD metvec(2);
  metvec(0)=this->px();
  metvec(1)=this->py();
  double signif = -1;
  if(std::fabs(metmat.Determinant())>0.000001){
    metmat.Invert();
    signif = metvec * (metmat * metvec);
  }
  return signif;
}

// Returns the vector of all corrections applied to the x component of the
// missing transverse momentum, mEx
//____________________________________________________________________________||
std::vector<double> MET::dmEx() const 
{
  std::vector<double> deltas;
  std::vector<CorrMETData>::const_iterator i;
  for( i = corr.begin(); i != corr.end(); i++ )
    {
      deltas.push_back( i->mex );
    }
  return deltas;
}

// Returns the vector of all corrections applied to the y component of the
// missing transverse momentum, mEy
//____________________________________________________________________________||
std::vector<double> MET::dmEy() const 
{
  std::vector<double> deltas;
  std::vector<CorrMETData>::const_iterator i;
  for( i = corr.begin(); i != corr.end(); i++ )
    {
      deltas.push_back( i->mey );
    }
  return deltas;
}

// Returns the vector of all corrections applied to the scalar sum of the 
// transverse energy (over all objects)
//____________________________________________________________________________||
std::vector<double> MET::dsumEt() const 
{
  std::vector<double> deltas;
  std::vector<CorrMETData>::const_iterator i;
  for( i = corr.begin(); i != corr.end(); i++ )
    {
      deltas.push_back( i->sumet );
    }
  return deltas;
}

// returns the significance matrix
//____________________________________________________________________________||
TMatrixD MET::getSignificanceMatrix(void) const
{
  TMatrixD result(2,2);
  result(0,0)=signif_dxx;
  result(0,1)=signif_dxy;
  result(1,0)=signif_dyx;
  result(1,1)=signif_dyy;
  return result;
}

// Required RecoCandidate polymorphism
//____________________________________________________________________________||
bool MET::overlap( const Candidate & ) const 
{
  return false;
}

//____________________________________________________________________________||
void MET::setSignificanceMatrix(const TMatrixD &inmatrix)
{
  signif_dxx=inmatrix(0,0);
  signif_dxy=inmatrix(0,1);
  signif_dyx=inmatrix(1,0);
  signif_dyy=inmatrix(1,1);
  return;
}

//____________________________________________________________________________||
