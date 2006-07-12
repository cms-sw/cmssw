// File: MET.cc
// Description: see MET.h
// Author: Michael Schmitt, R. Cavanaugh University of Florida
// Creation Date:  MHS MAY 30, 2005 initial version

#include "DataFormats/METReco/interface/MET.h"

using namespace std;
using namespace reco;

//---------------------------------------------------------------------------
// Default Constructor;
//-----------------------------------
MET::MET()
{
  sumet = 0.0;
  elongit = 0.0;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
// Constructer for the case when only p4_ =  (mEx, mEy, 0, mEt) is known.
// The vertex information is currently not used (but may be in the future)
// and is required by the RecoCandidate constructer.
//-----------------------------------
MET::MET( const LorentzVector& p4_, const Point& vtx_ ) : 
  RecoCandidate( 0, p4_, vtx_ )
{
  sumet = 0.0;
  elongit = 0.0;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
// Constructer for the case when the SumET is known in addition to 
// p4_ = (mEx, mEy, 0, mEt).  The vertex information is currently not used
// (but see above).
//-----------------------------------
MET::MET( double sumet_, const LorentzVector& p4_, const Point& vtx_ ) : 
  RecoCandidate( 0, p4_, vtx_ ) 
{
  sumet = sumet_;
  elongit = 0.0;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
// Constructor for the case when the SumET, the corrections which
// were applied to the MET, as well the MET itself p4_ = (mEx, mEy, 0, mEt)
// are all known.  See above concerning the vertex information. 
//-----------------------------------
MET::MET( double sumet_, std::vector<CorrMETData> corr_, 
	  const LorentzVector& p4_, const Point& vtx_ ) : 
  RecoCandidate( 0, p4_, vtx_ ) 
{
  sumet = sumet_;
  elongit = 0.0;
  //-----------------------------------
  // Fill the vector containing the corrections (corr) with vector of 
  // known corrections (corr_) passed in via the constructor.
  std::vector<CorrMETData>::const_iterator i;
  for( i = corr_.begin(); i != corr_.end();  i++ ) 
    {
      corr.push_back( *i );
    }
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
// Returns the vector of all corrections applied to the x component of the
// missing transverse momentum, mEx
//-----------------------------------
std::vector<double> MET::dmEx()
{
  std::vector<double> deltas;
  std::vector<CorrMETData>::const_iterator i;
  for( i = corr.begin(); i != corr.end(); i++ )
    {
      deltas.push_back( i->mex );
    }
  return deltas;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
// Returns the vector of all corrections applied to the y component of the
// missing transverse momentum, mEy
//-----------------------------------
std::vector<double> MET::dmEy()
{
  std::vector<double> deltas;
  std::vector<CorrMETData>::const_iterator i;
  for( i = corr.begin(); i != corr.end(); i++ )
    {
      deltas.push_back( i->mey );
    }
  return deltas;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
// Returns the vector of all corrections applied to the scalar sum of the 
// transverse energy (over all objects)
//-----------------------------------
std::vector<double> MET::dsumEt()
{
  std::vector<double> deltas;
  std::vector<CorrMETData>::const_iterator i;
  for( i = corr.begin(); i != corr.end(); i++ )
    {
      deltas.push_back( i->sumet );
    }
  return deltas;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
// Required RecoCandidate polymorphism
//-----------------------------------
bool MET::overlap( const Candidate & ) const 
{
  return false;
}
//---------------------------------------------------------------------------
