// File: MET.cc
// Description: see MET.h
// Author: Michael Schmitt, R. Cavanaugh University of Florida
// Creation Date:  MHS MAY 30, 2005 initial version

#include "DataFormats/METReco/interface/MET.h"

using namespace std;
using namespace reco;

MET::MET()
{
  sumet = 0.0;
}

MET::MET( const LorentzVector& p4_, const Point& vtx_ ) : RecoCandidate( 0, p4_, vtx_ )
{
  sumet = 0.0;
}

MET::MET( double sumet_, const LorentzVector& p4_, const Point& vtx_ ) : RecoCandidate( 0, p4_, vtx_ ) 
{
  sumet = sumet_;
}

MET::MET( double sumet_, std::vector<CorrMETData> corr_, const LorentzVector& p4_, const Point& vtx_ ) : RecoCandidate( 0, p4_, vtx_ ) 
{
  sumet = sumet_;

  std::vector<CorrMETData>::const_iterator i;
  for( i = corr_.begin(); i != corr_.end();  i++ ) 
    {
      corr.push_back( *i );
    }
}

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

bool MET::overlap( const Candidate & ) const 
{
  return false;
}
