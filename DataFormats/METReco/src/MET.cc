// File: BaseMET.cc
// Description: see BaseMET.h
// Author: Michael Schmitt, R. Cavanaugh University of Florida
// Creation Date:  MHS MAY 30, 2005 initial version

#include "DataFormats/METReco/interface/MET.h"

using namespace std;
using namespace reco;

MET::MET()
{
  data.mex   = 0.0;
  data.mey   = 0.0;
  data.mez   = 0.0;
  data.met   = 0.0;
  data.sumet = 0.0;
  data.phi   = 0.0;
}

MET::MET( double mex, double mey )
{
  data.mex   = mex;
  data.mey   = mey;
  data.mez   = 0.0;
  data.met   = sqrt( mex*mex + mey*mey );
  data.sumet = 0.0;
  data.phi   = atan2( mey, mex );
}

MET::MET( double mex, double mey, double sumet )
{
  data.mex   = mex;
  data.mey   = mey;
  data.mez   = 0.0;
  data.met   = sqrt( mex*mex + mey*mey );
  data.sumet = sumet;
  data.phi   = atan2( mey, mex );
}

MET::MET( double mex, double mey, double sumet, double mez )
{
  data.mex   = mex;
  data.mey   = mey;
  data.mez   = mez;
  data.met   = sqrt( mex*mex + mey*mey );
  data.sumet = sumet;
  data.phi   = atan2( mey, mex );
}

MET::MET( CommonMETData data_ ) 
{
  data.mex   = data_.mex;
  data.mey   = data_.mey;
  data.mez   = data_.mez;
  data.met   = data_.met;
  data.sumet = data_.sumet;
  data.phi   = data_.phi;
}

MET::MET( CommonMETData data_, std::vector<CommonMETData> corr_ ) 
{
  data.mex   = data_.mex;
  data.mey   = data_.mey;
  data.mez   = data_.mez;
  data.met   = data_.met;
  data.sumet = data_.sumet;
  data.phi   = data_.phi;

  std::vector<CommonMETData>::const_iterator i;
  for( i = corr_.begin(); i != corr_.end();  i++ ) 
    {
      corr.push_back( *i );
    }
}

std::vector<double> MET::dmEt()
{
  std::vector<double> deltas;
  std::vector<CommonMETData>::const_iterator i;
  for( i = corr.begin(); i != corr.end(); i++ )
    {
      deltas.push_back( i->met );
    }
  return deltas;
}

std::vector<double> MET::dmEx()
{
  std::vector<double> deltas;
  std::vector<CommonMETData>::const_iterator i;
  for( i = corr.begin(); i != corr.end(); i++ )
    {
      deltas.push_back( i->mex );
    }
  return deltas;
}

std::vector<double> MET::dmEy()
{
  std::vector<double> deltas;
  std::vector<CommonMETData>::const_iterator i;
  for( i = corr.begin(); i != corr.end(); i++ )
    {
      deltas.push_back( i->mey );
    }
  return deltas;
}

std::vector<double> MET::dsumEt()
{
  std::vector<double> deltas;
  std::vector<CommonMETData>::const_iterator i;
  for( i = corr.begin(); i != corr.end(); i++ )
    {
      deltas.push_back( i->sumet );
    }
  return deltas;
}

std::vector<double> MET::dphi()
{
  std::vector<double> deltas;
  std::vector<CommonMETData>::const_iterator i;
  for( i = corr.begin(); i != corr.end(); i++ )
    {
      deltas.push_back( i->phi );
    }
  return deltas;
}
