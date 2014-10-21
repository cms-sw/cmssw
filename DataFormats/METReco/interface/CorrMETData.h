// -*- C++ -*-
#ifndef METRECO_CORR_MET_DATA_H
#define METRECO_CORR_MET_DATA_H

/// \class CorrMETData
/// 
/// \short a MET correction term
/// 
/// CorrMETData represents a MET correction term.
/// 
/// \author Michael Schmitt, Richard Cavanaugh The University of Florida; Tai Sakuma, Texas A&M University

//____________________________________________________________________________||
struct CorrMETData
{

  double mex;
  double mey;

  double sumet; // to be deleted

  CorrMETData() : mex(0.0), mey(0.0), sumet(0.0) { }

  CorrMETData(const CorrMETData& corr) : mex(corr.mex), mey(corr.mey), sumet(corr.sumet) { }

  CorrMETData& operator+=(const CorrMETData& rhs)
  {
    mex += rhs.mex;	 
    mey += rhs.mey;	 
    sumet += rhs.sumet;	 
    return *this;
  }

  CorrMETData& operator*=(const double& rhs)
  {
    mex *= rhs;
    mey *= rhs;	 
    sumet *= rhs;	 
    return *this;
  }

  friend CorrMETData operator+(const CorrMETData& lhs, const CorrMETData& rhs)
  {
    return CorrMETData(lhs) += rhs;
  }

  friend CorrMETData operator*(const double& lhs, const CorrMETData& rhs)
  {
    return CorrMETData(rhs) *= lhs;
  }

  friend CorrMETData operator*(const CorrMETData& lhs, const double& rhs)
  {
    return CorrMETData(lhs) *= rhs;
  }

};

//____________________________________________________________________________||
#endif // METRECO_CORR_MET_DATA_H
