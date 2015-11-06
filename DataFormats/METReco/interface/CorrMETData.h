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
  double significance; // to be deleted

  CorrMETData() : mex(0.0), mey(0.0), sumet(0.0), significance(0.0) { }

  CorrMETData(const CorrMETData& corr) : mex(corr.mex), mey(corr.mey), sumet(corr.sumet), significance(corr.significance) { }

  CorrMETData& operator+=(const CorrMETData& rhs)
  {
    mex += rhs.mex;	 
    mey += rhs.mey;	 
    sumet += rhs.sumet;	 
    significance += rhs.significance;
    return *this;
  }

  friend CorrMETData operator+(const CorrMETData& lhs, const CorrMETData& rhs)
  {
    CorrMETData ret(lhs);
    ret += rhs;
    return ret;
  }

  friend CorrMETData operator*(const double& lhs, const CorrMETData& rhs)
  {
    CorrMETData ret;
    ret.mex = lhs*rhs.mex;
    ret.mey = lhs*rhs.mey;
    ret.sumet = lhs*rhs.sumet;
    ret.significance = lhs*rhs.significance;
    return ret;
  }

};

//____________________________________________________________________________||
#endif // METRECO_CORR_MET_DATA_H
