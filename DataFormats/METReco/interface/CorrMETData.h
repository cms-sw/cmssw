#ifndef METRECO_CORR_MET_DATA_H
#define METRECO_CORR_MET_DATA_H

/// \class CorrMETData
/// 
/// \short a MET correction term
/// 
/// CorrMETData represents a MET correction term.
/// 
/// \author Michael Schmitt, Richard Cavanaugh The University of Florida
/// 
/// \version $Id$

//____________________________________________________________________________||
struct CorrMETData
{

  double mex;
  double mey;
  double sumet;
  double significance;

  CorrMETData() : mex(0.0), mey(0.0), sumet(0.0), significance(0.0) { }

  CorrMETData(const CorrMETData& corr) : mex(corr.mex), mey(corr.mey), sumet(corr.sumet), significance(corr.significance) { }
};

//____________________________________________________________________________||
#endif // METRECO_CORR_MET_DATA_H
