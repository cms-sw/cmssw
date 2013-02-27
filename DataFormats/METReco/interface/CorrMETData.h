// -*- C++ -*-
// $Id: CorrMETData.h,v 1.5 2012/09/10 18:21:56 sakuma Exp $
#ifndef METRECO_CORR_MET_DATA_H
#define METRECO_CORR_MET_DATA_H

/// \class CorrMETData
/// 
/// \short a MET correction term
/// 
/// CorrMETData represents a MET correction term.
/// 
/// \author Michael Schmitt, Richard Cavanaugh The University of Florida

//____________________________________________________________________________||
struct CorrMETData
{

  double mex;
  double mey;

  double sumet; // to be deleted
  double significance; // to be deleted

  CorrMETData() : mex(0.0), mey(0.0), sumet(0.0), significance(0.0) { }

  CorrMETData(const CorrMETData& corr) : mex(corr.mex), mey(corr.mey), sumet(corr.sumet), significance(corr.significance) { }
};

//____________________________________________________________________________||
#endif // METRECO_CORR_MET_DATA_H
