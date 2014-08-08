// -*- C++ -*-
#ifndef METRECO_COMMON_MET_DATA_H
#define METRECO_COMMON_MET_DATA_H

/// \class CommonMETData
///
/// \short Structure containing data common to all types of MET
///
/// \author Michael Schmitt, Richard Cavanaugh The University of Florida

//____________________________________________________________________________||
struct CommonMETData
{
  double met;
  double mex;
  double mey;
  double mez;
  double sumet;
  double phi; // MM: used in mva/noPU MET
};

//____________________________________________________________________________||
#endif // METRECO_COMMON_MET_DATA_H
