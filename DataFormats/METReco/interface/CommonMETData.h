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
  CommonMETData() :met(0), mex(0), mey(0), mez(0), sumet(0), phi(0) {}
  double met;
  double mex;
  double mey;
  double mez;
  double sumet;
  double phi; // MM: used in mva/noPU MET
};

//____________________________________________________________________________||
#endif // METRECO_COMMON_MET_DATA_H
