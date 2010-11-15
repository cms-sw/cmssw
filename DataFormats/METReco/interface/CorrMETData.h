#ifndef METRECO_CORR_MET_DATA_H
#define METRECO_CORR_MET_DATA_H

 /** \class CorrMETData
 *
 * \short Structure containing data common to all types of MET
 *
 * CorrMETData holds correction information for all types of MET.
 *
 * \author Michael Schmitt, Richard Cavanaugh The University of Florida
 *
 * \version   1st Version June 14, 2005.
 *
 ************************************************************/

#include <vector>

//const int MET_LABEL_LEN = 24;

struct CorrMETData {

  //char label[MET_LABEL_LEN];

  double mex;
  double mey;
  double sumet;
  double significance;

  CorrMETData() {
    mex=0.;
    mey=0.;
    sumet=0.;
    significance=0.;
  }
};

#endif // METRECO_CORR_MET_DATA_H
