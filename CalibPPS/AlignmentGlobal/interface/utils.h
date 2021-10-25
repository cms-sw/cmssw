/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*  Mateusz Kocot (mateuszkocot99@gmail.com)
****************************************************************************/

#ifndef CalibPPS_AlignmentGlobal_utils_h
#define CalibPPS_AlignmentGlobal_utils_h

#include "TProfile.h"

namespace alig_utils {

  // Fits a linear function to a TProfile.
  int fitProfile(TProfile* p,
                 double x_mean,
                 double x_rms,
                 unsigned int fitProfileMinBinEntries,
                 unsigned int fitProfileMinNReasonable,
                 double& sl,
                 double& sl_unc);

}  // namespace alig_utils

#endif
