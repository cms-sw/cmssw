/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
****************************************************************************/

#ifndef CalibPPS_AlignmentRelative_Utilities_h
#define CalibPPS_AlignmentRelative_Utilities_h

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include <TMatrixD.h>

class CTPPSRPAlignmentCorrectionsData;
class AlignmentGeometry;

extern void printId(unsigned int id);

extern void print(TMatrixD &m, const char *label = nullptr, bool mathematicaFormat = false);

/**
 * NOTE ON ERROR PROPAGATION
 *
 * It is not possible to split (and merge again) the experimental errors between the RP and sensor
 * contributions. To do so, one would need to keep the entire covariance matrix. Thus, it has been
 * decided to save:
 *   RP errors = the uncertainty of the common shift/rotation
 *   sensor error = the full experimental uncertainty
 * In consequence: RP and sensor errors SHOULD NEVER BE SUMMED!
 **/
extern void factorRPFromSensorCorrections(const CTPPSRPAlignmentCorrectionsData &input,
                                          CTPPSRPAlignmentCorrectionsData &expanded,
                                          CTPPSRPAlignmentCorrectionsData &factored,
                                          const AlignmentGeometry &,
                                          bool equalWeights = false,
                                          unsigned int verbosity = 0);

#endif
