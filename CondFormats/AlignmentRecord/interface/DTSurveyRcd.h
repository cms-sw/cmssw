#ifndef CondFormats_AlignmentRecord_DTSurveyRcd_H
#define CondFormats_AlignmentRecord_DTSurveyRcd_H

/** \class DTSurveyRcd
 *
 *  DB record to hold values of alignment parameters from survey.
 *
 *  $Date: 2007/10/18 07:31:48 $
 *  $Revision: 1.1 $
 *  \author Jim Pivarski
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class DTSurveyRcd:
  public edm::eventsetup::EventSetupRecordImplementation<DTSurveyRcd>
{
};

#endif
