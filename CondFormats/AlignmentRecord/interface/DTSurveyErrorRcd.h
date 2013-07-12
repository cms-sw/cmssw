#ifndef CondFormats_AlignmentRecord_DTSurveyErrorRcd_H
#define CondFormats_AlignmentRecord_DTSurveyErrorRcd_H

/** \class DTSurveyErrorRcd
 *
 *  DB record to hold errors of alignment parameters from survey.
 *
 *  $Date: 2007/07/06 16:01:02 $
 *  $Revision: 1.1 $
 *  \author Jim Pivarski
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class DTSurveyErrorRcd:
  public edm::eventsetup::EventSetupRecordImplementation<DTSurveyErrorRcd>
{
};

#endif
