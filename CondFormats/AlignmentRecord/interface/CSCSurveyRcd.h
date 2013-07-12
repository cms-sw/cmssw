#ifndef CondFormats_AlignmentRecord_CSCSurveyRcd_H
#define CondFormats_AlignmentRecord_CSCSurveyRcd_H

/** \class CSCSurveyRcd
 *
 *  DB record to hold values of alignment parameters from survey.
 *
 *  $Date: 2007/07/06 16:01:02 $
 *  $Revision: 1.1 $
 *  \author Jim Pivarski
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class CSCSurveyRcd:
  public edm::eventsetup::EventSetupRecordImplementation<CSCSurveyRcd>
{
};

#endif
