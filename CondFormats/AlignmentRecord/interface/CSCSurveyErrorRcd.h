#ifndef CondFormats_AlignmentRecord_CSCSurveyErrorRcd_H
#define CondFormats_AlignmentRecord_CSCSurveyErrorRcd_H

/** \class CSCSurveyErrorRcd
 *
 *  DB record to hold errors of alignment parameters from survey.
 *
 *  $Date: 2007/07/06 16:01:02 $
 *  $Revision: 1.1 $
 *  \author Jim Pivarski
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class CSCSurveyErrorRcd:
  public edm::eventsetup::EventSetupRecordImplementation<CSCSurveyErrorRcd>
{
};

#endif
