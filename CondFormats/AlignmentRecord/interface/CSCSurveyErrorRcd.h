#ifndef CondFormats_AlignmentRecord_CSCSurveyErrorRcd_H
#define CondFormats_AlignmentRecord_CSCSurveyErrorRcd_H

/** \class CSCSurveyErrorRcd
 *
 *  DB record to hold errors of alignment parameters from survey.
 *
 *  $Date: 2007/10/18 07:31:48 $
 *  $Revision: 1.1 $
 *  \author Jim Pivarski
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class CSCSurveyErrorRcd:
  public edm::eventsetup::EventSetupRecordImplementation<CSCSurveyErrorRcd>
{
};

#endif
