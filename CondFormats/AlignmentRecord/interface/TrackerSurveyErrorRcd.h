#ifndef CondFormats_AlignmentRecord_TrackerSurveyErrorRcd_H
#define CondFormats_AlignmentRecord_TrackerSurveyErrorRcd_H

/** \class TrackerSurveyErrorRcd
 *
 *  DB record to hold errors of alignment parameters from survey.
 *
 *  $Date: 2007/04/03 15:24:56 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class TrackerSurveyErrorRcd:
  public edm::eventsetup::EventSetupRecordImplementation<TrackerSurveyErrorRcd>
{
};

#endif
