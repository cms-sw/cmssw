#ifndef CondFormats_DataRecord_TrackerSurveyErrorRcd_H
#define CondFormats_DataRecord_TrackerSurveyErrorRcd_H

/** \class TrackerSurveyErrorRcd
 *
 *  DB record to hold errors of alignment parameters from survey.
 *
 *  $Date: 2007/03/22 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class TrackerSurveyErrorRcd:
  public edm::eventsetup::EventSetupRecordImplementation<TrackerSurveyErrorRcd>
{
};

#endif
