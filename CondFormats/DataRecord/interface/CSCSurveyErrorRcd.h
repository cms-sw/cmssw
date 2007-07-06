#ifndef CondFormats_DataRecord_CSCSurveyErrorRcd_H
#define CondFormats_DataRecord_CSCSurveyErrorRcd_H

/** \class CSCSurveyErrorRcd
 *
 *  DB record to hold errors of alignment parameters from survey.
 *
 *  $Date: Tue Jul  3 11:46:26 CEST 2007 $
 *  $Revision: 1.1 $
 *  \author Jim Pivarski
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class CSCSurveyErrorRcd:
  public edm::eventsetup::EventSetupRecordImplementation<CSCSurveyErrorRcd>
{
};

#endif
