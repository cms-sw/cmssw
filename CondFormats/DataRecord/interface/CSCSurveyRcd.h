#ifndef CondFormats_DataRecord_CSCSurveyRcd_H
#define CondFormats_DataRecord_CSCSurveyRcd_H

/** \class CSCSurveyRcd
 *
 *  DB record to hold values of alignment parameters from survey.
 *
 *  $Date: Tue Jul  3 11:46:08 CEST 2007 $
 *  $Revision: 1.1 $
 *  \author Jim Pivarski
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class CSCSurveyRcd:
  public edm::eventsetup::EventSetupRecordImplementation<CSCSurveyRcd>
{
};

#endif
