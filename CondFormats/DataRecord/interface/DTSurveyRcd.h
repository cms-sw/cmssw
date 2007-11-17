#ifndef CondFormats_DataRecord_DTSurveyRcd_H
#define CondFormats_DataRecord_DTSurveyRcd_H

/** \class DTSurveyRcd
 *
 *  DB record to hold values of alignment parameters from survey.
 *
 *  $Date: Tue Jul  3 11:45:27 CEST 2007 $
 *  $Revision: 1.1 $
 *  \author Jim Pivarski
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class DTSurveyRcd:
  public edm::eventsetup::EventSetupRecordImplementation<DTSurveyRcd>
{
};

#endif
