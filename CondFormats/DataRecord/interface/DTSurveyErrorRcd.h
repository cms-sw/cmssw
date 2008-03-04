#ifndef CondFormats_DataRecord_DTSurveyErrorRcd_H
#define CondFormats_DataRecord_DTSurveyErrorRcd_H

/** \class DTSurveyErrorRcd
 *
 *  DB record to hold errors of alignment parameters from survey.
 *
 *  $Date: Tue Jul  3 11:45:51 CEST 2007 $
 *  $Revision: 1.1 $
 *  \author Jim Pivarski
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class DTSurveyErrorRcd:
  public edm::eventsetup::EventSetupRecordImplementation<DTSurveyErrorRcd>
{
};

#endif
