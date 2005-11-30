#ifndef CALIBTRACKER_RECORDS_SISTRIPCONTROLCONNECTIVITY_H
#define CALIBTRACKER_RECORDS_SISTRIPCONTROLCONNECTIVITY_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     SiStripControlConnectivityRcd
// 
/**\class SiStripControlConnectivityRcd SiStripControlConnectivityRcd.h CalibTracker/Records/interface/SiStripControlConnectivityRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Wed Aug 10 08:13:43 CEST 2005
// $Id: SiStripControlConnectivityRcd.h,v 1.1 2005/11/29 14:18:35 gbruno Exp $
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/SiStripControlCablingRcd.h"
#include "boost/mpl/vector.hpp"

class SiStripControlConnectivityRcd : public edm::eventsetup::DependentRecordImplementation<SiStripControlConnectivityRcd,
  boost::mpl::vector<SiStripControlCablingRcd> > {};

#endif /* RECORDS_SISTRIPCONTROLCONNECTIVITY_H */

