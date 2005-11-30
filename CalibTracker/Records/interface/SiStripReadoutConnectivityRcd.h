#ifndef CALIBTRACKER_RECORDS_SISTRIPREADOUTCONNECTIVITY_H
#define CALIBTRACKER_RECORDS_SISTRIPREADOUTCONNECTIVITY_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     SiStripReadoutConnectivityRcd
// 
/**\class SiStripReadoutConnectivityRcd SiStripReadoutConnectivityRcd.h CalibTracker/Records/interface/SiStripReadoutConnectivityRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Wed Aug 10 08:13:43 CEST 2005
// $Id: SiStripReadoutConnectivityRcd.h,v 1.1 2005/11/29 14:18:35 gbruno Exp $
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/SiStripReadOutCablingRcd.h"
#include "boost/mpl/vector.hpp"

class SiStripReadoutConnectivityRcd : public edm::eventsetup::DependentRecordImplementation<SiStripReadoutConnectivityRcd,
  boost::mpl::vector<SiStripReadOutCablingRcd> > {};

#endif /* RECORDS_SISTRIPREADOUTCONNECTIVITY_H */

