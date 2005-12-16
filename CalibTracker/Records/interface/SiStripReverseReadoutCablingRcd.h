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
// $Id: SiStripReadoutConnectivityRcd.h,v 1.2 2005/11/30 22:16:40 dutta Exp $
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/SiStripReadoutCablingRcd.h"
#include "boost/mpl/vector.hpp"

class SiStripReverseReadoutCablingRcd : public edm::eventsetup::DependentRecordImplementation<SiStripReverseReadoutCablingRcd,
  boost::mpl::vector<SiStripReadoutCablingRcd> > {};

#endif /* RECORDS_SISTRIPREADOUTCONNECTIVITY_H */

