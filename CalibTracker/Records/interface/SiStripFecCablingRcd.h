#ifndef CALIBTRACKER_RECORDS_SISTRIPFECCABLING_H
#define CALIBTRACKER_RECORDS_SISTRIPFECCABLING_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     SiStripFecCablingRcd
// 
/**\class SiStripFecCablingRcd SiStripFecCablingRcd.h CalibTracker/Records/interface/SiStripFecCablingRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      dkcira

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "boost/mpl/vector.hpp"

class SiStripFecCablingRcd : public edm::eventsetup::DependentRecordImplementation<SiStripFecCablingRcd,
  boost::mpl::vector<SiStripFedCablingRcd> > {};

#endif /* CALIBTRACKER_RECORDS_SISTRIPFECCABLING_H */

