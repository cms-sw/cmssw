#ifndef CALIBTRACKER_RECORDS_SISTRIPDETCABLING_H
#define CALIBTRACKER_RECORDS_SISTRIPDETCABLING_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     SiStripDetCablingRcd
// 
/**\class SiStripDetCablingRcd SiStripDetCablingRcd.h CalibTracker/Records/interface/SiStripDetCablingRcd.h

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

class SiStripDetCablingRcd : public edm::eventsetup::DependentRecordImplementation<SiStripDetCablingRcd,
  boost::mpl::vector<SiStripFedCablingRcd> > {};

#endif /* CALIBTRACKER_RECORDS_SISTRIPDETCABLING_H */

