#ifndef CALIBTRACKER_RECORDS_SISTRIPSTRUCTURE_H
#define CALIBTRACKER_RECORDS_SISTRIPSTRUCTURE_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     SiStripStructureRcd
// 
/**\class SiStripStructureRcd SiStripStructureRcd.h CalibTracker/Records/interface/SiStripStructureRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Sat Dec 31 12:01:43 CEST 2005
// $Id: SiStripStructureRcd.h,v 1.1 2005/12/31 12:35:09 gbruno Exp $
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/SiStripReadoutCablingRcd.h"
#include "boost/mpl/vector.hpp"

class SiStripStructureRcd : public edm::eventsetup::DependentRecordImplementation<SiStripStructureRcd,
  boost::mpl::vector<SiStripReadoutCablingRcd> > {};

#endif /* RECORDS_SISTRIPSTRUCTURE_H */

