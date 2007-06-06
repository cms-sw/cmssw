#ifndef Records_SiStripGainRcd_h
#define Records_SiStripGainRcd_h
// -*- C++ -*-
//
// Package:     Records
// Class  :     SiStripGainRcd
// 
/**\class SiStripGainRcd SiStripGainRcd.h CalibTracker/Records/interface/SiStripGainRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Fri Apr 27 14:25:40 CEST 2007
// $Id$
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include "boost/mpl/vector.hpp"

class SiStripGainRcd : public edm::eventsetup::DependentRecordImplementation<SiStripGainRcd, boost::mpl::vector<SiStripApvGainRcd> > {};

#endif
