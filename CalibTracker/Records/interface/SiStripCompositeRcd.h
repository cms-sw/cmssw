#ifndef CALIBTRACKER_RECORDS_SISTRIPCOMPOSITE_H
#define CALIBTRACKER_RECORDS_SISTRIPCOMPOSITE_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     SiStripCompositeRcd
// 
/**\class SiStripCompositeRcd SiStripCompositeRcd.h CalibTracker/Records/interface/SiStripCompositeRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Wed Aug 10 08:13:43 CEST 2005
// $Id: SiStripCompositeRcd.h,v 1.2 2005/08/11 17:51:47 dutta Exp $
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CalibFormats/Records/interface/SiStripReadoutConnectivityRcd.h"
#include "CalibFormats/Records/interface/SiStripControlConnectivityRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "boost/mpl/vector.hpp"

class SiStripCompositeRcd : public edm::eventsetup::DependentRecordImplementation<SiStripCompositeRcd,					         		  boost::mpl::vector<TrackerDigiGeometryRecord>,				  boost::mpl::vector<SiStripReadoutConnectivityRcd>, 
   boost::mpl::vector<SiStripControlConnectivityRcd> > {};

#endif /* RECORDS_SISTRIPCOMPOSITE_H */

