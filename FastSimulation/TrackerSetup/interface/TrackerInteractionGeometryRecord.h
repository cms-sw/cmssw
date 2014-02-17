#ifndef FastSimulation_TrackerSetup_TrackerInteractionGeometryRecord_h
#define FastSimulation_TrackerSetup_TrackerInteractionGeometryRecord_h
// -*- C++ -*-
//
// Package:     TrackerSetup
// Class  :     TrackerInteractionGeometryRecord
// 
/**\class TrackerInteractionGeometryRecord TrackerInteractionGeometryRecord.h FastSimulation/TrackerSetup/interface/TrackerInteractionGeometryRecord.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Wed Nov 21 12:14:34 CET 2007
// $Id: TrackerInteractionGeometryRecord.h,v 1.1 2007/11/22 08:30:04 pjanot Exp $
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "boost/mpl/vector.hpp"

class TrackerInteractionGeometryRecord : public edm::eventsetup::DependentRecordImplementation<TrackerInteractionGeometryRecord, boost::mpl::vector<TrackerRecoGeometryRecord> > {};

#endif
