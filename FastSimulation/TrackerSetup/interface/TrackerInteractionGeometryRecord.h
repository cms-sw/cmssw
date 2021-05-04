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
// $Id$
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

class TrackerInteractionGeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<TrackerInteractionGeometryRecord,
                                                            edm::mpl::Vector<TrackerRecoGeometryRecord> > {};

#endif
