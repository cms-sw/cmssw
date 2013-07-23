#ifndef FastSimulation_ParticlePropagator_MagneticFieldMapRecord_h
#define FastSimulation_ParticlePropagator_MagneticFieldMapRecord_h
// -*- C++ -*-
//
// Package:     ParticlePropagator
// Class  :     MagneticFieldMapRecord
// 
/**\class MagneticFieldMapRecord MagneticFieldMapRecord.h FastSimulation/ParticlePropagator/interface/MagneticFieldMapRecord.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Wed Nov 21 12:14:34 CET 2007
// $Id: MagneticFieldMapRecord.h,v 1.1 2007/11/22 20:23:22 pjanot Exp $
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "boost/mpl/vector.hpp"

class MagneticFieldMapRecord : public edm::eventsetup::DependentRecordImplementation<MagneticFieldMapRecord, boost::mpl::vector<IdealMagneticFieldRecord,TrackerInteractionGeometryRecord> > {};

#endif
