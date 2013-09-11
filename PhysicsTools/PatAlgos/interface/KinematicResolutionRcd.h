#ifndef PatAlgos_KinematicResolutionRcd_h
#define PatAlgos_KinematicResolutionRcd_h
// -*- C++ -*-
//
// Package:     PatAlgos
// Class  :     KinematicResolutionRcd
// 
/**\class KinematicResolutionRcd ParticleResolutionRcd.h PhysicsTools/PatAlgos/interface/ParticleResolutionRcd.h

 Description: Interface for getting Kinematic Resolutions through EventSetup

*/
//
// Author:      
// Created:     Sun Jun 24 16:53:34 CEST 2007
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"


#include "boost/mpl/vector.hpp"


class KinematicResolutionRcd : 
    public edm::eventsetup::DependentRecordImplementation<
            KinematicResolutionRcd,
            boost::mpl::vector<IdealMagneticFieldRecord>
        > {};
#endif
