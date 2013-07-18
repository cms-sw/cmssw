// -*- C++ -*-
//
// Package:     PatAlgos
// Class  :     KinematicResolutionRcd
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      
// Created:     Sun Jun 24 16:53:34 CEST 2007
// $Id: KinematicResolutionRcd.cc,v 1.3 2010/02/20 21:00:30 wmtan Exp $

#include "PhysicsTools/PatAlgos/interface/KinematicResolutionRcd.h"
#include "PhysicsTools/PatAlgos/interface/KinematicResolutionProvider.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"

TYPELOOKUP_DATA_REG(KinematicResolutionProvider);
EVENTSETUP_RECORD_REG(KinematicResolutionRcd);
