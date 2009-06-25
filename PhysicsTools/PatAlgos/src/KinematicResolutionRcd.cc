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
// $Id: KinematicResolutionRcd.cc,v 1.1.2.1 2009/04/30 09:11:47 gpetrucc Exp $

#include "PhysicsTools/PatAlgos/interface/KinematicResolutionRcd.h"
#include "PhysicsTools/PatAlgos/interface/KinematicResolutionProvider.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"

EVENTSETUP_DATA_REG(KinematicResolutionProvider);
EVENTSETUP_RECORD_REG(KinematicResolutionRcd);
