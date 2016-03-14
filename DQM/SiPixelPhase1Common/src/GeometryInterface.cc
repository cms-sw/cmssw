// -*- C++ -*-
//
// Package:    SiPixelPhase1Common
// Class:      GeometryInterface
// 
// Geometry depedence goes here. 
//
// Original Author:  Marcel Schneider

#include "DQM/SiPixelPhase1Common/interface/GeometryInterface.h"

GeometryInterface GeometryInterface::instance;

void GeometryInterface::load(edm::EventSetup const& iSetup, const edm::ParameterSet& iConfig) {
  is_loaded = true;
}

