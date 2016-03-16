// -*- C++ -*-
//
// Package:    SiPixelPhase1Common
// Class:      GeometryInterface
// 
// Geometry depedence goes here. 
//
// Original Author:  Marcel Schneider

#include "DQM/SiPixelPhase1Common/interface/GeometryInterface.h"

// edm stuff
#include "FWCore/Framework/interface/ESHandle.h"

// Tracker Geometry/Topology  suff
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"

// Alignment stuff
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

// C++ stuff
#include <cassert>

GeometryInterface GeometryInterface::instance;

void GeometryInterface::load(edm::EventSetup const& iSetup, const edm::ParameterSet& iConfig) {
  loadFromAlignment(iSetup, iConfig);
  is_loaded = true;
}


void GeometryInterface::loadFromAlignment(edm::EventSetup const& iSetup, const edm::ParameterSet& iConfig) {
  // This function collects information about ALignables, which make up most of
  // the Tracker Geometry. Alignables provide a way to explore the tracker as a
  // hierarchy, but there is no easy (and fast) way to find out which partitions
  // a DetId belongs to (we can see that it is in a Ladder, but not in which
  // Ladder). To work around this, we use the fact that DetIds are highly 
  // structured and work out the bit patterns that have waht we need.
  
  // Get a Topology
  edm::ESHandle<TrackerTopology> trackerTopologyHandle;
  iSetup.get<TrackerTopologyRcd>().get(trackerTopologyHandle);
  auto trackerTopology = trackerTopologyHandle.product();
  
  // Make a Geomtry
  edm::ESHandle<GeometricDet> geometricDet;
  iSetup.get<IdealGeometryRecord>().get(geometricDet);
  edm::ESHandle<PTrackerParameters> trackerParams;
  iSetup.get<PTrackerParametersRcd>().get(trackerParams);
  TrackerGeomBuilderFromGeometricDet trackerGeometryBuilder;
  auto trackerGeometry = trackerGeometryBuilder.build(&(*geometricDet), *trackerParams, trackerTopology);

  assert(trackerGeometry);
  assert(trackerTopology);
  std::cout << "Ptrs: " << trackerGeometry << " " << trackerTopology << "\n";
  AlignableTracker* trackerAlignables = new AlignableTracker(trackerGeometry, trackerTopology);
  assert(trackerAlignables);

  struct {
    void traverseAlignables(Alignable* compositeAlignable) {
      for (auto* alignable : compositeAlignable->components()) {
	auto type = AlignableObjectId::idToString(alignable->alignableObjectId());
	std::cout << "+++ Alignable: " << type << "\n";
	if (alignable->components().size() > 0) {
	  traverseAlignables(alignable);
	}
      }
    }
  } alignabelIterator;

  alignabelIterator.traverseAlignables(trackerAlignables);

  delete trackerAlignables;
}

