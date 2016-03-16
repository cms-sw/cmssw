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
#include <cstdint>
#include <iostream>
#include <iomanip>

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
  AlignableTracker* trackerAlignables = new AlignableTracker(trackerGeometry, trackerTopology);
  assert(trackerAlignables);

  struct BitInfo {
    uint32_t characteristicBits;
    uint32_t characteristicMask;
    uint32_t variableBase;
    uint32_t variableMask;
    Alignable* currenParent;
  };

  std::map<align::StructureType, BitInfo> infos;

  struct {
    void traverseAlignables(Alignable* compositeAlignable, std::map<align::StructureType, BitInfo>& infos) {
      std::cout << "+++++ Alignable " << AlignableObjectId::idToString(compositeAlignable->alignableObjectId()) << "\n";
	
      auto alignable = compositeAlignable;
      auto& info = infos[alignable->alignableObjectId()];
      if (info.characteristicBits == 0) {
	info.characteristicBits = alignable->id();
	info.characteristicMask = 0xFFFFFFFF;
	info.variableMask = 0x0;
      } 
      // climb up the hierarchy and widen characteristics.
      for (auto alignable = compositeAlignable; alignable; alignable = alignable->mother()) {
	auto& info = infos[alignable->alignableObjectId()];
	// ^ gives changed bits, ~ unchanged, & to collect all always  unchanged
	info.characteristicMask &= ~(info.characteristicBits ^ compositeAlignable->id());
      }
      // variable mask must be local to the hierarchy
      if (info.currenParent != alignable->mother() || !alignable->mother()) {
	info.currenParent = alignable->mother();
	info.variableBase = alignable->id();
      } else {
	// ^ gives changed bits, | to collect all ever changed
	info.variableMask |= info.variableBase ^ alignable->id();
      }

      for (auto* alignable : compositeAlignable->components()) {
	// for e.g. full Pixel, there is no variable part. so wee need to look at 
	// children to see where characteristic ends.
	//info.characteristicMask &= ~(info.characteristicBits ^ alignable->id());
	traverseAlignables(alignable, infos);
      }
    }
  } alignableIterator;

  alignableIterator.traverseAlignables(trackerAlignables, infos);

  for (auto el : infos) {
    auto info = el.second;
    auto type = AlignableObjectId::idToString(el.first);
    // since we did not update the characteristic for all children,
    // we have false mask bits in the lsbs.
    // We assume a left-to right ordererd hierarchy ad kill the
    // characteristic bits from the left.
    // +1 flips all bits up to (including) the first 0 from the right.
    //info.characteristicMask &= (info.characteristicMask + 1);
    // the characteristicBits that are masked out don't matter,
    // to normalize we set them to 0.
    info.characteristicBits &= info.characteristicMask;

    std::cout << std::hex << std::setfill('0') << std::setw(8)
              << "+++ Type " << info.characteristicBits << " "
                                  << info.characteristicMask << " "
                                  << info.variableMask << " " << type << "\n";
  }
  delete trackerAlignables;
}

