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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Tracker Geometry/Topology  stuff
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// C++ stuff
#include <cassert>
#include <cstdint>
#include <iostream>
#include <iomanip>

// WTH is this needed? clang wants it for linking...
const GeometryInterface::Value GeometryInterface::UNDEFINED;

void GeometryInterface::load(edm::EventSetup const& iSetup) {
  //loadFromAlignment(iSetup, iConfig);
  loadFromTopology(iSetup, iConfig);
  loadTimebased(iSetup, iConfig);
  loadModuleLevel(iSetup, iConfig);
  loadFEDCabling(iSetup, iConfig);
  edm::LogInfo log("GeometryInterface");
  log << "Known colum names:\n";
  for (auto e : ids) log << "+++ column: " << e.first 
    << " ok " << bool(extractors[e.second]) << " max " << max_value[e.second] << "\n";
  is_loaded = true;
}

#if 0
// Alignment stuff
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"

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
    void traverseAlignables(Alignable* compositeAlignable, std::map<align::StructureType, BitInfo>& infos, std::vector<InterestingQuantities>& all_modules) {
      //edm::LogTrace("GeometryInterface") << "+++++ Alignable " << AlignableObjectId::idToString(compositeAlignable->alignableObjectId()) << " " << std::hex << compositeAlignable->id() << "\n";
      auto alignable = compositeAlignable;

      if(alignable->alignableObjectId() == align::AlignableDetUnit) {
	// this is a Module
	all_modules.push_back(InterestingQuantities{.sourceModule = alignable->id()});
      }

      auto& info = infos[alignable->alignableObjectId()];
      // default values -- sort of a constructor for BitInfo
      if (info.characteristicBits == 0) {
	info.characteristicBits = alignable->id();
	info.characteristicMask = 0xFFFFFFFF;
	info.variableMask = 0x0;
      } 

      // variable mask must be local to the hierarchy
      if (info.currenParent != alignable->mother() || !alignable->mother()) {
	info.currenParent = alignable->mother();
	info.variableBase = alignable->id();
      } else {
	// ^ gives changed bits, | to collect all ever changed
	info.variableMask |= info.variableBase ^ compositeAlignable->id();
      }

      auto leafVariableMask = info.variableMask;
      // climb up the hierarchy and widen characteristics.
      for (auto alignable = compositeAlignable; alignable; alignable = alignable->mother()) {
	auto& info = infos[alignable->alignableObjectId()];
	// ^ gives changed bits, ~ unchanged, & to collect all always  unchanged
	info.characteristicMask &= ~(info.characteristicBits ^ compositeAlignable->id());
	// sometimes we have "noise" in the lower bits and the higher levels claim 
	// variable bits that belong to lower elements. Clear these.
	if (info.variableMask != leafVariableMask) info.variableMask &= ~leafVariableMask;
      }
      for (auto* alignable : compositeAlignable->components()) {
	traverseAlignables(alignable, infos, all_modules);
      }
    }
  } alignableIterator;

  alignableIterator.traverseAlignables(trackerAlignables, infos, all_modules);

  for (auto el : infos) {
    auto info = el.second;
    auto type = AlignableObjectId::idToString(el.first);
    // the characteristicBits that are masked out don't matter,
    // to normalize we set them to 0.
    info.characteristicBits &= info.characteristicMask;

    edm::LogInfo("GeometryInterface") << std::hex << std::setfill('0') << std::setw(8)
              << "+++ Type " << info.characteristicBits << " "
                                  << info.characteristicMask << " "
                                  << info.variableMask << " " << type << "\n";
    int variable_shift = 0;
    while (info.variableMask && ((info.variableMask >> variable_shift) & 1) == 0) variable_shift++;
    addExtractor(
      intern(type),
      [info, variable_shift] (InterestingQuantities const& iq) {
	uint32_t id = iq.sourceModule.rawId();
	if ((id & info.characteristicMask) == (info.characteristicBits & info.characteristicMask)) {
	  uint32_t pos = (id & info.variableMask) >> variable_shift;
	  return Value(pos); 
	} else {
	  return Value(UNDEFINED);
	}
      },
      info.variableMask >> variable_shift
    );

  }
  delete trackerAlignables;
}
#endif

void GeometryInterface::loadFromTopology(edm::EventSetup const& iSetup, const edm::ParameterSet& iConfig) {
  // Get a Topology
  edm::ESHandle<TrackerTopology> trackerTopologyHandle;
  iSetup.get<TrackerTopologyRcd>().get(trackerTopologyHandle);
  assert(trackerTopologyHandle.isValid());

  for (std::pair<std::string, TrackerTopology::BitMask> e : trackerTopologyHandle->namedPartitions()) {
    auto mask = e.second;
    addExtractor(intern(e.first),
      [mask] (InterestingQuantities const& iq) {
	if(!mask.valid(iq.sourceModule)) return UNDEFINED;
	return Value(mask.apply(iq.sourceModule));
      },
      mask.mask_ 
    );
  }
  
  // Get a Geometry
  edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometryHandle);
  assert(trackerGeometryHandle.isValid());
  // TODO: Just pixel would be fine for now.
  auto detids = trackerGeometryHandle->detIds();
  for (DetId id : detids) {
    // for ROCs etc., they eed to be added here as well.
    all_modules.push_back(InterestingQuantities{.sourceModule = id });
  }
}
 
void GeometryInterface::loadTimebased(edm::EventSetup const& iSetup, const edm::ParameterSet& iConfig) {
  // extractors for quantities that are roughly time-based. We cannot book plots based on these; they have to
  // be grouped away in step1.
  addExtractor(intern("Lumisection"),
    [] (InterestingQuantities const& iq) {
      if(!iq.sourceEvent) return UNDEFINED;
      return Value(iq.sourceEvent->luminosityBlock());
    },
    5000
  );
  addExtractor(intern("BX"),
    [] (InterestingQuantities const& iq) {
      if(!iq.sourceEvent) return UNDEFINED;
      return Value(iq.sourceEvent->bunchCrossing());
    },
    3600 // TODO: put actual max. BX
  );
}

void GeometryInterface::loadModuleLevel(edm::EventSetup const& iSetup, const edm::ParameterSet& iConfig) {
  // stuff that is within modules. Might require some phase0/phase1/strip switching later
  addExtractor(intern("row"),
    [] (InterestingQuantities const& iq) {
      return Value(iq.row);
    },
    200 //TODO: use actual number of rows here. Where can we find that?
  );
  addExtractor(intern("col"),
    [] (InterestingQuantities const& iq) {
      return Value(iq.col);
    },
    200 //TODO: use actual number of cols here. Where can we find that?
  );

  addExtractor(intern("DetId"),
    [] (InterestingQuantities const& iq) {
      uint32_t id = iq.sourceModule.rawId();
      return Value(id);
    },
    0 // No sane value possible here.
  );

  // TODO: ROCs ans stuff here.
}

void GeometryInterface::loadFEDCabling(edm::EventSetup const& iSetup, const edm::ParameterSet& iConfig) {
  edm::ESHandle<SiPixelFedCablingMap> theCablingMap;
  iSetup.get<SiPixelFedCablingMapRcd>().get(theCablingMap);
  std::map<DetId, Value> fedmap;

  if (theCablingMap.isValid()) {
    auto map = theCablingMap.product();

    for(auto iq : all_modules) {
      std::vector<sipixelobjects::CablingPathToDetUnit> paths = map->pathToDetUnit(iq.sourceModule.rawId());
      for (auto p : paths) {
	//std::cout << "+++ cabling " << iq.sourceModule.rawId() << " " << p.fed << " " << p.link << " " << p.roc << "\n";
	fedmap[iq.sourceModule] = Value(p.fed);
      }
    }
  } else {
    edm::LogError("GeometryInterface") << "+++ No cabling map. Cannot extract FEDs.\n";
  }

  addExtractor(intern("FED"),
    [fedmap] (InterestingQuantities const& iq) {
      auto it = fedmap.find(iq.sourceModule);
      if (it == fedmap.end()) return GeometryInterface::UNDEFINED;
      return it->second;
    },
    1255
  );
}
