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
    << " ok " << bool(extractors[e.second]) << " min " << min_value[e.second] << " max " << max_value[e.second] << "\n";
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

  std::vector<ID> geomquantities;

  for (std::pair<std::string, TrackerTopology::BitMask> e : trackerTopologyHandle->namedPartitions()) {
    auto mask = e.second;
    geomquantities.push_back(intern(e.first));
    addExtractor(intern(e.first),
      [mask] (InterestingQuantities const& iq) {
	if(!mask.valid(iq.sourceModule)) return UNDEFINED;
	return Value(mask.apply(iq.sourceModule));
      },
      UNDEFINED, 0
    );
  }
  
  // Get a Geometry
  edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometryHandle);
  assert(trackerGeometryHandle.isValid());
  
  // We need to track some extra stuff here for the Shells later.
  auto pxlayer  = extractors[intern("PXLayer")];
  auto pxladder = extractors[intern("PXLadder")];
  std::vector<Value> maxladders;

  // Now travrse the detector and collect whatever we need.
  auto detids = trackerGeometryHandle->detIds();
  for (DetId id : detids) {
    if (id.subdetId() != PixelSubdetector::PixelBarrel && id.subdetId() != PixelSubdetector::PixelEndcap) continue;
    auto iq = InterestingQuantities{.sourceModule = id };
    for (ID q : geomquantities) {
      Value v = extractors[q](iq);
      if (v != UNDEFINED) {
	if (v < min_value[q]) min_value[q] = v;
	if (v > max_value[q]) max_value[q] = v;
      }
    }
    auto layer = pxlayer(iq);
    if (layer != UNDEFINED) {
      if (layer >= Value(maxladders.size())) maxladders.resize(layer+1);
      auto ladder = pxladder(iq);
      if (ladder > maxladders[layer]) maxladders[layer] = ladder;
    }
    // for ROCs etc., they need to be added here as well.
    all_modules.push_back(iq);
  }

  // Shells are a concept that cannot be derived from bitmasks. 
  // Use hardcoded logic here.
  // This contains a lot more assumptions about general geometry than the rest
  // of the code, but it might work for Phase0 as well.
  Value innerring = 22; //TODO: Hardcoded number here.
  Value outerring = max_value[intern("PXBlade")] - innerring;
  auto pxblade  = extractors[intern("PXBlade")];
  addExtractor(intern("PXRing"), 
    [pxblade, innerring] (InterestingQuantities const& iq) {
      auto blade = pxblade(iq);
      if (blade == UNDEFINED) return UNDEFINED;
      if (blade <= innerring) return Value(1);
      else return Value(2);
    }, 1, 2
  );

  auto pxbarrel = extractors[intern("PXBarrel")];
  auto pxendcap = extractors[intern("PXEndcap")];
  auto pxmodule = extractors[intern("PXBModule")];
  Value maxmodule = max_value[intern("PXBModule")];
  addExtractor(intern("HalfCylinder"),
    [pxendcap, pxblade, innerring, outerring] (InterestingQuantities const& iq) {
      auto ec = pxendcap(iq);
      if (ec == UNDEFINED) return UNDEFINED;
      auto blade = pxblade(iq);
      auto inring  = blade > innerring ? blade - innerring : blade;
      auto perring = blade > innerring ? outerring : innerring;
      int frac = (int) ((inring-1) / float(perring) * 4); // floor semantics here
      if (frac == 0 || frac == 3) return 10*ec + 1; // inner half TODO: with 22 this is not well defined.
      if (frac == 1 || frac == 2) return 10*ec + 2; // outer half
      assert(!"HalfCylinder logic problem");
      return UNDEFINED;
    }, 0, 0 // N/A
  );
 
  addExtractor(intern("Shell"),
    [pxbarrel, pxladder, pxlayer, pxmodule, maxladders, maxmodule] (InterestingQuantities const& iq) {
      if(pxbarrel(iq) == UNDEFINED) return UNDEFINED;
      auto layer  = pxlayer(iq);
      auto ladder = pxladder(iq);
      auto module = pxmodule(iq);
      int frac = (int) ((ladder-1) / float(maxladders[layer]) * 4); // floor semantics
      Value dir = module <= (maxmodule/2) ? 1 : 2;  // minus/plus TODO: or other way round?
      if (frac == 0 || frac == 3) return 10*dir + 1; // inner half
      if (frac == 1 || frac == 2) return 10*dir + 2; // outer half
      assert(!"Shell logic problem");
      return UNDEFINED;
    }, 0, 0 // N/A
  );
  
  addExtractor(intern("PXForward"),
    [pxendcap] (InterestingQuantities const& iq) {
      if (pxendcap(iq) == UNDEFINED) return UNDEFINED;
      else return 0;
    },
    0, 0
  );

  addExtractor(intern(""), // A dummy column. Not much special handling required.
    [] (InterestingQuantities const& iq) { return 0; },
    0, 0
  );

}
 
void GeometryInterface::loadTimebased(edm::EventSetup const& iSetup, const edm::ParameterSet& iConfig) {
  // extractors for quantities that are roughly time-based. We cannot book plots based on these; they have to
  // be grouped away in step1.
  addExtractor(intern("Lumisection"),
    [] (InterestingQuantities const& iq) {
      if(!iq.sourceEvent) return UNDEFINED;
      return Value(iq.sourceEvent->luminosityBlock());
    },
    1, 5000
  );
  addExtractor(intern("LumiDecade"),
    [] (InterestingQuantities const& iq) {
      if(!iq.sourceEvent) return UNDEFINED;
      return Value(iq.sourceEvent->luminosityBlock() % 10);
    },
    0, 9
  );
  addExtractor(intern("BX"),
    [] (InterestingQuantities const& iq) {
      if(!iq.sourceEvent) return UNDEFINED;
      return Value(iq.sourceEvent->bunchCrossing());
    },
    1, 3600 // TODO: put actual max. BX
  );
}

void GeometryInterface::loadModuleLevel(edm::EventSetup const& iSetup, const edm::ParameterSet& iConfig) {
  // stuff that is within modules. Might require some phase0/phase1/strip switching later
  addExtractor(intern("row"),
    [] (InterestingQuantities const& iq) {
      return Value(iq.row);
    },
    0, 159
  );
  addExtractor(intern("col"),
    [] (InterestingQuantities const& iq) {
      return Value(iq.col);
    },
    0, 415 
  );

  // TODO: ROC dimensions could be configurable
  addExtractor(intern("ROC"),
    [] (InterestingQuantities const& iq) {
      int fedrow = int(iq.row / 80.0f);
      int fedcol = int(iq.col / 52.0f);
      if (fedrow == 0) return Value(fedcol);
      if (fedrow == 1) return Value(15 - fedcol);
      return UNDEFINED;
    },
    0, 15
  );

  // arbitrary per-ladder numbering (for inefficiencies)
  auto pxmodule = extractors[intern("PXBModule")];
  auto pxpanel  = extractors[intern("PXPanel")];
  auto roc      = extractors[intern("ROC")];
  addExtractor(intern("ROCinLadder"),
    [pxmodule, roc] (InterestingQuantities const& iq) {
      auto mod = pxmodule(iq);
      if (mod == UNDEFINED) return UNDEFINED;
      return Value(roc(iq) + 16 * (mod-1));
    },
    0, 127
  );
  addExtractor(intern("ROCinBlade"),
    [pxmodule, pxpanel, roc] (InterestingQuantities const& iq) {
      auto mod = pxpanel(iq);
      if (mod == UNDEFINED) return UNDEFINED;
      return Value(roc(iq) + 16 * (mod-1));
    },
    0, 31
  );

  addExtractor(intern("DetId"),
    [] (InterestingQuantities const& iq) {
      uint32_t id = iq.sourceModule.rawId();
      return Value(id);
    },
    0, 0 // No sane value possible here.
  );
}

void GeometryInterface::loadFEDCabling(edm::EventSetup const& iSetup, const edm::ParameterSet& iConfig) {
  edm::ESHandle<SiPixelFedCablingMap> theCablingMap;
  iSetup.get<SiPixelFedCablingMapRcd>().get(theCablingMap);
  std::map<DetId, Value> fedmap;
  uint32_t minFED = UNDEFINED, maxFED = 0;

  if (theCablingMap.isValid()) {
    auto map = theCablingMap.product();

    for(auto iq : all_modules) {
      std::vector<sipixelobjects::CablingPathToDetUnit> paths = map->pathToDetUnit(iq.sourceModule.rawId());
      for (auto p : paths) {
	//std::cout << "+++ cabling " << iq.sourceModule.rawId() << " " << p.fed << " " << p.link << " " << p.roc << "\n";
	fedmap[iq.sourceModule] = Value(p.fed);
	if (p.fed > maxFED) maxFED = p.fed;
	if (p.fed < minFED) minFED = p.fed;
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
    minFED, maxFED
  );
}
