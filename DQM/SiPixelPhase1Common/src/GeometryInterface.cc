// -*- C++ -*-
//
// Package:    SiPixelPhase1Common
// Class:      GeometryInterface
//
// Geometry depedence goes here.
//
// Original Author:  Marcel Schneider

#include "DQM/SiPixelPhase1Common/interface/GeometryInterface.h"

// general plotting helpers
#include "DQM/SiPixelPhase1Common/interface/SiPixelCoordinates.h"

// edm stuff
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESInputTag.h"

// Tracker Geometry/Topology  stuff
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameReverter.h"

// Pixel names
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"

// C++ stuff
#include <cassert>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <memory>

const GeometryInterface::Value GeometryInterface::UNDEFINED = 999999999.9f;

GeometryInterface::GeometryInterface(const edm::ParameterSet& conf,
                                     edm::ConsumesCollector&& iC,
                                     edm::Transition transition)
    : iConfig(conf), cablingMapLabel_{conf.getParameter<std::string>("CablingMapLabel")} {
  if (transition == edm::Transition::BeginRun) {
    trackerGeometryToken_ = iC.esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>();
    trackerTopologyToken_ = iC.esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>();
    siPixelFedCablingMapToken_ =
        iC.esConsumes<SiPixelFedCablingMap, SiPixelFedCablingMapRcd, edm::Transition::BeginRun>();
    if (!cablingMapLabel_.empty()) {
      labeledSiPixelFedCablingMapToken_ =
          iC.esConsumes<SiPixelFedCablingMap, SiPixelFedCablingMapRcd, edm::Transition::BeginRun>(
              edm::ESInputTag("", cablingMapLabel_));
    }
  } else {
    trackerGeometryToken_ =
        iC.esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::EndLuminosityBlock>();
    trackerTopologyToken_ = iC.esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::EndLuminosityBlock>();
    siPixelFedCablingMapToken_ =
        iC.esConsumes<SiPixelFedCablingMap, SiPixelFedCablingMapRcd, edm::Transition::EndLuminosityBlock>();
    if (!cablingMapLabel_.empty()) {
      labeledSiPixelFedCablingMapToken_ =
          iC.esConsumes<SiPixelFedCablingMap, SiPixelFedCablingMapRcd, edm::Transition::EndLuminosityBlock>(
              edm::ESInputTag("", cablingMapLabel_));
    }
  }
}

void GeometryInterface::load(edm::EventSetup const& iSetup) {
  const TrackerGeometry& trackerGeometry = iSetup.getData(trackerGeometryToken_);
  const TrackerTopology& trackerTopology = iSetup.getData(trackerTopologyToken_);
  const SiPixelFedCablingMap& siPixelFedCablingMap = iSetup.getData(siPixelFedCablingMapToken_);

  const SiPixelFedCablingMap* labeledSiPixelFedCablingMap = &siPixelFedCablingMap;
  if (!cablingMapLabel_.empty()) {
    labeledSiPixelFedCablingMap = &iSetup.getData(labeledSiPixelFedCablingMapToken_);
  }

  //loadFromAlignment(iSetup, iConfig);
  loadFromTopology(trackerGeometry, trackerTopology, iConfig);
  loadTimebased(iConfig);
  loadModuleLevel(iConfig);
  loadFEDCabling(labeledSiPixelFedCablingMap);
  loadFromSiPixelCoordinates(trackerGeometry, trackerTopology, siPixelFedCablingMap, iConfig);
  edm::LogInfo log("GeometryInterface");
  log << "Known colum names:\n";
  for (auto e : ids)
    log << "+++ column: " << e.first << " ok " << bool(extractors[e.second]) << " min " << min_value[e.second]
        << " max " << max_value[e.second] << "\n";
  is_loaded = true;
}

void GeometryInterface::loadFromTopology(const TrackerGeometry& trackerGeometry,
                                         const TrackerTopology& trackerTopology,
                                         const edm::ParameterSet& iConfig) {
  std::vector<ID> geomquantities;

  struct TTField {
    const TrackerTopology* tt;
    TrackerTopology::DetIdFields field;
    Value operator()(InterestingQuantities const& iq) {
      if (tt->hasField(iq.sourceModule, field))
        return tt->getField(iq.sourceModule, field);
      else
        return UNDEFINED;
    };
  };

  const TrackerTopology* tt = &trackerTopology;

  std::vector<std::pair<std::string, TTField>> namedPartitions{
      {"PXEndcap", {tt, TrackerTopology::PFSide}},

      {"PXLayer", {tt, TrackerTopology::PBLayer}},
      {"PXLadder", {tt, TrackerTopology::PBLadder}},
      {"PXBModule", {tt, TrackerTopology::PBModule}},

      {"PXBlade", {tt, TrackerTopology::PFBlade}},
      {"PXDisk", {tt, TrackerTopology::PFDisk}},
      {"PXPanel", {tt, TrackerTopology::PFPanel}},
      {"PXFModule", {tt, TrackerTopology::PFModule}},
  };

  for (auto& e : namedPartitions) {
    geomquantities.push_back(intern(e.first));
    addExtractor(intern(e.first), e.second, UNDEFINED, UNDEFINED);
  }

  auto pxbarrel = [](InterestingQuantities const& iq) {
    return iq.sourceModule.subdetId() == PixelSubdetector::PixelBarrel ? 0 : UNDEFINED;
  };
  auto pxforward = [](InterestingQuantities const& iq) {
    return iq.sourceModule.subdetId() == PixelSubdetector::PixelEndcap ? 0 : UNDEFINED;
  };
  auto pxall = [](InterestingQuantities const& iq) { return 0; };
  addExtractor(intern("PXBarrel"), pxbarrel, 0, 0);
  addExtractor(intern("PXForward"), pxforward, 0, 0);
  addExtractor(intern("PXAll"), pxall, 0, 0);

  // Redefine the disk numbering to use the sign
  auto pxendcap = extractors[intern("PXEndcap")];
  auto diskid = intern("PXDisk");
  auto pxdisk = extractors[diskid];
  extractors[diskid] = [pxdisk, pxendcap](InterestingQuantities const& iq) {
    auto disk = pxdisk(iq);
    if (disk == UNDEFINED)
      return UNDEFINED;
    auto endcap = pxendcap(iq);
    return endcap == 1 ? -disk : disk;
  };

  // DetId and module names
  auto detid = [](InterestingQuantities const& iq) {
    uint32_t id = iq.sourceModule.rawId();
    return Value(id);
  };
  addExtractor(intern("DetId"), detid, 0, 0  // No sane value possible here.
  );
  // these are just aliases with special handling in formatting
  // the names are created with PixelBarrelName et. al. later
  addExtractor(intern("PXModuleName"), detid, 0, 0);

  int phase = iConfig.getParameter<int>("upgradePhase");
  bool isUpgrade = phase == 1;

  // Now traverse the detector and collect whatever we need.
  auto detids = trackerGeometry.detIds();
  for (DetId id : detids) {
    if (id.subdetId() != PixelSubdetector::PixelBarrel && id.subdetId() != PixelSubdetector::PixelEndcap)
      continue;
    auto iq = InterestingQuantities{nullptr, id, 0, 0};

    // prepare pretty names
    std::string name = "";
    if (id.subdetId() == PixelSubdetector::PixelBarrel) {  // Barrel
      PixelBarrelName mod(id, tt, isUpgrade);
      name = mod.name();
    } else {  // assume Endcap
      PixelEndcapName mod(id, tt, isUpgrade);
      name = mod.name();
    }
    format_value[std::make_pair(intern("PXModuleName"), Value(id.rawId()))] = name;

    // we record each module 4 times, one for each corner, so we also get ROCs
    // in booking (at least for the ranges)
    const PixelGeomDetUnit* detUnit = dynamic_cast<const PixelGeomDetUnit*>(trackerGeometry.idToDetUnit(id));
    assert(detUnit);
    const PixelTopology* topo = &detUnit->specificTopology();
    iq.row = 0;
    iq.col = 0;
    all_modules.push_back(iq);
    iq.row = topo->nrows() - 1;
    iq.col = 0;
    all_modules.push_back(iq);
    iq.row = 0;
    iq.col = topo->ncolumns() - 1;
    all_modules.push_back(iq);
    iq.row = topo->nrows() - 1;
    iq.col = topo->ncolumns() - 1;
    all_modules.push_back(iq);
  }
}

void GeometryInterface::loadFromSiPixelCoordinates(const TrackerGeometry& trackerGeometry,
                                                   const TrackerTopology& trackerTopology,
                                                   const SiPixelFedCablingMap& siPixelFedCablingMap,
                                                   const edm::ParameterSet& iConfig) {
  // TODO: SiPixelCoordinates has a large overlap with theis GeometryInterface
  // in general.
  // Rough convention is to use own code for things that are easy and fast to
  // determine, and use SiPixelCoordinates for complicated things.
  // SiPixelCoordinates uses lookup maps for everything, so it is faster than
  // most other code, but still slow on DQM scales.
  int phase = iConfig.getParameter<int>("upgradePhase");

  // this shared pointer is kept alive by the references in the lambdas that follow.
  // That is a bit less obvious than keeping it as a member but more correct.
  auto coord = std::make_shared<SiPixelCoordinates>(phase);

  // note that we should reeinit for each event. But this probably won't explode
  // thanks to the massive memoization in SiPixelCoordinates which is completely
  // initialized while booking.
  coord->init(&trackerTopology, &trackerGeometry, &siPixelFedCablingMap);

  // SiPixelCoordinates uses a different convention for UNDEFINED:
  auto from_coord = [](double in) { return (in == -9999.0) ? UNDEFINED : Value(in); };

  // Rings are a concept that cannot be derived from bitmasks.
  addExtractor(intern("PXRing"), [coord, from_coord](InterestingQuantities const& iq) {
    return from_coord(coord->ring(iq.sourceModule));
  });

  // Quadrant names.
  auto pxbarrel = extractors[intern("PXBarrel")];
  addExtractor(
      intern("HalfCylinder"),
      [coord, pxbarrel](InterestingQuantities const& iq) {
        if (pxbarrel(iq) != UNDEFINED)
          return UNDEFINED;
        int quadrant = coord->quadrant(iq.sourceModule);
        switch (quadrant) {
          case 1:
            return Value(12);  // mO
          case 2:
            return Value(11);  // mI
          case 3:
            return Value(22);  // pO
          case 4:
            return Value(21);  // pI
          default:
            return UNDEFINED;
        }
      },
      0,
      0  // N/A
  );
  addExtractor(
      intern("Shell"),
      [coord, pxbarrel](InterestingQuantities const& iq) {
        if (pxbarrel(iq) == UNDEFINED)
          return UNDEFINED;
        int quadrant = coord->quadrant(iq.sourceModule);
        switch (quadrant) {
          case 1:
            return Value(12);  // mO
          case 2:
            return Value(11);  // mI
          case 3:
            return Value(22);  // pO
          case 4:
            return Value(21);  // pI
          default:
            return UNDEFINED;
        }
      },
      0,
      0  // N/A
  );

  // Online Numbering.
  addExtractor(intern("SignedLadder"), [coord, from_coord](InterestingQuantities const& iq) {
    return from_coord(coord->signed_ladder(iq.sourceModule()));
  });
  addExtractor(intern("SignedModule"), [coord, from_coord](InterestingQuantities const& iq) {
    return from_coord(coord->signed_module(iq.sourceModule()));
  });
  addExtractor(intern("SignedBlade"), [coord, from_coord](InterestingQuantities const& iq) {
    return from_coord(coord->signed_blade(iq.sourceModule()));
  });

  // Flipped and Outer ladders
  addExtractor(intern("OuterLadder"), [coord, from_coord](InterestingQuantities const& iq) {
    return from_coord(coord->outer(iq.sourceModule()));
  });
  addExtractor(intern("FlippedLadder"), [coord, from_coord](InterestingQuantities const& iq) {
    return from_coord(coord->flipped(iq.sourceModule()));
  });

  // Pixel Map axis.
  // TODO: automatic range and binning for phase0 are incorrect.
  // Should be set manually here.
  addExtractor(
      intern("SignedModuleCoord"),  // BPIX x
      [coord, from_coord](InterestingQuantities const& iq) {
        return from_coord(coord->signed_module_coord(iq.sourceModule(), std::make_pair(int(iq.row), int(iq.col))));
      },
      UNDEFINED,
      UNDEFINED,
      1.0 / 8.0);
  addExtractor(
      intern("SignedLadderCoord"),  // BPIX y
      [coord, from_coord](InterestingQuantities const& iq) {
        return from_coord(coord->signed_ladder_coord(iq.sourceModule(), std::make_pair(int(iq.row), int(iq.col))));
      },
      UNDEFINED,
      UNDEFINED,
      1.0 / 2.0);
  addExtractor(
      intern("SignedDiskCoord"),  // FPIX x (per-ring)
      [coord, from_coord](InterestingQuantities const& iq) {
        return from_coord(coord->signed_disk_coord(iq.sourceModule(), std::make_pair(int(iq.row), int(iq.col))));
      },
      UNDEFINED,
      UNDEFINED,
      1.0 / 8.0);
  addExtractor(
      intern("SignedDiskRingCoord"),  // FPIX x (FPIX-as-one-plot)
      [coord, from_coord](InterestingQuantities const& iq) {
        return from_coord(coord->signed_disk_ring_coord(iq.sourceModule(), std::make_pair(int(iq.row), int(iq.col))));
      },
      UNDEFINED,
      UNDEFINED,
      1.0 / 16.0);
  addExtractor(
      intern("SignedBladePanelCoord"),  // FPIX y
      [coord, from_coord, phase](InterestingQuantities const& iq) {
        if (phase == 0) {
          return from_coord(coord->signed_blade_coord(iq.sourceModule(), std::make_pair(int(iq.row), int(iq.col))));
        } else if (phase == 1) {
          return from_coord(
              coord->signed_blade_panel_coord(iq.sourceModule(), std::make_pair(int(iq.row), int(iq.col))));
        } else {
          return UNDEFINED;  // TODO: phase2
        }
      },
      UNDEFINED,
      UNDEFINED,
      phase == 1 ? 0.25 : 0.2);
  addExtractor(
      intern("SignedShiftedBladePanelCoord"),  // FPIX-as-one y
      [coord, from_coord, phase](InterestingQuantities const& iq) {
        if (phase == 0) {
          return from_coord(coord->signed_blade_coord(iq.sourceModule(), std::make_pair(int(iq.row), int(iq.col))));
        } else if (phase == 1) {
          return from_coord(
              coord->signed_shifted_blade_panel_coord(iq.sourceModule(), std::make_pair(int(iq.row), int(iq.col))));
        } else {
          return UNDEFINED;  // TODO: phase2
        }
      },
      UNDEFINED,
      UNDEFINED,
      phase == 1 ? 0.25 : 0.1  // half-roc for phase0
  );
  addExtractor(
      intern("SignedBladePanel"),  // per-module FPIX y
      [coord, from_coord](InterestingQuantities const& iq) {
        return from_coord(coord->signed_blade_panel_coord(iq.sourceModule(), std::make_pair(int(iq.row), int(iq.col))));
      },
      UNDEFINED,
      UNDEFINED,
      1.0 / 2.0);

  addExtractor(
      intern("SignedBladePanel"),
      [coord, from_coord](InterestingQuantities const& iq) {
        return from_coord(coord->signed_blade_panel_coord(iq.sourceModule(), std::make_pair(int(iq.row), int(iq.col))));
      },
      UNDEFINED,
      UNDEFINED,
      1.0 / 2.0);

  // more readout-related things.
  addExtractor(intern("ROC"), [coord, from_coord](InterestingQuantities const& iq) {
    return from_coord(coord->roc(iq.sourceModule(), std::make_pair(int(iq.row), int(iq.col))));
  });
  addExtractor(intern("Sector"), [coord, from_coord](InterestingQuantities const& iq) {
    return from_coord(coord->sector(iq.sourceModule()));
  });
  addExtractor(intern("Channel"), [coord, from_coord](InterestingQuantities const& iq) {
    return from_coord(coord->channel(iq.sourceModule(), std::make_pair(int(iq.row), int(iq.col))));
  });
}

void GeometryInterface::loadTimebased(const edm::ParameterSet& iConfig) {
  // extractors for quantities that are roughly time-based. We cannot book plots based on these; they have to
  // be grouped away in step1.
  addExtractor(
      intern("Lumisection"),
      [](InterestingQuantities const& iq) {
        if (!iq.sourceEvent)
          return UNDEFINED;
        return Value(iq.sourceEvent->luminosityBlock());
      },
      1,
      iConfig.getParameter<int>("max_lumisection"));

  int onlineblock = iConfig.getParameter<int>("onlineblock");
  int n_onlineblocks = iConfig.getParameter<int>("n_onlineblocks");
  addExtractor(
      intern("OnlineBlock"),
      [onlineblock](InterestingQuantities const& iq) {
        if (!iq.sourceEvent)
          return UNDEFINED;
        return Value(onlineblock + iq.sourceEvent->luminosityBlock() / onlineblock);
      },
      // note: this range is not visible anywhere (if the RenderPlugin does its job),
      // but the strange range allows the RenderPlugin to know the block size.
      onlineblock,
      onlineblock + n_onlineblocks - 1);

  int lumiblock = iConfig.getParameter<int>("lumiblock");
  addExtractor(
      intern("LumiBlock"),
      [lumiblock](InterestingQuantities const& iq) {
        if (!iq.sourceEvent)
          return UNDEFINED;
        // The '-1' is for making 1-10 the same block rather than 0-9
        // The '+0.5' makes the block span an integer range rather n.5-m.5
        return Value(((iq.sourceEvent->luminosityBlock() - 1) / lumiblock) + 0.5);
      },
      -0.5,
      iConfig.getParameter<int>("max_lumisection") / lumiblock);

  addExtractor(
      intern("BX"),
      [](InterestingQuantities const& iq) {
        if (!iq.sourceEvent)
          return UNDEFINED;
        return Value(iq.sourceEvent->bunchCrossing());
      },
      1,
      iConfig.getParameter<int>("max_bunchcrossing"));
}

void GeometryInterface::loadModuleLevel(const edm::ParameterSet& iConfig) {
  // stuff that is within modules. Might require some phase0/phase1/strip switching later
  addExtractor(
      intern("row"),
      [](InterestingQuantities const& iq) { return Value(iq.row); },
      0,
      iConfig.getParameter<int>("module_rows") - 1);
  addExtractor(
      intern("col"),
      [](InterestingQuantities const& iq) { return Value(iq.col); },
      0,
      iConfig.getParameter<int>("module_cols") - 1);
}

void GeometryInterface::loadFEDCabling(const SiPixelFedCablingMap* labeledSiPixelFedCablingMap) {
  std::shared_ptr<SiPixelFrameReverter> siPixelFrameReverter =
      // I think passing the bare pointer here is safe, but who knows...
      std::make_shared<SiPixelFrameReverter>(labeledSiPixelFedCablingMap);

  addExtractor(intern("FED"), [siPixelFrameReverter](InterestingQuantities const& iq) {
    if (iq.sourceModule == 0xFFFFFFFF)
      return Value(iq.col);  // hijacked for the raw data plugin
    return Value(siPixelFrameReverter->findFedId(iq.sourceModule.rawId()));
  });

  // TODO: ranges should be set manually below, since booking probably cannot
  // infer them correctly (no ROC-level granularity)
  // PERF: this is slow. Prefer SiPixelCordinates versions here.
  addExtractor(intern("LinkInFed"), [siPixelFrameReverter](InterestingQuantities const& iq) {
    if (iq.sourceModule == 0xFFFFFFFF)
      return Value(iq.row);  // hijacked for the raw data plugin
    sipixelobjects::GlobalPixel gp = {iq.row, iq.col};
    return Value(siPixelFrameReverter->findLinkInFed(iq.sourceModule.rawId(), gp));
  });
  // not sure if this is useful anywhere.
  addExtractor(intern("RocInLink"), [siPixelFrameReverter](InterestingQuantities const& iq) {
    sipixelobjects::GlobalPixel gp = {iq.row, iq.col};
    return Value(siPixelFrameReverter->findRocInLink(iq.sourceModule.rawId(), gp));
  });
  // This might be equivalent to our ROC numbering.
  addExtractor(intern("RocInDet"), [siPixelFrameReverter](InterestingQuantities const& iq) {
    sipixelobjects::GlobalPixel gp = {iq.row, iq.col};
    return Value(siPixelFrameReverter->findRocInDet(iq.sourceModule.rawId(), gp));
  });
}

std::string GeometryInterface::formatValue(Column col, Value val) {
  auto it = format_value.find(std::make_pair(col, val));
  if (it != format_value.end())
    return it->second;

  // non-number output names (_pO etc.) are hardwired here.
  std::string name = pretty(col);
  std::string value = "_" + std::to_string(int(val));
  if (val == 0)
    value = "";                     // hide Barrel_0 etc.
  if (name == "PXDisk" && val > 0)  // +/- sign for disk num
    value = "_+" + std::to_string(int(val));
  // pretty (legacy?) names for Shells and HalfCylinders
  std::map<int, std::string> shellname{{11, "_mI"}, {12, "_mO"}, {21, "_pI"}, {22, "_pO"}};
  if (name == "HalfCylinder" || name == "Shell")
    value = shellname[int(val)];
  if (val == UNDEFINED)
    value = "_UNDEFINED";
  return format_value[std::make_pair(col, val)] = name + value;
}
