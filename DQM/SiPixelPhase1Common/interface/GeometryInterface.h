#ifndef SiPixel_GeometryInterface_h
#define SiPixel_GeometryInterface_h
// -*- C++ -*-
//
// Package:    SiPixelPhase1Common
// Class:      GeometryInterface
//
// The histogram manager uses this class to gather information about a sample.
// All geometry dependence goes here.
//
// Original Author: Marcel Schneider
//

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include <functional>
#include <map>
#include <string>
#include <array>

class GeometryInterface {
public:
  // an ID is produced by interning a string name.
  typedef int ID;
  // A column could have multiple IDs if it is a or-form.
  // Not used atm, makes many things much easier.
  typedef ID Column;
  typedef double Value;
  static const Value UNDEFINED;

  // Essentially a map backed by a vector (for the small counts here
  // this should always be faster). Most ops turned out to be not needed.
  typedef std::vector<std::pair<Column, Value>> Values;

  GeometryInterface(const edm::ParameterSet&,
                    edm::ConsumesCollector&&,
                    edm::Transition transition = edm::Transition::BeginRun);

  bool loaded() { return is_loaded; };

  // The hard work happens here.
  void load(edm::EventSetup const& iSetup);

  struct InterestingQuantities {
    // in this order the struct should fit 2 64bit words and is cheap to copy.
    const edm::Event* sourceEvent = nullptr;
    DetId sourceModule;
    int16_t col = 0;
    int16_t row = 0;
  };

  // This has to be fast, _should_ not malloc.
  void extractColumns(std::vector<Column> const& names, InterestingQuantities const& iq, Values& out) {
    out.clear();
    for (Column const& col : names) {
      auto val = extract(col, iq);
      out.push_back(val);
    }
  }

  // the pair return value is historical; it is only really needed with or-columns.
  // But it is cleaner to carry it around.
  std::pair<Column, Value> extract(Column const& col, InterestingQuantities const& iq) {
    assert(col != 0 || !"Extracting invalid column.");
    ID id = col;
    assert(ID(extractors.size()) > id || !"extractors vector too small!");
    auto& ex = extractors[id];
    if (!ex) {  // we have never heard about this. This is a typo for sure.
      edm::LogError("GeometryInterface") << "Undefined column used: " << unintern(id) << ". Check your spelling.\n";
    } else {
      auto val = ex(iq);
      if (val != UNDEFINED) {
        return std::make_pair(Column{id}, val);
      }
    }
    return std::make_pair(col, UNDEFINED);
  }

  Value extract(ID id, DetId did, edm::Event* ev = nullptr, int16_t col = 0, int16_t row = 0) {
    InterestingQuantities iq = {ev, did, col, row};
    return extractors[id](iq);
  }

  // TODO: for Phase0 (and maybe also Phase2) this should include the 4 corners
  // of each ROC (or the *correct* corners of the respective modules).
  std::vector<InterestingQuantities> const& allModules() { return all_modules; }

  Value maxValue(ID id) { return max_value[id]; };
  Value minValue(ID id) { return min_value[id]; };
  Value binWidth(ID id) { return bin_width[id]; };

  // turn string into an ID, adding it if needed.
  ID intern(std::string const& id) {
    auto it = ids.find(id);
    if (it == ids.end()) {
      ids[id] = ++max_id;
      extractors.resize(max_id + 1);
    }
    return ids[id];
  };

  // turn an ID back into a string. Only for pretty output (including histo
  // labels), so it can be slow (though intern() does not have to be fast
  // either).
  std::string unintern(ID id) {
    for (auto& e : ids)
      if (e.second == id)
        return e.first;
    return "INVALID";
  }

  std::string pretty(Column col) { return unintern(col); }

  std::string formatValue(Column, Value);

private:
  void loadFromTopology(const TrackerGeometry&, const TrackerTopology&, const edm::ParameterSet&);
  void loadFromSiPixelCoordinates(const TrackerGeometry&,
                                  const TrackerTopology&,
                                  const SiPixelFedCablingMap&,
                                  const edm::ParameterSet&);
  void loadTimebased(const edm::ParameterSet& iConfig);
  void loadModuleLevel(const edm::ParameterSet& iConfig);
  void loadFEDCabling(const SiPixelFedCablingMap*);

  const edm::ParameterSet iConfig;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;
  // When converting this module to use esConsumes I preserved the previous behavior.
  // In one place a configurable label is allowed when getting SiPixelFedCablingMap
  // and in another place it always assumes an empty label. I am not sure if this is
  // actually correct (seems strange). Maybe it does not matter as in all the configurations
  // I could find in the repository the parameter was an empty string... An expert who
  // understands this might want to fix this or eliminate this comment if the behavior is
  // correct.
  edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> siPixelFedCablingMapToken_;
  edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> labeledSiPixelFedCablingMapToken_;
  std::string cablingMapLabel_;

  bool is_loaded = false;

  // This holds closures that compute the column values in step1.
  // can be a Vector since ids are dense.
  std::vector<std::function<Value(InterestingQuantities const& iq)>> extractors;
  // quantity range if it is known. Can be UNDEFINED, in this case booking will
  // determine the range. Map for ease of use.
  std::map<ID, Value> max_value;
  std::map<ID, Value> min_value;
  std::map<ID, Value> bin_width;

  // cache of pre-formatted values. Can be pre-populated while loading
  // (used for Pixel*Name)
  std::map<std::pair<Column, Value>, std::string> format_value;

  void addExtractor(ID id,
                    std::function<Value(InterestingQuantities const& iq)> func,
                    Value min = UNDEFINED,
                    Value max = UNDEFINED,
                    Value binwidth = 1) {
    max_value[id] = max;
    min_value[id] = min;
    bin_width[id] = binwidth;
    extractors[id] = func;
  }

  std::vector<InterestingQuantities> all_modules;

  // interning table. Maps string IDs to a dense set of integer IDs
  std::map<std::string, ID> ids{std::make_pair(std::string("INVALID"), ID(0))};
  ID max_id = 0;
};

#endif
