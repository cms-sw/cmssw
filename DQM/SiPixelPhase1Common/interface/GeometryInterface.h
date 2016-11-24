#ifndef SiPixel_GeometryInterface_h
#define SiPixel_GeometryInterface_h
// -*- C++ -*-
//
// Package:    SiPixelPhase1Common
// Class:      GeometryInterface
//
// The histogram manager uses this class to gather information about a sample.
// All geometry dependence goes here. This is a singleton, (ed::Service) but
// only for performance reasons.
// At some point we might need to switch to sth. more complicated (if we want
// to deal with more than one geometry per process).
//
// Original Author: Marcel Schneider
//

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <functional>
#include <map>
#include <string>
#include <array>

class GeometryInterface {
 public:
  // an ID is produced by interning a string name.
  typedef int ID;
  // A column can have multiple IDs if it is a or-form. If not, only [0] is
  // used, the other entries are 0. A limit of 2 is arbitrary here.
  // Once a value is assigned, the column is "normalized", and only the ID
  // of the extracted value is in [0]. Fuzzy matching is used to find a
  // normalized column when a multicolumn is given and vice versa.
  // Increasing the max number of columns needs fixing some todos.
  typedef std::array<ID, 2> Column;
  typedef int Value;
  static const Value UNDEFINED = 0x0FFFFFFF;

  // Essentially a map backed by a vector (for the small counts here
  // this should always be faster), with special handling for multi-colums.
  struct Values {
    std::vector<std::pair<Column, Value>> values;

    static bool columnMatch(Column const& a, Column const& b) {
      return a == b  // trivial case TODO: loops for size > 2
             || (a[0] == b[0] && a[1] == 0)   // normalized case 1
             || (a[0] == b[1] && a[1] == 0)   // normalized case 2
             || (b[0] == a[0] && b[1] == 0)   // normalized case 1 (inv)
             || (b[0] == a[1] && b[1] == 0);  // normalized case 2 (inv)
    }

    void put(Column c, Value v) {
      for (auto& e : values)
        if (columnMatch(e.first, c)) {
          e.first = c;
          e.second = v;
          return;
        }
      values.push_back(make_pair(c, v));  // else
    };
    void put(std::pair<Column, Value> v) { put(v.first, v.second); };
    std::pair<Column, Value> get(Column c) const {
      for (auto& e : values)
        if (columnMatch(e.first, c)) return e;
      return make_pair(Column{{0, 0}}, UNDEFINED);
    };
    void erase(Column c) {
      for (auto it = values.begin(); it != values.end(); ++it)
        if (columnMatch(it->first, c)) {
          values.erase(it);
          return;
        }
    }
    void clear() { values.clear(); };
    Value operator[](Column c) { return get(c).second; };
    bool operator<(Values const& other) const {
      return this->values < other.values;
    };
  };

  GeometryInterface(const edm::ParameterSet& conf) : iConfig(conf){};

  bool loaded() { return is_loaded; };

  // The hard work happens here.
  // this is _not_ thread save, but it should only be called in
  // booking/harvesting.
  void load(edm::EventSetup const& iSetup);

  struct InterestingQuantities {
    // in this order the struct should fit 2 64bit words and is cheap to copy.
    const edm::Event* sourceEvent;
    DetId sourceModule;
    int16_t col;
    int16_t row;
  };

  // This has to be fast, _should_ not malloc.
  void extractColumns(std::vector<Column> const& names,
                      InterestingQuantities const& iq, Values& out) {
    out.clear();
    for (Column const& col : names) {
      auto val = extract(col, iq);
      out.put(val.first, val.second);
    }
  };

  std::pair<Column, Value> extract(Column const& col, InterestingQuantities const& iq) {
    assert(col[0] != 0 || !"Extracting invalid column.");
    for (ID id : col) {
      if (id == 0) break;  // all set columns failed
      assert(ID(extractors.size()) > id || !"extractors vector too small!");
      auto& ex = extractors[id];
      if (!ex) {  // we have never heard about this. This is a typo for sure.
        edm::LogError("GeometryInterface")
            << "Undefined column used: " << unintern(id)
            << ". Check your spelling.\n";
      } else {
        auto val = ex(iq);
        if (val != UNDEFINED) {
          return std::make_pair(Column{{id, 0}}, val);  // double braces for g++
          break;
        }
      }
    }
    return std::make_pair(col, UNDEFINED);
  }

  Value extract(ID id, DetId did, edm::Event* ev = nullptr, int16_t col = 0,
                int16_t row = 0) {
    InterestingQuantities iq = {ev, did, col, row};
    return extractors[id](iq);
  }

  std::vector<InterestingQuantities> const& allModules() {
    return all_modules;
  };
  Value maxValue(ID id) { return max_value[id]; };
  Value minValue(ID id) { return min_value[id]; };

  // turn string into an ID, adding it if needed.
  // needs the lock since this will be called from the spec builder, which will
  // run in the constructor and this is parallel.
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
  // either). Also locks, might not be needed but better save than sorry.
  std::string unintern(ID id) {
    for (auto& e : ids)
      if (e.second == id) return e.first;
    return "INVALID";
  }

  std::string pretty(Column col) {
    if (col[1] == 0)  // normal column
      return unintern(col[0]);
    else
      return unintern(col[0]) +
             /*"|" +*/ unintern(col[1]);  // TODO: should loop.
  }

 private:
  // void loadFromAlignment(edm::EventSetup const& iSetup, const
  // edm::ParameterSet& iConfig);
  void loadFromTopology(edm::EventSetup const& iSetup,
                        const edm::ParameterSet& iConfig);
  void loadTimebased(edm::EventSetup const& iSetup,
                     const edm::ParameterSet& iConfig);
  void loadModuleLevel(edm::EventSetup const& iSetup,
                       const edm::ParameterSet& iConfig);
  void loadFEDCabling(edm::EventSetup const& iSetup,
                      const edm::ParameterSet& iConfig);

  const edm::ParameterSet iConfig;

  bool is_loaded = false;

  // This holds closures that compute the column values in step1.
  // can be a Vector since ids are dense.
  std::vector<std::function<Value(InterestingQuantities const& iq)>> extractors;
  // quantity range if it is known. Can be UNDEFINED, in this case booking will
  // determine the range.
  // map for ease of use.
  std::map<ID, Value> max_value;
  std::map<ID, Value> min_value;

  void addExtractor(ID id,
                    std::function<Value(InterestingQuantities const& iq)> func,
                    Value min = UNDEFINED, Value max = UNDEFINED) {
    max_value[id] = max;
    min_value[id] = min;
    extractors[id] = func;
  }

  std::vector<InterestingQuantities> all_modules;

  // interning table. Maps string IDs to a dense set of integer IDs
  std::map<std::string, ID> ids{std::make_pair(std::string("INVALID"), ID(0))};
  ID max_id = 0;
};

#endif
