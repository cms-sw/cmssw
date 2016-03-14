#ifndef SiPixel_GeometryInterface_h
#define SiPixel_GeometryInterface_h
// -*- C++ -*-
//
// Package:    SiPixelPhase1Common
// Class:      GeometryInterface
//
// The histogram manager uses this class to gather information about a sample.
// All geometry dependence goes here. This is a singleton, but only for
// performance reasons. At some point we might need to switch to sth. more 
// complicated (if we want to deal with more than one geometry per process) 
//
// Original Author: Marcel Schneider
//

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include <functional>
#include <map>
#include <string>

#include <iostream>

class GeometryInterface {
  public:
  typedef std::string Column;
  typedef std::map<Column, int> Values;

  static GeometryInterface& get() { return instance; };

  bool loaded() { return is_loaded; };

  // The hard work happens here.
  void load(edm::EventSetup const& iSetup, const edm::ParameterSet& iConfig);

  struct InterestingQuantities {
    DetId sourceModule;
    edm::Event *sourceEvent;
    int col; int row;
  };

  // This has to be fast, _should_ not malloc.
  // Current impl is not so good, but interning should fix it.
  const Values extractColumns(std::set<Column> names, InterestingQuantities const& iq) {
    Values out;
    for (auto col : names) {
      auto ex = extractors.find(col);
      if (ex == extractors.end()) {
	// we have never heard about this. This is a typo for sure.
	std::cout << "Undefined column used: " << col << ". Check your spelling.\n";
      } else {
        auto val = ex->second(iq);
	out[col] = val;
      }
    }
    return out;
  };


  private:

  static GeometryInterface instance;
  bool is_loaded = false;
  std::map<Column, std::function<int(InterestingQuantities const& iq)>> extractors;
};


#endif

