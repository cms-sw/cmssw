#ifndef SiPixel_HistogramManager_h
#define SiPixel_HistogramManager_h
// -*- C++ -*-
//
// Package:     SiPixelPhase1Common
// Class  :     HistogramManager
//
// This helper is used by the DQM plugins to create histograms for different
// sub-partitions of the detector. It records all the samples that go into 
// histograms and takes a SummationSpecification. From these, it generates all
// the histograms in the right places and with consistent labels.
//
// One HistogramManager records one quantity, which may be multidimensional.
// A plugin can use more than one HistogramManager, but probably it should be
// more than one plugin in that case.
// 

// CMSSW
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

// DQM stuff
#include "DQMServices/Core/interface/DQMStore.h"

// Own Stuff
#include "DQM/SiPixelPhase1Common/interface/SummationSpecification.h"
#include "DQM/SiPixelPhase1Common/interface/GeometryInterface.h"
#include "DQM/SiPixelPhase1Common/interface/AbstractHistogram.h"



// TODO: Should we use a namespace, and if yes, which?
class HistogramManager {
public:
  explicit HistogramManager(const edm::ParameterSet& iConfig);

  // Add a specification for a set of plot. this has to happen before fill()'ing, since it optimizes for the spec.
  void addSpec(SummationSpecification spec);
  // This allows for a fluent interface, where the spec is constructed as a chain of builder-calls.
  SummationSpecificationBuilder addSpec();

  // Event is only needed for time-based quantities; row, col only if strcture within module is interesting.
  void fill(double value, DetId sourceModule, const edm::Event *sourceEvent = nullptr, int col = 0, int row = 0); 
  // TODO: we need multi-dimensional version, but probably a hardcoded fill2 for 2D will do.
  
  // Initiate the geometry extraction and book all required frames. Requires the specs to be set.
  void book(DQMStore::IBooker& iBooker, edm::EventSetup const& iSetup);

  // These functions perform step2, for online (per lumisection) or offline (endRun) respectively.
  // TODO: we need a EventSetup in offline as well. we'll see.:q
  void executeHarvestingOnline(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter, edm::EventSetup const& iSetup);
  void executeHarvestingOffline(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter);

  HistogramManager& setName(std::string name) {this->name = name; return *this;};
  HistogramManager& setTitle(std::string title) {this->title = title; return *this;};
  HistogramManager& setXlabel(std::string xlabel) {this->xlabel = xlabel; return *this;};
  HistogramManager& setYlabel(std::string ylabel) {this->ylabel = ylabel; return *this;};
  HistogramManager& setDimensions(int dimensions) {this->dimensions = dimensions; return *this;};
  HistogramManager& setRange(int nbins, double min, double max) {range_nbins = nbins; range_min = min; range_max = max; return *this; };

//private: // we need a bit more access for testing
  const edm::ParameterSet& iConfig;
  GeometryInterface& geometryInterface = GeometryInterface::get();

  std::vector<SummationSpecification> specs;
  typedef std::map<GeometryInterface::Values, AbstractHistogram> Table;
  std::vector<Table> tables;
 
  bool columsFinal = false;
  std::set<GeometryInterface::Column> significantColumns;

  std::string topFolderName;

  std::string name;
  std::string title;
  std::string xlabel;
  std::string ylabel;
  int dimensions = 1;
  int range_nbins  = 100;
  double range_min = 0;
  double range_max = 1000;
};


#endif
