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

  // For step2, we need something here that that takes the spec and the DQMStore
  // and executes the rest of the spec. This may need a lot of information to
  // reconstruct the folder names used before... we'll see.

  void setName(std::string name) {this->name = name;};
  void setTitle(std::string title) {this->title = title;};
  void setXlabel(std::string xlabel) {this->xlabel = xlabel;};
  void setYlabel(std::string ylabel) {this->ylabel = ylabel;};
  void setDimensions(int dimensions) {this->dimensions = dimensions;};

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
  int dimensions;


};


#endif
