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
  explicit HistogramManager(const edm::ParameterSet& iConfig, GeometryInterface& geo);

  void addSpec(SummationSpecification spec);

  // Event is only needed for time-based quantities; row, col only if strcture within module is interesting.
  void fill(DetId sourceModule, const edm::Event *sourceEvent = nullptr, int col = 0, int row = 0); 
  void fill(double value, DetId sourceModule, const edm::Event *sourceEvent = nullptr, int col = 0, int row = 0); 
  void fill(double x, double y, DetId sourceModule, const edm::Event *sourceEvent = nullptr, int col = 0, int row = 0); 

  // This needs to be called after each event (in the analyzer) for per-event counting, like ndigis.
  void executePerEventHarvesting();
  
  // Initiate the geometry extraction and book all required frames. Requires the specs to be set.
  void book(DQMStore::IBooker& iBooker, edm::EventSetup const& iSetup);

  // These functions perform step2, for online (per lumisection) or offline (endRun) respectively.
  // TODO: we need a EventSetup in offline as well. we'll see.
  void executeHarvestingOnline(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter, edm::EventSetup const& iSetup);
  void executeHarvestingOffline(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter);

  typedef std::map<GeometryInterface::Values, AbstractHistogram> Table;
  // Set a handler to be called when a custom() step is hit. This can do 
  // arbitrary things to the histogram Table, including copying it for later 
  // use. Using such saved tables form other HistogramManagers, e.g. 
  // efficiencies can be computed here.
  template<typename FUNC>
  void setCustomHandler(FUNC handler) {customHandler = handler; };

private:
  const edm::ParameterSet& iConfig;
  GeometryInterface& geometryInterface;
  std::function<void(SummationStep& step, Table& t)> customHandler;

  std::vector<SummationSpecification> specs;
  std::vector<Table> tables;

  std::string makePath(GeometryInterface::Values const&);

  void executeStep1Spec(double x, double y,
                        GeometryInterface::Values& significantvalues, 
			SummationSpecification& s, 
			Table& t,
			SummationStep::Stage stage);
 
  void loadFromDQMStore(SummationSpecification& s, Table& t, DQMStore::IGetter& iGetter);
  void executeSave(SummationStep& step, Table& t, DQMStore::IBooker& iBooker);
  void executeGroupBy(SummationStep& step, Table& t);
  void executeReduce(SummationStep& step, Table& t);
  void executeExtend(SummationStep& step, Table& t);

  bool enabled;
  bool bookUndefined;
  std::string top_folder_name;
  std::string default_grouping;

  std::string name;
  std::string title;
  std::string xlabel;
  std::string ylabel;
  int dimensions;
  int range_nbins;
  double range_min;
  double range_max;

  // These are actually more like local variables, and they might be shadowed
  // by locals now and then. The point is to avoid reallocating the heap buffer
  // of the Values on every call.
  GeometryInterface::Values significantvalues;
  GeometryInterface::Values new_vals;
};


#endif
