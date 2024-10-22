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
// A plugin can use more than one HistogramManager, which can be held in a
// HistogramManagerHolder (SiPixelPhase1Base.h)
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

class HistogramManager {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  explicit HistogramManager(const edm::ParameterSet& iConfig, GeometryInterface& geo);

  void addSpec(SummationSpecification spec);

  // Event is only needed for time-based quantities; row, col only if strcture within module is interesting.
  void fill(DetId sourceModule, const edm::Event* sourceEvent = nullptr, int col = 0, int row = 0);
  void fill(double value, DetId sourceModule, const edm::Event* sourceEvent = nullptr, int col = 0, int row = 0);
  void fill(double x, double y, DetId sourceModule, const edm::Event* sourceEvent = nullptr, int col = 0, int row = 0);

  // This needs to be called after each event (in the analyzer) for per-event counting, like ndigis.
  void executePerEventHarvesting(edm::Event const* ev);

  // Initiate the geometry extraction and book all required frames. Requires the specs to be set.
  void book(DQMStore::IBooker& iBooker, edm::EventSetup const& iSetup);

  // These functions perform step2, for online (per lumisection) or offline (endRun) respectively.
  // Note that the EventSetup from PerLumi is used in offline as well, so PerLumi always has to be called first.
  void executePerLumiHarvesting(DQMStore::IBooker& iBooker,
                                DQMStore::IGetter& iGetter,
                                edm::LuminosityBlock const& lumiBlock,
                                edm::EventSetup const& iSetup);
  void executeHarvesting(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter);

  typedef std::map<GeometryInterface::Values, AbstractHistogram> Table;

private:
  GeometryInterface& geometryInterface;

  std::vector<SummationSpecification> specs;
  std::vector<Table> tables;
  std::vector<Table> counters;

  std::pair<std::string, std::string> makePathName(SummationSpecification const& s,
                                                   GeometryInterface::Values const&,
                                                   SummationStep const* upto);

  void fillInternal(double x,
                    double y,
                    int n_parameters,
                    GeometryInterface::InterestingQuantities const& iq,
                    std::vector<SummationStep>::iterator first,
                    std::vector<SummationStep>::iterator last,
                    AbstractHistogram& dest);

  void loadFromDQMStore(SummationSpecification& s, Table& t, DQMStore::IGetter& iGetter);
  void executeGroupBy(SummationStep const& step, Table& t, DQMStore::IBooker& iBooker, SummationSpecification const& s);
  void executeExtend(SummationStep const& step,
                     Table& t,
                     std::string const& reduction,
                     DQMStore::IBooker& iBooker,
                     SummationSpecification const& s);

public:  // these are available in config as is, and may be used in harvesting.
  bool enabled;
  bool perLumiHarvesting;
  bool bookUndefined;
  std::string top_folder_name;

  std::string name;
  std::string title;
  std::string xlabel;
  std::string ylabel;
  int dimensions;
  int range_x_nbins;
  double range_x_min;
  double range_x_max;
  int range_y_nbins;
  double range_y_min;
  double range_y_max;
  bool statsOverflows;

  // can be used in "custom" harvesting in online.
  edm::LuminosityBlock const* lumisection = nullptr;

private:
  // These are actually more like local variables, and they might be shadowed
  // by locals now and then. The point is to avoid reallocating the heap buffer
  // of the Values on every call.
  // iq/significantvalues are also used to cache the last set of columns
  // per-spec, to avoid unnecessary extractions.
  GeometryInterface::InterestingQuantities iq;
  // "immutable" cache
  std::vector<GeometryInterface::Values> significantvalues;
  // Direct links to the Histogram if the caching above succeeds.
  std::vector<AbstractHistogram*> fastpath;
};

#endif
