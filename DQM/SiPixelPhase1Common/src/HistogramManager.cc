// -*- C++ -*-
//
// Package:     SiPixelPhase1Common
// Class  :     HistogramManager
//
#include "DQM/SiPixelPhase1Common/interface/HistogramManager.h"

#include <sstream>
#include <boost/algorithm/string.hpp>

// Geometry stuff
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// Logger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HistogramManager::HistogramManager(const edm::ParameterSet& iconfig, GeometryInterface& geo)
    : geometryInterface(geo),
      enabled(iconfig.getParameter<bool>("enabled")),
      perLumiHarvesting(iconfig.getParameter<bool>("perLumiHarvesting")),
      bookUndefined(iconfig.getParameter<bool>("bookUndefined")),
      top_folder_name(iconfig.getParameter<std::string>("topFolderName")),
      name(iconfig.getParameter<std::string>("name")),
      title(iconfig.getParameter<std::string>("title")),
      xlabel(iconfig.getParameter<std::string>("xlabel")),
      ylabel(iconfig.getParameter<std::string>("ylabel")),
      dimensions(iconfig.getParameter<int>("dimensions")),
      range_x_nbins(iconfig.getParameter<int>("range_nbins")),
      range_x_min(iconfig.getParameter<double>("range_min")),
      range_x_max(iconfig.getParameter<double>("range_max")),
      range_y_nbins(iconfig.getParameter<int>("range_y_nbins")),
      range_y_min(iconfig.getParameter<double>("range_y_min")),
      range_y_max(iconfig.getParameter<double>("range_y_max")),
      statsOverflows(iconfig.getParameter<bool>("statsOverflows")) {
  auto spec_configs = iconfig.getParameter<edm::VParameterSet>("specs");
  for (const auto& spec : spec_configs) {
    // this would fit better in SummationSpecification(...), but it has to
    // happen here.
    auto conf = spec.getParameter<edm::ParameterSet>("conf");
    if (!conf.getParameter<bool>("enabled"))
      continue;
    addSpec(SummationSpecification(spec, geometryInterface));
  }
}

void HistogramManager::addSpec(SummationSpecification spec) {
  specs.push_back(spec);
  tables.push_back(Table());
  counters.push_back(Table());
  significantvalues.push_back(GeometryInterface::Values());
  fastpath.push_back(nullptr);
}

// This is the hottest function in the HistogramManager. Make sure it does not
// allocate memory or other expensive things.
// Currently the GeometryInterface::extract (some virtual calls) and the map
// lookups should be the most expensive things, but they should only happen if
// the module changed from the last call; an optimization that fails when row/
// col are used.
// fillInternal (called from here) does more lookups on the geometry, if EXTEND
// is used; we do not attempt the optimization there since in most cases row/
// col are involved.
void HistogramManager::fill(double x, double y, DetId sourceModule, const edm::Event* sourceEvent, int col, int row) {
  if (!enabled)
    return;

  // We only need to check the module here, since the fastpath is only used to
  // determine which plot is filled (not which bin inside) and fillInternal
  // re-extracts whatever it needs for the axis.
  // Since do not support booking based on ROC or time, the plot can only depend
  // on the module.
  // The sourceEvent check is not really effective (pointer is always the same)
  // but needed for initialisation.
  // NOTE that this might change if we want to support per-ROC booking
  // NOTE that we could even cache the bin to fill if iq and spec are identical,
  // also across HistogramManagers.
  bool cached = false;
  if (sourceModule == this->iq.sourceModule && sourceEvent == this->iq.sourceEvent) {
    cached = true;
  }
  iq = GeometryInterface::InterestingQuantities{sourceEvent, sourceModule, int16_t(col), int16_t(row)};
  for (unsigned int i = 0; i < specs.size(); i++) {
    auto& s = specs[i];
    auto& t = s.steps[0].type == SummationStep::COUNT ? counters[i] : tables[i];
    if (!cached) {  // recompute ME to fill (aka fastpath)
      significantvalues[i].clear();
      geometryInterface.extractColumns(s.steps[0].columns, iq, significantvalues[i]);
      auto histo = t.find(significantvalues[i]);
      if (histo == t.end()) {
        if (bookUndefined) {
          edm::LogError("HistogramManager") << "Missing Histogram!\n"
                                            << "name " << tables[i].begin()->second.th1->GetName() << "\n";
          assert(!"Histogram not booked! Probably inconsistent geometry description.");
        }
        fastpath[i] = nullptr;
      } else {
        fastpath[i] = &(histo->second);
      }
    }
    // A fastpath of nullptr means there is no ME for this iq, which can be
    // a valid cached result.
    if (fastpath[i]) {
      if (s.steps[0].type == SummationStep::COUNT) {
        fastpath[i]->count++;
      } else {
        fillInternal(x, y, this->dimensions, iq, s.steps.begin() + 1, s.steps.end(), *(fastpath[i]));
      }
    }
  }
}
void HistogramManager::fill(double x, DetId sourceModule, const edm::Event* sourceEvent, int col, int row) {
  assert(this->dimensions == 1);
  fill(x, 0.0, sourceModule, sourceEvent, col, row);
}
void HistogramManager::fill(DetId sourceModule, const edm::Event* sourceEvent, int col, int row) {
  assert(this->dimensions == 0);
  fill(0.0, 0.0, sourceModule, sourceEvent, col, row);
}

void HistogramManager::fillInternal(double x,
                                    double y,
                                    int n_parameters,
                                    GeometryInterface::InterestingQuantities const& iq,
                                    std::vector<SummationStep>::iterator first,
                                    std::vector<SummationStep>::iterator last,
                                    AbstractHistogram& dest) {
  double fx = 0, fy = 0, fz = 0;
  int tot_parameters = n_parameters;
  for (auto it = first; it != last; ++it) {
    if (it->stage != SummationStep::STAGE1)
      break;
    // The specification builder precomputes where x and y go, this loop will
    // always do 3 iterations to set x, y, z. The builder does not know how
    // many parameters we have, so we have to check that and count the total.
    switch (it->type) {
      case SummationStep::USE_X:
        if (it->arg[0] == '1' && n_parameters >= 1)
          fx = x;
        if (it->arg[0] == '2' && n_parameters >= 2)
          fx = y;
        break;
      case SummationStep::USE_Y:
        if (it->arg[0] == '1' && n_parameters >= 1)
          fy = x;
        if (it->arg[0] == '2' && n_parameters >= 2)
          fy = y;
        break;
      case SummationStep::USE_Z:
        if (it->arg[0] == '1' && n_parameters >= 1)
          fz = x;
        if (it->arg[0] == '2' && n_parameters >= 2)
          fz = y;
        break;
      case SummationStep::EXTEND_X:
        fx = geometryInterface.extract(it->columns[0], iq).second;
        tot_parameters++;
        break;
      case SummationStep::EXTEND_Y:
        fy = geometryInterface.extract(it->columns[0], iq).second;
        tot_parameters++;
        break;
      case SummationStep::PROFILE:
        break;  // profile does not make a difference here, only in booking
      default:
        assert(!"illegal step in STAGE1!");
    }
  }

  switch (tot_parameters) {
    case 1:
      dest.me->Fill(fx);
      break;
    case 2:
      dest.me->Fill(fx, fy);
      break;
    case 3:
      dest.me->Fill(fx, fy, fz);
      break;
    default:
      edm::LogError("HistogramManager") << "got " << tot_parameters << " dimensions\n"
                                        << "name " << dest.th1->GetName() << "\n";
      assert(!"More than 3 dimensions should never occur.");
  }
}

// This is only used for ndigis-like counting. It could be more optimized, but
// is probably fine for a per-event thing.
void HistogramManager::executePerEventHarvesting(const edm::Event* sourceEvent) {
  if (!enabled)
    return;
  for (unsigned int i = 0; i < specs.size(); i++) {
    auto& s = specs[i];
    auto& t = tables[i];
    auto& c = counters[i];
    if (s.steps[0].type != SummationStep::COUNT)
      continue;  // no counting, done
    assert((s.steps.size() >= 2 && s.steps[1].type == SummationStep::GROUPBY) ||
           !"Incomplete spec (but this cannot be caught in Python)");
    for (auto& e : c) {
      // the iq on the counter can only be a _sample_, since many modules
      // could be grouped on one counter. Even worse, the sourceEvent ptr
      // could be dangling if the counter was not touched in this event, so
      // we replace it. row/col are most likely useless as well.
      auto iq = e.second.iq_sample;
      iq.sourceEvent = sourceEvent;

      significantvalues[i].clear();
      geometryInterface.extractColumns(s.steps[1].columns, iq, significantvalues[i]);
      auto histo = t.find(significantvalues[i]);
      if (histo == t.end()) {
        if (!bookUndefined)
          continue;
        edm::LogError("HistogramManager") << "Histogram Missing!\n"
                                          << "name " << t.begin()->second.th1->GetName() << "\n"
                                          << "ctr "
                                          << " detid " << iq.sourceModule << "\n";
        assert(!"Histogram not booked! (per-event) Probably inconsistent geometry description.");
      }
      fillInternal(e.second.count, 0, 1, iq, s.steps.begin() + 2, s.steps.end(), histo->second);
      e.second.count = 0;
    }
  }
}

std::pair<std::string, std::string> HistogramManager::makePathName(SummationSpecification const& s,
                                                                   GeometryInterface::Values const& significantvalues,
                                                                   SummationStep const* upto) {
  std::ostringstream dir("");
  std::string suffix = "";

  // we omit the last value here, to get all disks next to each other etc.
  if (!significantvalues.empty()) {
    for (auto it = significantvalues.begin(); it != (significantvalues.end() - 1); ++it) {
      auto name = geometryInterface.formatValue(it->first, it->second);
      if (name.empty())
        continue;
      dir << name << "/";
    }
    auto e = significantvalues[significantvalues.size() - 1];
    suffix = "_" + geometryInterface.formatValue(e.first, e.second);
  }

  // PERF: this is actually independent of significantvalues and iq
  std::string name = this->name;
  for (SummationStep const& step : s.steps) {
    if (&step == upto)
      break;
    switch (step.type) {
      case SummationStep::COUNT:
        name = "num_" + name;
        break;
      case SummationStep::EXTEND_X:
      case SummationStep::EXTEND_Y: {
        if (step.stage != SummationStep::STAGE1)
          break;
        GeometryInterface::Column col0 = step.columns[0];
        std::string colname = geometryInterface.pretty(col0);
        name = name + "_per_" + colname;
        break;
      }
      case SummationStep::REDUCE: {
        auto red = step.arg;
        boost::algorithm::to_lower(red);
        name = red + "_" + name;
        break;
      }
      default:
        // Maybe PROFILE is worth showing.
        break;  // not visible in name
    }
  }
  return std::make_pair(top_folder_name + "/" + dir.str(), name + suffix);
}

void HistogramManager::book(DQMStore::IBooker& iBooker, edm::EventSetup const& iSetup) {
  if (!geometryInterface.loaded()) {
    geometryInterface.load(iSetup);
  }
  if (!enabled)
    return;

  struct MEInfo {
    int dimensions = 1;
    double range_x_min = 1e12;
    double range_x_max = -1e12;
    double range_y_min = 1e12;
    double range_y_max = -1e12;
    double range_z_min = 1e12;  // z range carried around but unused
    double range_z_max = -1e12;
    int range_x_nbins = 0;
    int range_y_nbins = 0;
    int range_z_nbins = 0;
    GeometryInterface::Value binwidth_x = 0;  // override nbins for geom-things
    GeometryInterface::Value binwidth_y = 0;
    std::string title, xlabel, ylabel, zlabel;
    bool do_profile = false;
    bool statsOverflows = true;
    ;
  };
  std::map<GeometryInterface::Values, MEInfo> toBeBooked;

  for (unsigned int i = 0; i < specs.size(); i++) {
    auto& s = specs[i];
    auto& t = tables[i];
    auto& c = counters[i];
    toBeBooked.clear();
    bool bookCounters = false;

    auto firststep = s.steps.begin();
    int n_parameters = this->dimensions;

    if (firststep->type != SummationStep::GROUPBY) {
      ++firststep;
      n_parameters = 1;
      bookCounters = true;
    }

    auto laststep = std::find_if(
        s.steps.begin(), s.steps.end(), [](SummationStep const& step) { return step.stage == SummationStep::STAGE2; });

    GeometryInterface::Values significantvalues;

    for (auto iq : geometryInterface.allModules()) {
      if (bookCounters) {
        // add an entry for the counting step if present
        geometryInterface.extractColumns(s.steps[0].columns, iq, significantvalues);
        c[significantvalues].iq_sample = iq;
      }

      geometryInterface.extractColumns(firststep->columns, iq, significantvalues);
      if (!bookUndefined) {
        // skip if any column is UNDEFINED
        bool ok = true;
        for (auto e : significantvalues)
          if (e.second == GeometryInterface::UNDEFINED)
            ok = false;
        if (!ok)
          continue;
      }

      auto histo = toBeBooked.find(significantvalues);
      if (histo == toBeBooked.end()) {
        // create new histo
        MEInfo& mei = toBeBooked[significantvalues];
        mei.title = this->title;
        mei.statsOverflows = this->statsOverflows;
        if (bookCounters)
          mei.title =
              "Number of " + mei.title + " per Event and " + geometryInterface.pretty(*(s.steps[0].columns.end() - 1));
        std::string xlabel = bookCounters ? "#" + this->xlabel : this->xlabel;

        // refer to fillInternal() for the actual execution
        // compute labels, title, type, user-set ranges here
        int tot_parameters = n_parameters;

#define SET_AXIS(to, from)                                                          \
  mei.to##label = from##label;                                                      \
  mei.range_##to##_min = ((it->nbins == -1) ? this->range_##from##_min : it->xmin); \
  mei.range_##to##_max = ((it->nbins == -1) ? this->range_##from##_max : it->xmax); \
  mei.range_##to##_nbins = ((it->nbins == -1) ? this->range_##from##_nbins : it->nbins)

        for (auto it = firststep + 1; it != laststep; ++it) {
          switch (it->type) {
            case SummationStep::USE_X:
              if (it->arg[0] == '1' && n_parameters >= 1) {
                SET_AXIS(x, x);
              }  // TODO: make use of current nbins, xmin, xmax if set
              if (it->arg[0] == '2' && n_parameters >= 2) {
                SET_AXIS(x, y);
              }
              break;
            case SummationStep::USE_Y:
              if (it->arg[0] == '1' && n_parameters >= 1) {
                SET_AXIS(y, x);
              }
              if (it->arg[0] == '2' && n_parameters >= 2) {
                SET_AXIS(y, y);
              }
              break;
            case SummationStep::USE_Z:
              if (it->arg[0] == '1' && n_parameters >= 1) {
                SET_AXIS(z, x);
              }
              if (it->arg[0] == '2' && n_parameters >= 2) {
                SET_AXIS(z, y);
              }
              break;
            case SummationStep::EXTEND_X: {
              assert(mei.range_x_nbins == 0);
              auto col = it->columns[0];
              mei.xlabel = geometryInterface.pretty(col);
              mei.title = mei.title + " by " + mei.xlabel;
              if (geometryInterface.minValue(col) != GeometryInterface::UNDEFINED)
                mei.range_x_min = geometryInterface.minValue(col);
              if (geometryInterface.maxValue(col) != GeometryInterface::UNDEFINED)
                mei.range_x_max = geometryInterface.maxValue(col);
              mei.binwidth_x = geometryInterface.binWidth(col);
              tot_parameters++;
            } break;
            case SummationStep::EXTEND_Y: {
              auto col = it->columns[0];
              mei.ylabel = geometryInterface.pretty(col);
              mei.title = mei.title + " by " + mei.ylabel;
              if (geometryInterface.minValue(col) != GeometryInterface::UNDEFINED)
                mei.range_y_min = geometryInterface.minValue(col);
              if (geometryInterface.maxValue(col) != GeometryInterface::UNDEFINED)
                mei.range_y_max = geometryInterface.maxValue(col);
              mei.binwidth_y = geometryInterface.binWidth(col);
              tot_parameters++;
            } break;
            case SummationStep::PROFILE:
              mei.do_profile = true;
              break;
            default:
              assert(!"illegal step in STAGE1! (booking)");
          }
        }
        mei.dimensions = tot_parameters;
        if (mei.do_profile)
          mei.title = "Profile of " + mei.title;
        if (!mei.zlabel.empty())
          mei.title = mei.title + " (Z: " + mei.zlabel + ")";
      }
      // only update range
      MEInfo& mei = toBeBooked[significantvalues];
      double val;

      for (auto it = firststep + 1; it != laststep; ++it) {
        switch (it->type) {
          case SummationStep::EXTEND_X:
            val = geometryInterface.extract(it->columns[0], iq).second;
            if (val == GeometryInterface::UNDEFINED)
              break;
            mei.range_x_min = std::min(mei.range_x_min, val);
            mei.range_x_max = std::max(mei.range_x_max, val);
            break;
          case SummationStep::EXTEND_Y:
            val = geometryInterface.extract(it->columns[0], iq).second;
            if (val == GeometryInterface::UNDEFINED)
              break;
            mei.range_y_min = std::min(mei.range_y_min, val);
            mei.range_y_max = std::max(mei.range_y_max, val);
            break;
          default:  // ignore the rest, code above will catch bugs
            break;
        }
      }
    }

    // Now do the actual booking.
    for (auto& e : toBeBooked) {
      AbstractHistogram& h = t[e.first];
      MEInfo& mei = e.second;
      auto name = makePathName(s, e.first, &(*laststep));
      iBooker.setCurrentFolder(name.first);

      // determine nbins for geometry derived quantities
      // due to how we counted above, we need to include lower and upper bound
      // For Coord-values, which are not precisely aligned with the bins, we force
      // alignment.
      if (mei.binwidth_x != 0) {
        double range = (mei.range_x_max - mei.range_x_min) / mei.binwidth_x;
        if ((range - int(range)) == 0.0) {
          mei.range_x_min -= mei.binwidth_x / 2;
          mei.range_x_max += mei.binwidth_x / 2;
        } else {
          mei.range_x_min = std::floor(mei.range_x_min / mei.binwidth_x) * mei.binwidth_x;
          mei.range_x_max = std::ceil(mei.range_x_max / mei.binwidth_x) * mei.binwidth_x;
        }
        mei.range_x_nbins = int((mei.range_x_max - mei.range_x_min) / mei.binwidth_x);
      }
      if (mei.binwidth_y != 0) {
        double range = (mei.range_y_max - mei.range_y_min) / mei.binwidth_y;
        if ((range - int(range)) == 0.0) {
          mei.range_y_min -= mei.binwidth_y / 2;
          mei.range_y_max += mei.binwidth_y / 2;
        } else {
          mei.range_y_min = std::floor(mei.range_y_min / mei.binwidth_y) * mei.binwidth_y;
          mei.range_y_max = std::ceil(mei.range_y_max / mei.binwidth_y) * mei.binwidth_y;
        }
        mei.range_y_nbins = int((mei.range_y_max - mei.range_y_min) / mei.binwidth_y);
      }

      if (mei.dimensions == 1) {
        h.me = iBooker.book1D(
            name.second, (mei.title + ";" + mei.xlabel).c_str(), mei.range_x_nbins, mei.range_x_min, mei.range_x_max);
      } else if (mei.dimensions == 2 && !mei.do_profile) {
        h.me = iBooker.book2D(name.second,
                              (mei.title + ";" + mei.xlabel + ";" + mei.ylabel).c_str(),
                              mei.range_x_nbins,
                              mei.range_x_min,
                              mei.range_x_max,
                              mei.range_y_nbins,
                              mei.range_y_min,
                              mei.range_y_max);
      } else if (mei.dimensions == 2 && mei.do_profile) {
        h.me = iBooker.bookProfile(name.second,
                                   (mei.title + ";" + mei.xlabel + ";" + mei.ylabel).c_str(),
                                   mei.range_x_nbins,
                                   mei.range_x_min,
                                   mei.range_x_max,
                                   0.0,
                                   0.0,
                                   "");
      } else if (mei.dimensions == 3 && mei.do_profile) {
        h.me = iBooker.bookProfile2D(name.second,
                                     (mei.title + ";" + mei.xlabel + ";" + mei.ylabel).c_str(),
                                     mei.range_x_nbins,
                                     mei.range_x_min,
                                     mei.range_x_max,
                                     mei.range_y_nbins,
                                     mei.range_y_min,
                                     mei.range_y_max,
                                     0.0,
                                     0.0);  // Z range is ignored if min==max
      } else {
        edm::LogError("HistogramManager") << "Illegal Histogram!\n"
                                          << "name " << name.second << "\n"
                                          << "dim " << mei.dimensions << " profile " << mei.do_profile << "\n";
        assert(!"Illegal Histogram kind.");
      }
      h.th1 = h.me->getTH1();
      h.me->setStatOverflows(mei.statsOverflows);
    }
  }
}

void HistogramManager::executePerLumiHarvesting(DQMStore::IBooker& iBooker,
                                                DQMStore::IGetter& iGetter,
                                                edm::LuminosityBlock const& lumiBlock,
                                                edm::EventSetup const& iSetup) {
  if (!enabled)
    return;
  // this should also give us the GeometryInterface for offline, though it is a
  // bit dirty and might explode.
  if (!geometryInterface.loaded()) {
    geometryInterface.load(iSetup);
  }
  if (perLumiHarvesting) {
    this->lumisection = &lumiBlock;  // "custom" steps can use this
    executeHarvesting(iBooker, iGetter);
    this->lumisection = nullptr;
  }
}

void HistogramManager::loadFromDQMStore(SummationSpecification& s, Table& t, DQMStore::IGetter& iGetter) {
  t.clear();
  GeometryInterface::Values significantvalues;
  auto firststep = s.steps.begin();
  if (firststep->type != SummationStep::GROUPBY)
    ++firststep;
  auto laststep = std::find_if(
      s.steps.begin(), s.steps.end(), [](SummationStep const& step) { return step.stage == SummationStep::STAGE2; });

  for (auto iq : geometryInterface.allModules()) {
    geometryInterface.extractColumns(firststep->columns, iq, significantvalues);

    auto histo = t.find(significantvalues);
    if (histo == t.end()) {
      auto name = makePathName(s, significantvalues, &(*laststep));
      std::string path = name.first + name.second;
      MonitorElement* me = iGetter.get(path);
      if (!me) {
        if (bookUndefined)
          edm::LogError("HistogramManager") << "ME " << path << " not found\n";
        // else this will happen quite often
      } else {
        AbstractHistogram& histo = t[significantvalues];
        histo.me = me;
        histo.th1 = histo.me->getTH1();
        histo.iq_sample = iq;
      }
    }
  }
}

void HistogramManager::executeGroupBy(SummationStep const& step,
                                      Table& t,
                                      DQMStore::IBooker& iBooker,
                                      SummationSpecification const& s) {
  // Simple regrouping, sum histos if they end up in the same place.
  Table out;
  GeometryInterface::Values significantvalues;
  for (auto& e : t) {
    TH1* th1 = e.second.th1;
    geometryInterface.extractColumns(step.columns, e.second.iq_sample, significantvalues);
    AbstractHistogram& new_histo = out[significantvalues];
    if (!new_histo.me) {
      auto name = makePathName(s, significantvalues, &step);
      iBooker.setCurrentFolder(name.first);
      if (dynamic_cast<TH1F*>(th1))
        new_histo.me = iBooker.book1D(name.second, (TH1F*)th1);
      else if (dynamic_cast<TH2F*>(th1))
        new_histo.me = iBooker.book2D(name.second, (TH2F*)th1);
      else if (dynamic_cast<TProfile*>(th1))
        new_histo.me = iBooker.bookProfile(name.second, (TProfile*)th1);
      else if (dynamic_cast<TProfile2D*>(th1))
        new_histo.me = iBooker.bookProfile2D(name.second, (TProfile2D*)th1);
      else
        assert(!"No idea how to book this.");
      new_histo.th1 = new_histo.me->getTH1();
      new_histo.iq_sample = e.second.iq_sample;
    } else {
      new_histo.th1->Add(th1);
    }
    new_histo.me->setStatOverflows(e.second.me->getStatOverflows());
  }
  t.swap(out);
}

void HistogramManager::executeExtend(SummationStep const& step,
                                     Table& t,
                                     std::string const& reduce_type,
                                     DQMStore::IBooker& iBooker,
                                     SummationSpecification const& s) {
  // For the moment only X.
  // first pass determines the range.
  std::map<GeometryInterface::Values, int> nbins;
  // separators collects meta info for the render plugin about the boundaries.
  // for each EXTEND, this is added to the axis label. In total this is not
  // fully correct since we have to assume the the substructure of each sub-
  // histogram is the same, which is e.g. not true for layers. It still works
  // since layers only contain leaves (ladders).
  std::map<GeometryInterface::Values, std::string> separators;

  GeometryInterface::Values significantvalues;

  for (auto& e : t) {
    geometryInterface.extractColumns(step.columns, e.second.iq_sample, significantvalues);
    int& n = nbins[significantvalues];
    assert(e.second.th1 || !"invalid histogram");
    // if we reduce, every histogram only needs one bin
    int bins = !reduce_type.empty() ? 1 : e.second.th1->GetXaxis()->GetNbins();
    if (bins > 1)
      separators[significantvalues] += std::to_string(n) + ",";
    n += bins;
  }
  for (auto& e : separators)
    e.second = "(" + e.second + ")/";

  Table out;
  for (auto& e : t) {
    geometryInterface.extractColumns(step.columns, e.second.iq_sample, significantvalues);
    TH1* th1 = e.second.th1;
    assert(th1);

    AbstractHistogram& new_histo = out[significantvalues];
    if (!new_histo.me) {
      // we put the name of the actual, last column of a input histo there.
      std::string colname = geometryInterface.pretty((e.first.end() - 1)->first);

      auto separator = separators[significantvalues];

      auto name = makePathName(s, significantvalues, &step);
      auto title =
          std::string("") + th1->GetTitle() + " per " + colname + ";" + colname + separator +
          (!reduce_type.empty() ? th1->GetYaxis()->GetTitle() : th1->GetXaxis()->GetTitle()) + ";" +
          (!reduce_type.empty() ? reduce_type + " of " + th1->GetXaxis()->GetTitle() : th1->GetYaxis()->GetTitle());
      iBooker.setCurrentFolder(name.first);

      if (th1->GetDimension() == 1) {
        new_histo.me =
            iBooker.book1D(name.second, title, nbins[significantvalues], 0.5, nbins[significantvalues] + 0.5);
      } else {
        assert(!"2D extend not implemented in harvesting.");
      }
      new_histo.th1 = new_histo.me->getTH1();
      new_histo.iq_sample = e.second.iq_sample;
      new_histo.count = 1;  // used as a fill pointer. Assumes histograms are
                            // ordered correctly (map should provide that)
    }

    // now add data.
    if (new_histo.th1->GetDimension() == 1) {
      if (reduce_type.empty()) {  // no reduction, concatenate
        for (int i = 1; i <= th1->GetXaxis()->GetNbins(); i++) {
          new_histo.th1->SetBinContent(new_histo.count, th1->GetBinContent(i));
          new_histo.th1->SetBinError(new_histo.count, th1->GetBinError(i));
          new_histo.count += 1;
        }
      } else if (reduce_type == "MEAN") {
        new_histo.th1->SetBinContent(new_histo.count, th1->GetMean());
        new_histo.th1->SetBinError(new_histo.count, th1->GetMeanError());
        new_histo.count += 1;
      } else {
        assert(!"Reduction type not supported");
      }
      new_histo.me->setStatOverflows(e.second.me->getStatOverflows());
    } else {
      assert(!"2D extend not implemented in harvesting.");
    }
  }
  t.swap(out);
}

void HistogramManager::executeHarvesting(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {
  if (!enabled)
    return;
  // Debug output
  for (auto& s : specs) {
    edm::LogInfo log("HistogramManager");
    log << "Specs for " << name << " ";
    s.dump(log, geometryInterface);
  }

  for (unsigned int i = 0; i < specs.size(); i++) {
    auto& s = specs[i];
    auto& t = tables[i];
    loadFromDQMStore(s, t, iGetter);

    std::string reduce_type = "";

    // now execute step2.
    for (SummationStep const& step : s.steps) {
      if (step.stage == SummationStep::STAGE2) {
        switch (step.type) {
          case SummationStep::SAVE:
            // no explicit implementation atm.
            break;
          case SummationStep::GROUPBY:
            executeGroupBy(step, t, iBooker, s);
            break;
          case SummationStep::REDUCE:
            // reduction is done in the following EXTEND
            reduce_type = step.arg;
            break;
          case SummationStep::EXTEND_X:
            executeExtend(step, t, reduce_type, iBooker, s);
            reduce_type = "";
            break;
          case SummationStep::EXTEND_Y:
            assert(!"EXTEND_Y currently not supported in harvesting.");
            break;
          default:
            assert(!"Operation not supported in harvesting.");
        }
      }
    }
  }
}
