// -*- C++ -*-
//
// Package:     SiPixelPhase1Common
// Class  :     HistogramManager
//
#include "DQM/SiPixelPhase1Common/interface/HistogramManager.h"

#include <sstream>

// Geometry stuff
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// Logger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HistogramManager::HistogramManager(const edm::ParameterSet& iconfig,
                                   GeometryInterface& geo)
    : iConfig(iconfig),
      geometryInterface(geo),
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
      range_y_max(iconfig.getParameter<double>("range_y_max")) {
  auto spec_configs = iconfig.getParameter<edm::VParameterSet>("specs");
  for (auto spec : spec_configs) {
    // this would fit better in SummationSpecification(...), but it has to
    // happen here.
    auto conf = spec.getParameter<edm::ParameterSet>("conf");
    if (!conf.getParameter<bool>("enabled")) continue;
    addSpec(SummationSpecification(spec, geometryInterface));
  }
}

void HistogramManager::addSpec(SummationSpecification spec) {
  specs.push_back(spec);
  tables.push_back(Table());
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
void HistogramManager::fill(double x, double y, DetId sourceModule,
                            const edm::Event* sourceEvent, int col, int row) {
  if (!enabled) return;
  bool cached = true;
  // We could be smarter on row/col and only check if they appear in the spec
  // but that just asks for bugs.
  if (col != this->iq.col || row != this->iq.row ||
      sourceModule != this->iq.sourceModule ||
      sourceEvent != this->iq.sourceEvent ||
      // TODO: add the RawData fake DetId here
      sourceModule == DetId(0)  // Hack for eventrate-like things, since the
                                // sourceEvent ptr might not change.
      ) {
    cached = false;
    iq = GeometryInterface::InterestingQuantities{sourceEvent, sourceModule, 
                                                  int16_t(col), int16_t(row)};
  }
  for (unsigned int i = 0; i < specs.size(); i++) {
    auto& s = specs[i];
    auto& t = tables[i];
    if (!cached) {
      significantvalues[i].clear();
      geometryInterface.extractColumns(s.steps[0].columns, iq,
                                       significantvalues[i]);
      fastpath[i] = nullptr;
    }
    
    if (!fastpath[i]) {
      auto histo = t.find(significantvalues[i]);
      if (histo == t.end()) {
        if (!bookUndefined) continue;
        std::cout << "+++ path " << makePath(significantvalues[i]) << "\n";
        std::cout << "+++ name " << t.begin()->second.th1->GetName() << "\n";
        assert(!"Histogram not booked! Probably inconsistent geometry description.");
      }

      fastpath[i] = &(histo->second);
    }
    if (s.steps[0].type == SummationStep::COUNT) {
      fastpath[i]->count++;
      fastpath[i]->iq_sample = iq;
    } else {
      fillInternal(x, y, this->dimensions, iq, s.steps.begin()+1, s.steps.end(), *(fastpath[i]));
    }
  }
}
void HistogramManager::fill(double x, DetId sourceModule,
                            const edm::Event* sourceEvent, int col, int row) {
  assert(this->dimensions == 1);
  fill(x, 0.0, sourceModule, sourceEvent, col, row);
}
void HistogramManager::fill(DetId sourceModule, const edm::Event* sourceEvent,
                            int col, int row) {
  assert(this->dimensions == 0);
  fill(0.0, 0.0, sourceModule, sourceEvent, col, row);
}

void HistogramManager::fillInternal(double x, double y, int n_parameters,
  GeometryInterface::InterestingQuantities const& iq,
  std::vector<SummationStep>::iterator first,
  std::vector<SummationStep>::iterator last,
  AbstractHistogram& dest) {

  double fx = 0, fy = 0, fz = 0;
  int tot_parameters = n_parameters;
  for (auto it = first; it != last; ++it) {
    if (it->stage != SummationStep::STAGE1) break;
    // The specification builder precomputes where x and y go, this loop will
    // always do 3 iterations to set x, y, z. The builder does not know how 
    // many parameters we have, so we have to check that and count the total.
    switch (it->type) {
      case SummationStep::USE_X:
        if (it->arg[0] == '1' && n_parameters >= 1) fx = x;
        if (it->arg[0] == '2' && n_parameters >= 2) fx = y;
        break;
      case SummationStep::USE_Y:
        if (it->arg[0] == '1' && n_parameters >= 1) fy = x;
        if (it->arg[0] == '2' && n_parameters >= 2) fy = y;
        break;
      case SummationStep::USE_Z:
        if (it->arg[0] == '1' && n_parameters >= 1) fz = x;
        if (it->arg[0] == '2' && n_parameters >= 2) fz = y;
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
        break; // profile does not make a difference here, only in booking
      default:
        assert(!"illegal step in STAGE1!");
    }
  }

  switch(tot_parameters) {
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
      std::cout << "+++ got " << tot_parameters << " dimensions\n";
      std::cout << "+++ name " << dest.th1->GetName() << "\n";
      assert(!"More than 3 dimensions should never occur.");
  }
}


// This is only used for ndigis-like counting. It could be more optimized, but
// is probably fine for a per-event thing.
void HistogramManager::executePerEventHarvesting(const edm::Event* sourceEvent) {
  if (!enabled) return;
  for (unsigned int i = 0; i < specs.size(); i++) {
    auto& s = specs[i];
    auto& t = tables[i];
    assert((s.steps.size() >= 2 && s.steps[1].type == SummationStep::GROUPBY)
          || !"Incomplete spec (but this cannot be caught in Python)");
    for (auto& e : t) {
      // TODO: this is terribly risky. It works if there is a differnt number
      // of columns in COUNT and GROUPBY each or if the columns are identical,
      // but not in other, possible cases. Solution: separate table for ctrs.
      if (e.first.values.size() == s.steps[0].columns.size()) {
        // the iq on the counter can only be a _sample_, since many modules
        // could be grouped on one counter. Even worse, the sourceEvent ptr
        // could be dangling if the counter was not touched in this event, so
        // we replace it. row/col are most likely useless as well.
        auto iq = e.second.iq_sample;
        iq.sourceEvent = sourceEvent;

        significantvalues[i].clear();
        geometryInterface.extractColumns(s.steps[1].columns, iq,
                                         significantvalues[i]);
        auto histo = t.find(significantvalues[i]);
        if (histo == t.end()) {
          if (!bookUndefined) continue;
          std::cout << "+++ path " << makePath(significantvalues[i]) << "\n";
          std::cout << "+++ name " << t.begin()->second.th1->GetName() << "\n";
          assert(!"Histogram not booked! (per-event) Probably inconsistent geometry description.");
        }
        fillInternal(e.second.count, 0, 1, iq, s.steps.begin()+2, s.steps.end(), histo->second);
        e.second.count = 0;
      }
    }
  }
}

std::string HistogramManager::makePath(
    GeometryInterface::Values const& significantvalues) {
  // non-number output names (_pO etc.) are hardwired here.
  // PERF: memoize the names in a map, probably ignoring row/col
  // TODO: more pretty names for DetIds using PixelBarrelName etc.
  std::ostringstream dir("");
  for (auto e : significantvalues.values) {
    std::string name = geometryInterface.pretty(e.first);
    std::string value = "_" + std::to_string(e.second);
    if (e.second == 0) value = "";         // hide Barrel_0 etc.
    if (name == "") continue;              // nameless dummy column is dropped
    if (name == "PXDisk" && e.second > 0)  // +/- sign for disk num
      value = "_+" + std::to_string(e.second);
    // pretty (legacy?) names for Shells and HalfCylinders
    std::map<int, std::string> shellname{
        {11, "_mI"}, {12, "_mO"}, {21, "_pI"}, {22, "_pO"}};
    if (name == "HalfCylinder" || name == "Shell") value = shellname[e.second];
    if (e.second == GeometryInterface::UNDEFINED) value = "_UNDEFINED";

    dir << name << value << "/";
  }
  return top_folder_name + "/" + dir.str();
}

std::string HistogramManager::makeName(SummationSpecification const& s,
    GeometryInterface::InterestingQuantities const& iq) {
  std::string name = this->name;
  for (SummationStep step : s.steps) {
    if (step.stage == SummationStep::FIRST || step.stage == SummationStep::STAGE1) {
      switch (step.type) {
        case SummationStep::COUNT:
          name = "num_" + name;
          break;
        case SummationStep::EXTEND_X:
        case SummationStep::EXTEND_Y: {
          GeometryInterface::Column col0 =
              geometryInterface.extract(step.columns[0], iq).first;
          std::string colname = geometryInterface.pretty(col0);
          name = name + "_per_" + colname;
          break;
        }
        default:
          // Maybe PROFILE is worth showing.
          break; // not visible in name
      }
    }
  }
  return name;
}


void HistogramManager::book(DQMStore::IBooker& iBooker,
                            edm::EventSetup const& iSetup) {
  if (!geometryInterface.loaded()) {
    geometryInterface.load(iSetup);
  }
  if (!enabled) return;

  struct MEInfo {
    int dimensions = 1;
    double range_x_min = 1e12;
    double range_x_max = -1e12;
    double range_y_min = 1e12;
    double range_y_max = -1e12;
    double range_z_min = 1e12; // z range carried around but unused
    double range_z_max = -1e12;
    int range_x_nbins = 0;
    int range_y_nbins = 0;
    int range_z_nbins = 0;
    std::string name, title, xlabel, ylabel, zlabel;
    bool do_profile = false;
  };
  std::map<GeometryInterface::Values, MEInfo> toBeBooked;

  for (unsigned int i = 0; i < specs.size(); i++) {
    auto& s = specs[i];
    auto& t = tables[i];
    toBeBooked.clear();
    bool bookCounters = false;

    auto firststep = s.steps.begin();
    int n_parameters = this->dimensions;
    if (firststep->type != SummationStep::GROUPBY) {
      ++firststep;
      n_parameters = 1;
      bookCounters = true;
    }
    GeometryInterface::Values significantvalues;

    for (auto iq : geometryInterface.allModules()) {

      if (bookCounters) {
        // add an entry for the counting step if present
        // TODO: use a independent table here.
        geometryInterface.extractColumns(s.steps[0].columns, iq,
                                         significantvalues);
        t[significantvalues].iq_sample = iq;
      }

      geometryInterface.extractColumns(firststep->columns, iq,
                                       significantvalues);
      if (!bookUndefined) {
        // skip if any column is UNDEFINED
        bool ok = true;
        for (auto e : significantvalues.values)
          if (e.second == GeometryInterface::UNDEFINED) ok = false;
        if (!ok) continue;
      }

      auto histo = toBeBooked.find(significantvalues);
      if (histo == toBeBooked.end()) {
        // create new histo
        MEInfo& mei = toBeBooked[significantvalues]; 
        mei.name = makeName(s, iq);
        mei.title = this->title;

        // refer to fillInternal() for the actual execution
        // compute labels, title, type, user-set ranges here
        int tot_parameters = n_parameters;
#define SET_AXIS(to, from) \
                mei.to##label = this->from##label; \
                mei.range_##to##_min = this->range_##from##_min; \
                mei.range_##to##_max = this->range_##from##_max; \
                mei.range_##to##_nbins = this->range_##from##_nbins 
        for (auto it = firststep+1; it != s.steps.end(); ++it) {
          if (it->stage != SummationStep::STAGE1) break;
          switch (it->type) {
            case SummationStep::USE_X:
              if (it->arg[0] == '1' && n_parameters >= 1) { SET_AXIS(x, x); }
              if (it->arg[0] == '2' && n_parameters >= 2) { SET_AXIS(x, y); }
              break;
            case SummationStep::USE_Y:
              if (it->arg[0] == '1' && n_parameters >= 1) { SET_AXIS(y, x); }
              if (it->arg[0] == '2' && n_parameters >= 2) { SET_AXIS(y, y); }
              break;
            case SummationStep::USE_Z:
              if (it->arg[0] == '1' && n_parameters >= 1) { SET_AXIS(z, x); }
              if (it->arg[0] == '2' && n_parameters >= 2) { SET_AXIS(z, y); }
              break;
            case SummationStep::EXTEND_X: {
              assert(mei.range_x_nbins == 0);
              auto col = geometryInterface.extract(it->columns[0], iq).first;
              mei.xlabel = geometryInterface.pretty(col);
              if(geometryInterface.minValue(col[0]) != GeometryInterface::UNDEFINED)
                mei.range_x_min = geometryInterface.minValue(col[0]);
              if(geometryInterface.maxValue(col[0]) != GeometryInterface::UNDEFINED)
                mei.range_x_max = geometryInterface.maxValue(col[0]);
              tot_parameters++; }
              break;
            case SummationStep::EXTEND_Y: {
              auto col = geometryInterface.extract(it->columns[0], iq).first;
              mei.ylabel = geometryInterface.pretty(col);
              if(geometryInterface.minValue(col[0]) != GeometryInterface::UNDEFINED)
                mei.range_y_min = geometryInterface.minValue(col[0]);
              if(geometryInterface.maxValue(col[0]) != GeometryInterface::UNDEFINED)
                mei.range_y_max = geometryInterface.maxValue(col[0]);
              tot_parameters++; }
              break;
            case SummationStep::PROFILE:
              mei.do_profile = true;
              break;
            default:
              assert(!"illegal step in STAGE1! (booking)");
          }
        }
        mei.dimensions = tot_parameters;
      } 
      // only update range
      MEInfo& mei = toBeBooked[significantvalues]; 
      double val;

      for (auto it = firststep+1; it != s.steps.end(); ++it) {
        if (it->stage != SummationStep::STAGE1) break;
        switch (it->type) {
          case SummationStep::EXTEND_X:
            val = geometryInterface.extract(it->columns[0], iq).second;
            if (val == GeometryInterface::UNDEFINED) break;
            mei.range_x_min = std::min(mei.range_x_min, val);
            mei.range_x_max = std::max(mei.range_x_max, val);
            break;
          case SummationStep::EXTEND_Y:
            val = geometryInterface.extract(it->columns[0], iq).second;
            if (val == GeometryInterface::UNDEFINED) break;
            mei.range_y_min = std::min(mei.range_y_min, val);
            mei.range_y_max = std::max(mei.range_y_max, val);
            break;
          case SummationStep::PROFILE:
            mei.do_profile = true;
            break;
          case SummationStep::USE_X:
          case SummationStep::USE_Y:
          case SummationStep::USE_Z:
            break;
          default:
            assert(!"illegal step in STAGE1! (booking)");
        }
      }
    }
    
    // Now do the actual booking.
    for (auto& e : toBeBooked) {
      AbstractHistogram& h = t[e.first];
      iBooker.setCurrentFolder(makePath(e.first));
      MEInfo& mei = e.second;

      if (mei.range_x_nbins == 0) {
        mei.range_x_min -= 0.5;
        mei.range_x_max += 0.5;
        mei.range_x_nbins = int(mei.range_x_max - mei.range_x_min);
      }
      if (mei.range_y_nbins == 0) {
        mei.range_y_min -= 0.5;
        mei.range_y_max += 0.5;
        mei.range_y_nbins = int(mei.range_y_max - mei.range_y_min);
      }

      if (mei.dimensions == 1) {
        h.me = iBooker.book1D(mei.name, (mei.title + ";" + mei.xlabel).c_str(),
                       mei.range_x_nbins, mei.range_x_min, mei.range_x_max);
      } else if (mei.dimensions == 2 && !mei.do_profile) {
        h.me = iBooker.book2D(mei.name, (mei.title + ";" + mei.xlabel + ";" + mei.ylabel).c_str(),
                       mei.range_x_nbins, mei.range_x_min, mei.range_x_max,
                       mei.range_y_nbins, mei.range_y_min, mei.range_y_max);
      } else if (mei.dimensions == 2 && mei.do_profile) {
        h.me = iBooker.bookProfile(mei.name, (mei.title + ";" + mei.xlabel + ";" + mei.ylabel).c_str(),
                       mei.range_x_nbins, mei.range_x_min, mei.range_x_max, 0.0, 0.0);
      } else if (mei.dimensions == 3 && mei.do_profile) {
        h.me = iBooker.bookProfile2D(mei.name, (mei.title + ";" + mei.xlabel + ";" + mei.ylabel).c_str(),
                       mei.range_x_nbins, mei.range_x_min, mei.range_x_max,
                       mei.range_y_nbins, mei.range_y_min, mei.range_y_max,
                       0.0, 0.0); // Z range is ignored if min==max
      } else {
        std::cout << "+++ name " << mei.name << "\n";
        std::cout << "+++ dim " << mei.dimensions << " profile " << mei.do_profile << "\n";
        assert(!"Illegal Histogram kind.");
      }
      h.th1 = h.me->getTH1();
    }
  }
}



void HistogramManager::executePerLumiHarvesting(DQMStore::IBooker& iBooker,
                                                DQMStore::IGetter& iGetter,
                                                edm::LuminosityBlock const& lumiBlock,
                                                edm::EventSetup const& iSetup) {
  if (!enabled) return;
  // this should also give us the GeometryInterface for offline, though it is a
  // bit dirty and might explode.
  if (!geometryInterface.loaded()) {
    geometryInterface.load(iSetup);
  }
  if (perLumiHarvesting) {
    this->lumisection = &lumiBlock; // "custom" steps can use this
    executeHarvesting(iBooker, iGetter);
    this->lumisection = nullptr;
  }
}

void HistogramManager::loadFromDQMStore(SummationSpecification& s, Table& t,
                                        DQMStore::IGetter& iGetter) {
}

void HistogramManager::executeSave(SummationStep& step, Table& t,
                                   DQMStore::IBooker& iBooker) {
}

void HistogramManager::executeGroupBy(SummationStep& step, Table& t) {
}

void HistogramManager::executeReduce(SummationStep& step, Table& t) {
}

void HistogramManager::executeExtend(SummationStep& step, Table& t, bool isX) {
}

#if 0

void HistogramManager::loadFromDQMStore(SummationSpecification& s, Table& t,
                                        DQMStore::IGetter& iGetter) {
  // This is essentially the booking code of step1, to reconstruct the ME names.
  // Once we have a name we load the ME and put it into the table.
  t.clear();
  for (auto iq : geometryInterface.allModules()) {
    // note that we call get() here for every single module. But the string
    // ops above are probably more expensive anyways...
    auto path_name = makeName(s, iq);
    std::string path = makePath(path_name.first) + path_name.second;
    MonitorElement* me = iGetter.get(path);
    if (!me) {
      if (bookUndefined)
        edm::LogError("HistogramManager") << "ME " << path << " not found\n";
      // else this will happen quite often
    } else {
      // only touch the table if a me is added. Empty items are illegal.
      AbstractHistogram& histo = t[path_name.first];
      histo.me = me;
      histo.th1 = histo.me->getTH1();
    }
  }
}

void HistogramManager::executeSave(SummationStep& step, Table& t,
                                   DQMStore::IBooker& iBooker) {
  // SAVE: traverse the table, book a ME for every TH1.
  for (auto& e : t) {
    if (e.second.me) continue;  // if there is a ME already, nothing to do
    assert(!bookUndefined || e.second.th1 ||
           !"Missing histogram. Something is broken.");
    if (!e.second.th1) continue;

    iBooker.setCurrentFolder(makePath(e.first));

    if (e.second.th1->GetDimension() == 1) {
      TAxis* ax = e.second.th1->GetXaxis();
      e.second.me =
          iBooker.book1D(e.second.th1->GetName(), e.second.th1->GetTitle(),
                         ax->GetNbins(), ax->GetXmin(), ax->GetXmax());
      e.second.me->setAxisTitle(ax->GetTitle());
      e.second.me->setAxisTitle(e.second.th1->GetYaxis()->GetTitle(), 2);
    } else {
      TAxis* axX = e.second.th1->GetXaxis();
      TAxis* axY = e.second.th1->GetYaxis();
      e.second.me =
          iBooker.book2D(e.second.th1->GetName(), e.second.th1->GetTitle(),
                         axX->GetNbins(), axX->GetXmin(), axX->GetXmax(),
                         axY->GetNbins(), axY->GetXmin(), axY->GetXmax());
      e.second.me->setAxisTitle(axX->GetTitle());
      e.second.me->setAxisTitle(axY->GetTitle(), 2);
    }

    e.second.me->getTH1()->Add(e.second.th1);
    e.second.th1 = e.second.me->getTH1();
    e.second.th1->SetStats(true);
//    delete e.second.th1;
  }
}

void HistogramManager::executeGroupBy(SummationStep& step, Table& t) {
  // Simple grouping. Drop colums, add histos if one is already present.
  Table out;
  for (auto& e : t) {
    GeometryInterface::Values const& old_vals(e.first);
    TH1* th1 = e.second.th1;
    GeometryInterface::Values new_vals;
    for (auto c : step.columns) new_vals.put(old_vals.get(c));
    AbstractHistogram& new_histo = out[new_vals];
    if (!new_histo.th1) {
      new_histo.th1 = (TH1*)th1->Clone();
    } else {
      new_histo.th1->Add(th1);
    }
  }
  t.swap(out);
}

void HistogramManager::executeReduce(SummationStep& step, Table& t) {
  Table out;
  for (auto& e : t) {
    GeometryInterface::Values const& vals(e.first);
    TH1* th1 = e.second.th1;
    AbstractHistogram& new_histo = out[vals];
    double reduced_quantity = 0;
    double reduced_quantity_error = 0;
    std::string label = "";
    std::string name = th1->GetName();
    // TODO: meaningful semantics in 2D case, errors
    if (step.arg == "MEAN") {
      reduced_quantity = th1->GetMean();
      reduced_quantity_error = th1->GetMeanError();
      label = label + "mean of " + th1->GetXaxis()->GetTitle();
      name = "mean_" + name;
    } else if (step.arg == "COUNT") {
      reduced_quantity = th1->GetEntries();
      label = label + "# of " + th1->GetXaxis()->GetTitle() + " entries";
      name = "num_" + name;
    } else /* if (step.arg) == ... TODO: more */ {
      edm::LogError("HistogramManager") << "+++ Reduction '" << step.arg
                                        << " not yet implemented\n";
    }
    new_histo.th1 = new TH1F(
        name.c_str(),
        (std::string("") + th1->GetTitle() + ";;" + label).c_str(), 1, 0, 1);
    new_histo.th1->SetBinContent(1, reduced_quantity);
    new_histo.th1->SetBinError(1, reduced_quantity_error);
  }
  t.swap(out);
}

void HistogramManager::executeExtend(SummationStep& step, Table& t, bool isX) {
  // For the moment only X.
  // first pass determines the range.
  std::map<GeometryInterface::Values, int> nbins;
  // separators collects meta info for the render plugin about the boundaries.
  // for each EXTEND, this is added to the axis label. In total this is not
  // fully correct since we have to assume the the substructure of each sub-
  // histogram is the same, which is e.g. not true for layers. It still works
  // since layers only contain leaves (ladders).
  std::map<GeometryInterface::Values, std::string> separators;
  for (auto& e : t) {
    GeometryInterface::Values new_vals(e.first);
    new_vals.erase(step.columns.at(0));
    TH1* th1 = e.second.th1;
    int& n = nbins[new_vals];
    int bins = 0;
    assert(th1 || !"invalid histogram");
    if (isX)
      bins = th1->GetXaxis()->GetNbins();
    else
      bins = th1->GetYaxis()->GetNbins();
    if (bins > 1) separators[new_vals] += std::to_string(n) + ",";
    n += bins;
  }
  for (auto& e : separators) e.second = "(" + e.second + ")/";

  Table out;
  for (auto& e : t) {
    GeometryInterface::Values const& old_vals(e.first);
    GeometryInterface::Column col0 = old_vals.get(step.columns.at(0)).first;
    GeometryInterface::Values new_vals(old_vals);
    new_vals.erase(step.columns.at(0));
    std::string colname = geometryInterface.pretty(col0);
    TH1* th1 = e.second.th1;
    assert(th1);
    auto separator = separators[new_vals];
    if (colname == "") separator = "";  // for dummy column

    AbstractHistogram& new_histo = out[new_vals];
    GeometryInterface::Values copy(new_vals);
    if (!new_histo.th1) {
      //std::cout << "+++ new TH1D for extend ";
      // We need to book. Two cases here: 1D or 2D.

      const char* title;
      if (isX)
        title = (std::string("") + th1->GetTitle() + " per " + colname + ";" +
                 colname + separator + th1->GetXaxis()->GetTitle() + ";" +
                 th1->GetYaxis()->GetTitle())
                    .c_str();
      else
        title = (std::string("") + th1->GetTitle() + " per " + colname + ";" +
                 th1->GetXaxis()->GetTitle() + ";" + colname + separator +
                 th1->GetYaxis()->GetTitle())
                    .c_str();

      if (th1->GetDimension() == 1 && isX) {
        // Output is 1D. Never the case for EXTEND_Y
        new_histo.th1 = (TH1*)new TH1F(th1->GetName(), title, nbins[new_vals], 0.5, nbins[new_vals] + 0.5);
//        new_histo.th1 = (TH1*)new TProfile(th1->GetName(), title, nbins[new_vals], 0.5, nbins[new_vals] + 0.5);
      } else {
        // output is 2D, input is 2D histograms.
        if (isX)
          new_histo.th1 =
              (TH1*)new TH2F(th1->GetName(), title, nbins[new_vals], 0.5,
                             nbins[new_vals] + 0.5, th1->GetYaxis()->GetNbins(),
                             0.5, th1->GetYaxis()->GetNbins() + 0.5);
        else
          new_histo.th1 =
              (TH1*)new TH2F(th1->GetName(), title, th1->GetXaxis()->GetNbins(),
                             0.5, th1->GetXaxis()->GetNbins() + 0.5,
                             nbins[new_vals], 0.5, nbins[new_vals] + 0.5);
      }
      // std::cout << "title " << new_histo.th1->GetTitle()<< "\n";
      new_histo.count = 1;  // used as a fill pointer. Assumes histograms are
                            // ordered correctly (map should provide that)
    }

    // now add data.
    if (new_histo.th1->GetDimension() == 1) {	
        for (int i = 1; i <= th1->GetXaxis()->GetNbins(); i++) {
        new_histo.th1->SetBinContent(new_histo.count, th1->GetBinContent(i));
        new_histo.th1->SetBinError(new_histo.count, th1->GetBinError(i));
        new_histo.count += 1;
      }
    } else {
      // 2D case.
      if (isX) {
        for (int i = 1; i <= th1->GetXaxis()->GetNbins(); i++) {
          for (int j = 1; j <= th1->GetYaxis()->GetNbins(); j++) {
            // TODO Error etc.?
            new_histo.th1->SetBinContent(new_histo.count, j,
                                         th1->GetBinContent(i, j));
            new_histo.th1->SetBinError   (new_histo.count, j, th1->GetBinError(i, j));
          }
          new_histo.count += 1;
        }
      } else {
        for (int j = 1; j <= th1->GetYaxis()->GetNbins(); j++) {
          for (int i = 1; i <= th1->GetXaxis()->GetNbins(); i++) {
            // TODO Error etc.?
            new_histo.th1->SetBinContent(i, new_histo.count,
                                         th1->GetBinContent(i, j));
            new_histo.th1->SetBinError   (i, new_histo.count, th1->GetBinError(i, j));
          }
          new_histo.count += 1;
        }
      }
    }
  }
  t.swap(out);
}

#endif

void HistogramManager::executeHarvesting(DQMStore::IBooker& iBooker,
                                         DQMStore::IGetter& iGetter) {
  if (!enabled) return;
  // edm::LogTrace("HistogramManager") << "HistogramManager: Step2 offline\n";
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

    // now execute step2.
    for (SummationStep step : s.steps) {
      if (step.stage == SummationStep::STAGE2) {
        switch (step.type) {
          case SummationStep::SAVE:
            executeSave(step, t, iBooker);
            break;
          case SummationStep::GROUPBY:
            executeGroupBy(step, t);
            break;
          case SummationStep::REDUCE:
            executeReduce(step, t);
            break;
          case SummationStep::EXTEND_X:
            executeExtend(step, t, true);
            break;
          case SummationStep::EXTEND_Y:
            executeExtend(step, t, false);
            break;
          case SummationStep::CUSTOM:
            if (customHandler) customHandler(step, t);
            break;
          default:
            assert(!"Operation not supported in harvesting.");
        }  // switch
      }    // if step2
    }      // for each step
  }        // for each spec
}
