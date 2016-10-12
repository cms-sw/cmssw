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
      range_nbins(iconfig.getParameter<int>("range_nbins")),
      range_min(iconfig.getParameter<double>("range_min")),
      range_max(iconfig.getParameter<double>("range_max")),
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

// note that this will be pretty hot. Ideally it should be malloc-free.
// if fastpath is non-null, it is use instead of looking up the map entry,
// else it will be set to the map entry for the next lookup.
void HistogramManager::executeStep1Spec(
    double x, double y, GeometryInterface::Values& significantvalues,
    SummationSpecification& s, Table& t, SummationStep::Stage stage,
    AbstractHistogram*& fastpath) {
  int dimensions = this->dimensions;
  double z = 0; // quantity for profiles
  for (unsigned int i = 0; i < s.steps.size(); i++) {
    auto& step = s.steps[i];
    if (step.stage == stage) {
      switch (step.type) {
        case SummationStep::SAVE:
          break;  // this happens automatically
        case SummationStep::COUNT: {
          // Two cases here: COUNT/EXTEND where this is a noop,
          // COUNT/GROUPBY per event where this triggers a counter.
          x = 0.0;
          y = 0.0;
          dimensions = 0;
          if (s.steps.at(i + 1).stage == SummationStep::STAGE1_2) {
            if (!fastpath)
              fastpath = &t[significantvalues];  // PERF: [] might be bad.
            fastpath->count++;
            return;  // no more filling to be done
          }
          break;
        }
        case SummationStep::EXTEND_X:
          assert((x == 0.0 && dimensions != 1) ||
                 (y == 0.0 && dimensions == 1) ||
                 !"Illegal EXTEND in step1");
          if (dimensions == 1) y = x;
          x = significantvalues[step.columns.at(0)];

          significantvalues.erase(step.columns.at(0));
          dimensions = dimensions < 2 ? dimensions+1 : dimensions;
          break;
        case SummationStep::EXTEND_Y:
          y = significantvalues[step.columns.at(0)];
          significantvalues.erase(step.columns.at(0));
          // EXTEND_Y on 2D data implies Profile (or bug)
          dimensions = dimensions == 2 ? 3 : 2; // 3 for 2D profile
          break;
        case SummationStep::GROUPBY: {
          assert(
              stage == SummationStep::STAGE1_2 ||
              !"Only COUNT/GROUPBY with per-event harvesting allowed in step1");
          dimensions = 1;
          if (!fastpath)
            x = (double)t[significantvalues]
                    .count;  // PERF: redundant lookup, caller has it.
          else
            x = (double)fastpath->count;  // we expect to get that from the per
                                          // event harvesting.
          fastpath =
              nullptr;  // we change significantvalues, so this becomes invalid.
          t[significantvalues].count = 0;
          new_vals.clear();
          for (auto c : step.columns) new_vals.put(significantvalues.get(c));
          significantvalues.values.swap(new_vals.values);
          break;
        }
        case SummationStep::REDUCE:
          // assert(step.arg == "MEAN")
          z = y = x;
          x = 0.0;
          dimensions = 2; // Profile always needs 2D (or 3D) fill
          break;
        case SummationStep::CUSTOM:
        case SummationStep::NO_TYPE:
          assert(!"Illegal step; booking should have caught this.");
      }
    }
  }
  if (!fastpath) {
    auto histo = t.find(significantvalues);  // avoid modification
    if (histo == t.end() || (!histo->second.th1 && !histo->second.me)) {
      // No histogram was booked.
      assert(!bookUndefined || !"All histograms were booked but one is missing. This is a problem in the booking process.");
      // else, ignore the sample.
      return;
    }
    fastpath = &(histo->second);
  }
  if (dimensions == 0)
    fastpath->fill(0);
  else if (dimensions == 1)
    fastpath->fill(x);
  else if (dimensions == 2)
    fastpath->fill(x, y);
  else /* dimensions == 3 */
    fastpath->fill(x, y, z);

}

void HistogramManager::fill(double x, double y, DetId sourceModule,
                            const edm::Event* sourceEvent, int col, int row) {
  if (!enabled) return;
  bool cached = true;
  // We could be smarter on row/col and only check if they appear in the spec
  // but that just asks for bugs.
  if (col != this->iq.col || row != this->iq.row ||
      sourceModule != this->iq.sourceModule ||
      sourceEvent != this->iq.sourceEvent ||
      sourceModule == DetId(0)  // Hack for eventrate-like things, since the
                                // sourceEvent ptr might not change.
      ) {
    cached = false;
    iq = GeometryInterface::InterestingQuantities{sourceModule, sourceEvent,
                                                  col, row};
  }
  for (unsigned int i = 0; i < specs.size(); i++) {
    auto& s = specs[i];
    auto& t = tables[i];
    // Try cached colums from last fill().
    if (!cached) {
      significantvalues[i].clear();
      geometryInterface.extractColumns(s.steps[0].columns, iq,
                                       significantvalues[i]);
      fastpath[i] = nullptr;
    }
    // copy the signifcant values here, since they might be modified.
    significantvalues_scratch = significantvalues[i];
    executeStep1Spec(x, y, significantvalues_scratch, s, t, SummationStep::STAGE1,
                     fastpath[i]);
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

// This is only used for ndigis-like counting. It could be more optimized, but
// is probably fine for a per-event thing.
void HistogramManager::executePerEventHarvesting() {
  // We have a masive problem here regarding semantics: for geometrical structure,
  // we always want to see all the counters, even if they are 0. The counters are
  // all created during booking ad then ket, all fine.
  // For time-based things however, we do not know the range before we see the
  // data, so we cannot book the counters beforhand. Also, it does not make sense
  // to record 0 counts: a event can only be in one Lumisection, it does not make
  // sense to record it as a 0 count for all others.
  // The hacky solution is to check that the quantity was UNDEFINED in booking,
  // which indicates sth. event-dependent, and in this case ignore the UNDEFINED
  // counters and delete the defined ones, that they don't pollute later events.
  if (!enabled) return;
  for (unsigned int i = 0; i < specs.size(); i++) {
    auto& s = specs[i];
    auto& t = tables[i];
    bool range_undefined = false;
    for (auto e : t) {
      // There are two types of entries: counters and histograms. We only want
      // ctrs here.
      // TODO: not fully correct, if we have e.g. EXTEND/COUNT/GROUPBY.
      if (e.first.values.size() == s.steps[0].columns.size()) {
        // this is actually useless, since we will use the fast path. But better
        // for consistence.
        significantvalues[i].values = e.first.values;

        bool defined = true;
        for (auto v : e.first.values)
          defined &= (v.second != GeometryInterface::UNDEFINED);
        if (!defined) {
          range_undefined = true;
          continue;
        }

        auto fastpath = &e.second;
        // copy the signifcant values here, since they might be modified.
        // (unlikely, again for consistence)
        significantvalues_scratch = significantvalues[i];
        executeStep1Spec(0.0, 0.0, significantvalues_scratch, s, t,
                         SummationStep::STAGE1_2, fastpath);
      }
    }
    if (range_undefined) {
      for (auto it = t.begin(); it != t.end(); ++it) {
        if (it->first.values.size() == s.steps[0].columns.size()) {
          bool defined = true;
          for (auto v : it->first.values)
            defined &= (v.second != GeometryInterface::UNDEFINED);
          if (defined) {
            it = t.erase(it);
          }
        }
      }
    }
  }
}

std::string HistogramManager::makePath(
    GeometryInterface::Values const& significantvalues) {
  // non-number output names (_pO etc.) are hardwired here.
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

void HistogramManager::book(DQMStore::IBooker& iBooker,
                            edm::EventSetup const& iSetup) {
  if (!geometryInterface.loaded()) {
    geometryInterface.load(iSetup);
  }
  if (!enabled) return;

  for (unsigned int i = 0; i < specs.size(); i++) {
    auto& s = specs[i];
    auto& t = tables[i];
    for (auto iq : geometryInterface.allModules()) {
      GeometryInterface::Values significantvalues;
      geometryInterface.extractColumns(s.steps[0].columns, iq,
                                       significantvalues);
      bool do_profile = false;
      auto dimensions = this->dimensions;
      std::string name = this->name;
      std::string title = this->title;
      std::string xlabel = this->xlabel;
      std::string ylabel = this->ylabel;
      int range_x_nbins = this->range_nbins;
      double range_x_min = this->range_min, range_x_max = this->range_max;
      int range_y_nbins = this->range_y_nbins;
      double range_y_min = this->range_y_min, range_y_max = this->range_y_max;
      GeometryInterface::Value observed_x = GeometryInterface::UNDEFINED;
      GeometryInterface::Value observed_y = GeometryInterface::UNDEFINED;
      for (SummationStep step : s.steps) {
        if (step.stage == SummationStep::STAGE1 ||
            step.stage == SummationStep::STAGE1_2) {
          switch (step.type) {
            case SummationStep::SAVE:
              break;  // this happens automatically
            case SummationStep::COUNT: {
              dimensions = 0;
              title = "Count of " + title;
              name = "num_" + name;
              ylabel = "#" + xlabel;
              xlabel = "";
              range_x_nbins = range_y_nbins = 1;
              range_x_min = range_y_min = 0;
              range_x_max = range_y_max = 1;
              AbstractHistogram& ctr = t[significantvalues];
              ctr.kind = MonitorElement::DQM_KIND_INT;
              ctr.count = 0;
              break;
            }
            case SummationStep::EXTEND_X: {
              GeometryInterface::Column col0 =
                  significantvalues.get(step.columns.at(0)).first;
              std::string colname = geometryInterface.pretty(col0);
              title = title + " per " + colname;
              name = name + "_per_" + colname;
              if (dimensions == 1) {
                ylabel = xlabel;
                range_y_min = range_x_min;
                range_y_max = range_x_max;
                range_y_nbins = range_x_nbins;
                observed_y = observed_x;
              }
              dimensions = dimensions == 0 ? 1 : 2;
              xlabel = colname;
              observed_x = significantvalues.get(col0).second;
              if (geometryInterface.minValue(col0[0]) != GeometryInterface::UNDEFINED) {
                range_x_min = geometryInterface.minValue(col0[0]) - 0.5;
                range_x_max = geometryInterface.maxValue(col0[0]) + 0.5;
                range_x_nbins = int(range_x_max - range_x_min);
              } else {
                range_x_min = range_x_max = GeometryInterface::UNDEFINED;
                range_x_nbins = 0;
              }
              significantvalues.erase(col0);
              break;
            }
            case SummationStep::EXTEND_Y: {
              GeometryInterface::Column col0 =
                  significantvalues.get(step.columns.at(0)).first;
              std::string colname = geometryInterface.pretty(col0);
              assert(dimensions != 2 || !"2D to 2D reduce NYI in step1");
              dimensions = 2;
              title = title + " per " + colname;
              name = name + "_per_" + colname;
              if (do_profile) { // we loose the Z- (former Y-) label here
                title = title + " (Z: " + ylabel + ")";
              }
              observed_y = significantvalues.get(col0).second;
              ylabel = colname;
              if (geometryInterface.minValue(col0[0]) != GeometryInterface::UNDEFINED) {
                range_y_min = geometryInterface.minValue(col0[0]) - 0.5;
                range_y_max = geometryInterface.maxValue(col0[0]) + 0.5;
                range_y_nbins = int(range_y_max - range_y_min);
              } else {
                range_y_min = range_y_max = GeometryInterface::UNDEFINED;
                range_y_nbins = 0;
              }
              significantvalues.erase(col0);
              break;
            }
            case SummationStep::GROUPBY: {
              assert(dimensions == 0 || !"Only COUNT/GROUPBY with per-event harvesting allowed in step1");
              dimensions = 1;
              GeometryInterface::Values new_vals;
              for (auto c : step.columns)
                new_vals.put(significantvalues.get(c));
              range_x_nbins = this->range_nbins;
              range_x_min = this->range_min;
              range_x_max = this->range_max;
              xlabel = ylabel + " per Event";
              if (s.steps[0].columns.size() > 0)
                xlabel =
                    xlabel + " and " +
                    geometryInterface.pretty(
                        significantvalues
                            .get(s.steps[0]
                                     .columns[s.steps[0].columns.size() - 1])
                            .first);
              ylabel = "#Entries";
              significantvalues = new_vals;
              break;
            }
            case SummationStep::REDUCE: {
              assert(step.arg == "MEAN" || !"Only profiles in step 1");
              assert(dimensions == 1 || !"Profile needs a 1D mean");
              assert(do_profile == false || !"Already doing a profile!");
              title = "Profile of " + title;
              do_profile = true;
              ylabel = xlabel;
              range_y_nbins = range_x_nbins;
              range_y_min = range_x_min;
              range_y_max = range_x_max;
              range_x_nbins = 1;
              range_x_min = 0;
              range_x_max = 1;
              dimensions = 0;
              break;
            }
            case SummationStep::CUSTOM:
            case SummationStep::NO_TYPE:
              assert(!"Operation not supported in step1. Try save() before to switch to Harvesting.");
          }
        }
      }

      if (!bookUndefined) {
        // skip if any column is UNDEFINED
        bool ok = true;
        for (auto e : significantvalues.values)
          if (e.second == GeometryInterface::UNDEFINED) ok = false;
        if (!ok) continue;
      }

      AbstractHistogram& histo = t[significantvalues];

      // track min and max values for all modules to get Geometry-dependent
      // things (e.g. #Ladders per Layer) right
      if (observed_x != GeometryInterface::UNDEFINED) {
        if (observed_x > histo.range_x_max) histo.range_x_max = observed_x;
        if (observed_x < histo.range_x_min) histo.range_x_min = observed_x;
      }
      if (observed_y != GeometryInterface::UNDEFINED) {
        if (observed_y > histo.range_y_max) histo.range_y_max = observed_y;
        if (observed_y < histo.range_y_min) histo.range_y_min = observed_y;
      }

      if (int(range_x_min) != GeometryInterface::UNDEFINED)
        histo.range_x_min = range_x_min;
      if (int(range_x_max) != GeometryInterface::UNDEFINED)
        histo.range_x_max = range_x_max;
      if (int(range_y_min) != GeometryInterface::UNDEFINED)
        histo.range_y_min = range_y_min;
      if (int(range_y_max) != GeometryInterface::UNDEFINED)
        histo.range_y_max = range_y_max;

      histo.range_x_nbins = range_x_nbins;
      histo.range_y_nbins = range_y_nbins;

      histo.name = name;
      histo.title = title;
      histo.xlabel = xlabel;
      histo.ylabel = ylabel;

      if (dimensions == 0 || dimensions == 1) {
        if (!do_profile) {
          histo.kind = MonitorElement::DQM_KIND_TH1F;
        } else {
          histo.kind = MonitorElement::DQM_KIND_TPROFILE;
        }
      } else if (dimensions == 2) {
        if (!do_profile) {
          histo.kind = MonitorElement::DQM_KIND_TH2F;
        } else {
          histo.kind = MonitorElement::DQM_KIND_TPROFILE2D;
        }
      }
      assert(histo.kind != MonitorElement::DQM_KIND_INVALID);
    }
    // above, we only saved all the parameters, but did not book yet. This is
    // needed since we determine the ranges iteratively, but we need to know
    // them precisely for booking; they cannot be changed later.
    for (auto& e : t) {
      iBooker.setCurrentFolder(makePath(e.first));
      AbstractHistogram& h = e.second;

      if (h.range_x_nbins == 0) {
        h.range_x_min -= 0.5;
        h.range_x_max += 0.5;
        h.range_x_nbins = int(h.range_x_max - h.range_x_min);
      }
      if (h.range_y_nbins == 0) {
        h.range_y_min -= 0.5;
        h.range_y_max += 0.5;
        h.range_y_nbins = int(h.range_y_max - h.range_y_min);
      }

      if (h.kind == MonitorElement::DQM_KIND_TH1F) {
        h.me = iBooker.book1D(h.name, (h.title + ";" + h.xlabel).c_str(),
                       h.range_x_nbins, h.range_x_min, h.range_x_max);
      } else if (h.kind == MonitorElement::DQM_KIND_TH2F) {
        h.me = iBooker.book2D(h.name, (h.title + ";" + h.xlabel + ";" + h.ylabel).c_str(),
                       h.range_x_nbins, h.range_x_min, h.range_x_max,
                       h.range_y_nbins, h.range_y_min, h.range_y_max);
      } else if (h.kind == MonitorElement::DQM_KIND_TPROFILE) {
        h.me = iBooker.bookProfile(h.name, (h.title + ";" + h.xlabel + ";" + h.ylabel).c_str(),
                       h.range_x_nbins, h.range_x_min, h.range_x_max, -(1./0.), 1./0.);
      } else if (h.kind == MonitorElement::DQM_KIND_TPROFILE2D) {
        h.me = iBooker.bookProfile2D(h.name, (h.title + ";" + h.xlabel + ";" + h.ylabel).c_str(),
                       h.range_x_nbins, h.range_x_min, h.range_x_max,
                       h.range_y_nbins, h.range_y_min, h.range_y_max,
                       0.0, 0.0); // Z range is ignored if min==max
      } else if (h.kind == MonitorElement::DQM_KIND_INT) {
        // counter, nothing to do
        continue; // avoid the getTH1 below
      } else {
        assert(!"Illegal Histogram kind.");
      }

      h.th1 = h.me->getTH1();
    }
  }
}

void HistogramManager::executePerLumiHarvesting(DQMStore::IBooker& iBooker,
                                                DQMStore::IGetter& iGetter,
                                                edm::EventSetup const& iSetup) {
  if (!enabled) return;
  // this should also give us the GeometryInterface for offline, though it is a
  // bit dirty and might explode.
  if (!geometryInterface.loaded()) {
    geometryInterface.load(iSetup);
  }
  if (perLumiHarvesting) {
    executeHarvesting(iBooker, iGetter);
  }
}

void HistogramManager::loadFromDQMStore(SummationSpecification& s, Table& t,
                                        DQMStore::IGetter& iGetter) {
  // This is essentially the booking code of step1, to reconstruct the ME names.
  // Once we have a name we load the ME and put it into the table.
  t.clear();
  for (auto iq : geometryInterface.allModules()) {
    std::string name = this->name;
    GeometryInterface::Values significantvalues;
    geometryInterface.extractColumns(s.steps[0].columns, iq, significantvalues);
    // PERF: if (!bookUndefined) we could skip a lot here.
    for (SummationStep step : s.steps) {
      if (step.stage == SummationStep::STAGE1 ||
          step.stage == SummationStep::STAGE1_2) {
        switch (step.type) {
          case SummationStep::SAVE:
            break;  // this happens automatically
          case SummationStep::COUNT:
            name = "num_" + name;
            break;
          case SummationStep::EXTEND_X:
          case SummationStep::EXTEND_Y: {
            GeometryInterface::Column col0 =
                significantvalues.get(step.columns.at(0)).first;
            std::string colname = geometryInterface.pretty(col0);
            name = name + "_per_" + colname;
            significantvalues.erase(col0);
            break;
          }
          case SummationStep::GROUPBY: {
            GeometryInterface::Values new_vals;
            for (auto c : step.columns) new_vals.put(significantvalues.get(c));
            significantvalues = new_vals;
            break;
          }
          case SummationStep::REDUCE:
            break; // not visible in name
          case SummationStep::CUSTOM:
          case SummationStep::NO_TYPE:
            assert(!"Illegal step; booking should have caught this.");
        }
      }
    }
    // note that we call get() here for every single module. But the string
    // ops above are probably more expensive anyways...
    std::string path = makePath(significantvalues) + name;
    MonitorElement* me = iGetter.get(path);
    if (!me) {
      if (bookUndefined)
        edm::LogError("HistogramManager") << "ME " << path << " not found\n";
      // else this will happen quite often
    } else {
      // only touch the table if a me is added. Empty items are illegal.
      AbstractHistogram& histo = t[significantvalues];
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
          case SummationStep::COUNT:
          case SummationStep::NO_TYPE:
            assert(!"Operation not supported in harvesting.");
        }  // switch
      }    // if step2
    }      // for each step
  }        // for each spec
}
