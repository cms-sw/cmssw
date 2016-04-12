// -*- C++ -*-
//
// Package:     SiPixelPhase1Common
// Class  :     HistogramManager
//
#include "DQM/SiPixelPhase1Common/interface/HistogramManager.h"

#include <sstream>

// Geometry stuff
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// Logger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HistogramManager::HistogramManager(const edm::ParameterSet& iconfig, GeometryInterface& geo) :
  iConfig(iconfig),
  geometryInterface(geo),
  enabled(iconfig.getParameter<bool>("enabled")),
  bookUndefined(iconfig.getParameter<bool>("bookUndefined")),
  top_folder_name(iconfig.getParameter<std::string>("topFolderName")),
  name(iconfig.getParameter<std::string>("name")),
  title(iconfig.getParameter<std::string>("title")),
  xlabel(iconfig.getParameter<std::string>("xlabel")),
  ylabel(iconfig.getParameter<std::string>("ylabel")),
  dimensions(iconfig.getParameter<int>("dimensions")),
  range_nbins(iconfig.getParameter<int>("range_nbins")),
  range_min(iconfig.getParameter<double>("range_min")),
  range_max(iconfig.getParameter<double>("range_max"))
{ 
  auto spec_configs = iconfig.getParameter<edm::VParameterSet>("specs");
  for (auto spec : spec_configs) {
    addSpec(SummationSpecification(spec, geometryInterface));
  }
}

void HistogramManager::addSpec(SummationSpecification spec) {
  specs.push_back(spec);
  tables.push_back(Table());
}

// note that this will be pretty hot. Ideally it should be malloc-free.
void HistogramManager::fill(double x, double y, DetId sourceModule, const edm::Event *sourceEvent, int col, int row) {
  if (!enabled) return;
  auto iq = GeometryInterface::InterestingQuantities{
              sourceModule, sourceEvent, col, row
	    };							    
  for (unsigned int i = 0; i < specs.size(); i++) {
    auto& s = specs[i];
    auto& t = tables[i];
    // TODO: we could recycle the last Values if iq has not changed (This is common)
    // row/col is a bit harder then (only pass if needed, for perf reasons)
    // Caching has to happen per-spec, of course.
    GeometryInterface::Values significantvalues; // TODO move this to a member 
    geometryInterface.extractColumns(s.steps[0].columns, iq, significantvalues);
    int dimensions = this->dimensions;
    for (SummationStep step : s.steps) {
      if (step.stage == SummationStep::STAGE1) {
	switch(step.type) {
	case SummationStep::SAVE: break; // this happens automatically
	case SummationStep::COUNT:
	  // this does not really do anything. ATM only COUNT/EXTEND is supported, and
	  // this is handled below.
	  x = 0.0; y = 0.0;
	  dimensions = 0;
	  break;
	case SummationStep::EXTEND_X:
	  assert((x == 0.0 && dimensions != 1) || !"Can only EXTEND on COUNTs in step1");
	  x = significantvalues[step.columns.at(0)];
	  significantvalues.erase(step.columns.at(0));
	  dimensions = dimensions == 0 ? 1 : 2;
	  break;
	case SummationStep::EXTEND_Y:
	  assert(y == 0.0 || !"Can only EXTEND on COUNTs in step1");
	  y = significantvalues[step.columns.at(0)];
	  significantvalues.erase(step.columns.at(0));
	  dimensions = 2;
	  break;
	case SummationStep::GROUPBY: 
	case SummationStep::REDUCE:
	case SummationStep::CUSTOM:
	case SummationStep::NO_TYPE:
	  assert(!"Illegal step; booking should have caught this.");
	}
      }
    }
    auto& histo = t[significantvalues];
    if (!histo.th1 && !histo.me) {
      // No histogram was booked.
      assert(!bookUndefined || !"All histograms were booked but one is missing. This is a problem in the booking process.");
      // else, ignore the sample.
      continue;
    }
    if (dimensions == 0) {
      histo.fill(0);
    } else if (dimensions == 1)
      histo.fill(x);
    else /* dimensions == 2 */ {
      histo.fill(x, y);
    }
  }
}
void HistogramManager::fill(double x, DetId sourceModule, const edm::Event *sourceEvent, int col, int row) {
  assert(this->dimensions == 1);
  fill(x, 0.0, sourceModule, sourceEvent, col, row);
}
void HistogramManager::fill(DetId sourceModule, const edm::Event *sourceEvent, int col, int row) {
  assert(this->dimensions == 0);
  fill(0.0, 0.0, sourceModule, sourceEvent, col, row);
}
  
void HistogramManager::book(DQMStore::IBooker& iBooker, edm::EventSetup const& iSetup) {
  if (!enabled) return;
  if (!geometryInterface.loaded()) {
    geometryInterface.load(iSetup);
  }

  for (unsigned int i = 0; i < specs.size(); i++) {
    auto& s = specs[i];
    auto& t = tables[i];
    for (auto iq : geometryInterface.allModules()) {
      GeometryInterface::Values significantvalues;
      geometryInterface.extractColumns(s.steps[0].columns, iq, significantvalues);
      if (!bookUndefined) {
	// skip if any column is UNDEFINED
	bool ok = true;
	for (auto e : significantvalues.values) 
	  if (e.second == GeometryInterface::UNDEFINED) ok = false;
	if (!ok) continue;
      }
      auto dimensions = this->dimensions;
      std::string name = this->name;
      std::string title = this->title;
      std::string xlabel = this->xlabel;
      std::string ylabel = this->ylabel;
      int range_x_nbins = this->range_nbins;
      double range_x_min = this->range_min, range_x_max = this->range_max;
	// TODO: proper 2D range.
      int range_y_nbins = this->range_nbins;
      double range_y_min = this->range_min, range_y_max = this->range_max;
      for (SummationStep step : s.steps) {
	if (step.stage == SummationStep::STAGE1) {
	  GeometryInterface::Values new_vals;
	  switch(step.type) {
	  case SummationStep::SAVE: break; // this happens automatically
	  case SummationStep::COUNT:
	    dimensions = 0;
	    title = "Count of " + title;
	    name = "num_" + name;
	    ylabel = "# of " + xlabel;
	    xlabel = "";
	    range_x_nbins = range_y_nbins = 1;
	    range_x_min = range_y_min = 0;
	    range_x_max = range_y_max = 1;
	    break;
	  case SummationStep::EXTEND_X: {
	    GeometryInterface::Column col0 = significantvalues.get(step.columns.at(0)).first;
	    std::string colname = geometryInterface.pretty(col0);
	    assert(dimensions != 1 || !"1D to 1D reduce NYI in step1");
	    dimensions = dimensions == 0 ? 1 : 2;
	    title = title + " per " + colname;
	    name = name + "_per_" + colname;
	    xlabel = colname;
	    range_x_min = 0;
	    range_x_nbins = geometryInterface.maxValue(col0[0]);
	    range_x_max = range_x_nbins;
	    significantvalues.erase(col0);
	    break;}
	  case SummationStep::EXTEND_Y: {
	    GeometryInterface::Column col0 = significantvalues.get(step.columns.at(0)).first;
	    std::string colname = geometryInterface.pretty(col0);
	    assert(dimensions != 2 || !"2D to 2D reduce NYI in step1");
	    dimensions = 2;
	    title = title + " per " + colname;
	    name = name + "_per_" + colname;
	    ylabel = colname;
	    range_y_min = 0;
	    range_y_nbins = geometryInterface.maxValue(col0[0]);
	    range_y_max = range_y_nbins;
	    significantvalues.erase(col0);
	    break;}
	  case SummationStep::GROUPBY:
	  case SummationStep::REDUCE:
	  case SummationStep::CUSTOM:
	  case SummationStep::NO_TYPE:
	    assert(!"Operation not supported in step1. Try save() before to switch to Harvesting.");
	  }
	}
      }

      AbstractHistogram& histo = t[significantvalues];
      if (histo.me) continue;

      std::ostringstream dir("");
      for (auto c : s.steps[0].columns) {
	  auto entry = significantvalues.get(c);
	  // col[0] = 0 implies col[1] = 0 which means invalid colum. This one was not there.
	  if (entry.first[0] != 0) dir << geometryInterface.pretty(entry.first) << "_" << entry.second << "/";
      }

      iBooker.setCurrentFolder(top_folder_name + "/" + dir.str());

      if (dimensions == 0 || dimensions == 1) {
      	histo.me = iBooker.book1D(name.c_str(), (title + ";" + xlabel).c_str(), range_x_nbins, range_x_min, range_x_max);
      } else if (dimensions == 2) {
	histo.me = iBooker.book2D(name.c_str(), (title + ";" + xlabel + ";" + ylabel).c_str(), 
	                          range_x_nbins, range_x_min, range_x_max, range_y_nbins, range_y_min, range_y_max);
      }
    }
  }
}

void HistogramManager::executeHarvestingOnline(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter, edm::EventSetup const& iSetup) {
  if (!enabled) return;
  // this should also give us the GeometryInterface for offline, though it is a bit dirty and might explode.
  if (!geometryInterface.loaded()) {
    geometryInterface.load(iSetup);
  }
}

void HistogramManager::loadFromDQMStore(SummationSpecification& s, Table& t, DQMStore::IGetter& iGetter) {
  // This is essentially the booking code of step1, to reconstruct the ME names.
  // Once we have a name we load the ME and put it into the table.
  for (auto iq : geometryInterface.allModules()) {
    std::string name = this->name;
    GeometryInterface::Values significantvalues;
    geometryInterface.extractColumns(s.steps[0].columns, iq, significantvalues);
    // TODO: if (!bookUndefined) we could skip a lot here.
    for (SummationStep step : s.steps) {
      if (step.stage == SummationStep::STAGE1) {
	GeometryInterface::Values new_vals;
	switch(step.type) {
	case SummationStep::SAVE: break; // this happens automatically
	case SummationStep::COUNT:
	  name = "num_" + name;
	  break;
	case SummationStep::EXTEND_X:
	case SummationStep::EXTEND_Y:{
	  GeometryInterface::Column col0 = significantvalues.get(step.columns.at(0)).first;
	  std::string colname = geometryInterface.pretty(col0);
	  name = name + "_per_" + colname;
	  significantvalues.erase(col0);
	  break;}
	case SummationStep::GROUPBY:
	case SummationStep::REDUCE:
	case SummationStep::CUSTOM:
	case SummationStep::NO_TYPE:
	  assert(!"Illegal step; booking should have caught this.");
	}
      }
    }

    std::ostringstream dir(top_folder_name + "/", std::ostringstream::ate);
    for (auto c : s.steps[0].columns) {
	auto entry = significantvalues.get(c);
	// col[0] = 0 implies col[1] = 0 which means invalid colum. This one was not there.
	if (entry.first[0] != 0) dir << geometryInterface.pretty(entry.first) << "_" << entry.second << "/";
    }
    // note that we call get() here for every single module. But the string 
    // ops above are probably more expensive anyways...
    MonitorElement* me = iGetter.get(dir.str() + name);
    if (!me) {
      if(bookUndefined)
	edm::LogError("HistogramManager") << "ME " << dir.str() + name << " not found\n";
      // else this will happen quite often
    } else {
      // only touch the able if a me is added. Empty items are illegal.
      AbstractHistogram& histo = t[significantvalues];
      histo.me = me;
      histo.th1 = histo.me->getTH1();
    }
  }
}

void HistogramManager::executeSave(SummationStep& step, Table& t, DQMStore::IBooker& iBooker) {
  // SAVE: traverse the table, book a ME for every TH1.
  for (auto& e : t) {
    if (e.second.me) continue; // if there is a ME already, nothing to do
    assert(!bookUndefined || e.second.th1 || !"Missing histogram. Something is broken.");
    if (!e.second.th1) continue;

    GeometryInterface::Values vals(e.first);
    std::ostringstream dir("");
    for (auto entry : vals.values) {
	dir << geometryInterface.pretty(entry.first) << "_" << entry.second << "/";
    }
    iBooker.setCurrentFolder(top_folder_name + "/" + dir.str());

    if (e.second.th1->GetDimension() == 1) {
      TAxis* ax = e.second.th1->GetXaxis();
      e.second.me = iBooker.book1D(e.second.th1->GetName(), e.second.th1->GetTitle(), ax->GetNbins(), ax->GetXmin(), ax->GetXmax());
      e.second.me->setAxisTitle(ax->GetTitle());
      e.second.me->setAxisTitle(e.second.th1->GetYaxis()->GetTitle(), 2);
    } else {
      TAxis* axX = e.second.th1->GetXaxis();
      TAxis* axY = e.second.th1->GetYaxis();
      e.second.me = iBooker.book2D(e.second.th1->GetName(), e.second.th1->GetTitle(), 
	axX->GetNbins(), axX->GetXmin(), axX->GetXmax(),
	axY->GetNbins(), axY->GetXmin(), axY->GetXmax());
      e.second.me->setAxisTitle(axX->GetTitle());
      e.second.me->setAxisTitle(axY->GetTitle(), 2);
    }

    e.second.me->getTH1()->Add(e.second.th1);
    //delete e.second.th1;
    e.second.th1 = e.second.me->getTH1(); 
  }
}

void HistogramManager::executeGroupBy(SummationStep& step, Table& t) {
  // Simple grouping. Drop colums, add histos if one is already present.
  Table out;
  for (auto& e : t) {
    GeometryInterface::Values const& old_vals(e.first);
    TH1 *th1 = e.second.th1;
    GeometryInterface::Values new_vals;
    for (auto c : step.columns) new_vals.put(old_vals.get(c));
    AbstractHistogram& new_histo = out[new_vals];
    if (!new_histo.th1) {
      new_histo.th1 = (TH1*) th1->Clone();
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
    TH1 *th1 = e.second.th1;
    AbstractHistogram& new_histo = out[vals];
    double reduced_quantity = 0;
    std::string label = "";
    std::string name = this->name;
    // TODO: meaningful semantics in 2D case, errors
    if (step.arg == "MEAN") {
      reduced_quantity = th1->GetMean();
      label = label + "mean of " + th1->GetXaxis()->GetTitle();
      name = "mean_" + name;
    } else if (step.arg == "COUNT") {
      reduced_quantity = th1->GetEntries();
      label = label + "# of " + th1->GetXaxis()->GetTitle() + " entries";
      name = "num_" + name;
    } else /* if (step.arg) == ... TODO: more */ {
      edm::LogError("HistogramManager") << "+++ Reduction '" << step.arg << " not yet implemented\n";
    }
    new_histo.is0d = true;
    new_histo.th1 = new TH1D(name.c_str(), (";;" + label).c_str(), 1, 0, 1); 
    new_histo.th1->SetBinContent(1, reduced_quantity);
  }
  t.swap(out);
} 

void HistogramManager::executeExtend(SummationStep& step, Table& t) {
  // For the moment only X.
  // first pass determines the range.
  std::map<GeometryInterface::Values, int> nbins;
  for (auto& e : t) {
    GeometryInterface::Values new_vals(e.first);
    new_vals.erase(step.columns.at(0));
    TH1 *th1 = e.second.th1;
    int& n = nbins[new_vals];
    assert(th1 || !"invalid histogram");
    n += th1->GetXaxis()->GetNbins(); 
  }

  Table out;
  for (auto& e : t) {
    GeometryInterface::Values const& old_vals(e.first);
    GeometryInterface::Column col0 = old_vals.get(step.columns.at(0)).first;
    GeometryInterface::Values new_vals(old_vals);
    new_vals.erase(step.columns.at(0));
    //std::cout << "+++ Extending for " << geometryInterface.pretty(step.columns.at(0)) << 
     //" actually " << geometryInterface.pretty(col0) << " value "<< old_vals.get(step.columns.at(0)).second << "\n";
    //std::cout << "+++ old_vals: "; for (auto e : old_vals.values) std::cout << e.first[0] << " " << e.first[1] << ":" << e.second << " "; std::cout << "\n";
    std::string colname = geometryInterface.pretty(col0);
    //std::cout << "+++ new_vals: "; for (auto e : new_vals.values) std::cout << geometryInterface.pretty(e.first) << ":" << e.second << " "; std::cout << "\n";
    TH1 *th1 = e.second.th1;
    assert(th1);
    
    AbstractHistogram& new_histo = out[new_vals];
    GeometryInterface::Values copy(new_vals);
    if (!new_histo.th1) {
      //std::cout << "+++ new TH1D for extend ";
      // We need to book. Two cases here: 1D or 2D.
      if (th1->GetDimension() == 1) {
	// Output is 1D
	new_histo.th1 = (TH1*) new TH1D(th1->GetName(), (std::string("") 
					+ th1->GetYaxis()->GetTitle() + " per " + colname
					+ ";" + colname + "/" + th1->GetXaxis()->GetTitle()
					+ ";" + th1->GetYaxis()->GetTitle()).c_str(), 
					nbins[new_vals], 0, nbins[new_vals]);
      } else {
	// output is 2D, input is 2D histograms.
	new_histo.th1 = (TH1*) new TH2D(th1->GetName(), (std::string("") 
					+ th1->GetTitle() + " per " + colname
					+ ";" + colname + "/" + th1->GetXaxis()->GetTitle()
					+ ";" + th1->GetYaxis()->GetTitle()).c_str(), 
					nbins[new_vals], 0, nbins[new_vals],
					th1->GetYaxis()->GetNbins(), 0, th1->GetYaxis()->GetNbins());
      }
      std::cout << "title " << new_histo.th1->GetTitle()<< "\n";
      new_histo.count = 1; // used as a fill pointer. Assumes histograms are ordered correctly (map should provide that)
    } 

    // now add data.
    if (th1->GetDimension() == 1) {
      for (int i = 1; i <= th1->GetXaxis()->GetNbins(); i++) {
	// TODO Error etc.?
	new_histo.th1->SetBinContent(new_histo.count, th1->GetBinContent(i)); 
	new_histo.count += 1; 
      }
    } else {
      // 2D case.
      for (int i = 1; i <= th1->GetXaxis()->GetNbins(); i++) {
	for (int j = 1; j <= th1->GetYaxis()->GetNbins(); j++) {
	  // TODO Error etc.?
	  new_histo.th1->SetBinContent(new_histo.count, j, th1->GetBinContent(i, j)); 
	}
	new_histo.count += 1; 
      }
    }
  }
  t.swap(out);
}

void HistogramManager::executeHarvestingOffline(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {
  if (!enabled) return;
  //edm::LogTrace("HistogramManager") << "HistogramManager: Step2 offline\n";
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
	  executeExtend(step, t);
	  break;
	case SummationStep::EXTEND_Y:
	  assert(!"EXTEND_Y NIY"); break; // TODO: similar to EXTEND_X
	case SummationStep::CUSTOM:
	  assert(customHandler);
	  customHandler(step, t);
	case SummationStep::COUNT:
	case SummationStep::NO_TYPE:
	  assert(!"Operation not supported in harvesting.");
	} // switch
      } // if step2
    } // for each step
  } // for each spec
}


