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

// Logger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HistogramManager::HistogramManager(const edm::ParameterSet& iconfig) :
  iConfig(iconfig),
  topFolderName(iconfig.getParameter<std::string>("TopFolderName"))
{ }

void HistogramManager::addSpec(SummationSpecification spec) {
  specs.push_back(spec);
  tables.push_back(Table());
}

SummationSpecificationBuilder HistogramManager::addSpec() {
  addSpec(SummationSpecification());
  return SummationSpecificationBuilder(specs[specs.size()-1], geometryInterface);
}

// note that this will be pretty hot. Ideally it should be malloc-free.
void HistogramManager::fill(double x, double y, DetId sourceModule, const edm::Event *sourceEvent, int col, int row) {
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
	case SummationStep::NO_TYPE:
	  assert(!"step1 spec step is NYI.");
	}
      }
    }
    if (dimensions == 0) {
      t[significantvalues].fill(0);
    } else if (dimensions == 1)
      t[significantvalues].fill(x);
    else /* dimensions == 2 */ {
      t[significantvalues].fill(x, y);
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
  if (!geometryInterface.loaded()) {
    geometryInterface.load(iSetup, iConfig);
  }

  for (unsigned int i = 0; i < specs.size(); i++) {
    auto& s = specs[i];
    auto& t = tables[i];
    for (auto iq : geometryInterface.allModules()) {
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
      GeometryInterface::Values significantvalues;
      geometryInterface.extractColumns(s.steps[0].columns, iq, significantvalues);
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
	    assert(dimensions != 1 || !"1D to 1D reduce NYI");
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
	    assert(dimensions != 2 || !"2D to 2D reduce NYI");
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
	  case SummationStep::NO_TYPE:
	    assert(!"step1 spec step is NYI.");
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

      iBooker.setCurrentFolder(topFolderName + "/" + dir.str());

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
  // this should also give us the GeometryInterface for offline, though it is a bit dirty and might explode.
  if (!geometryInterface.loaded()) {
    geometryInterface.load(iSetup, iConfig);
  }
}

void HistogramManager::executeHarvestingOffline(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {
  //edm::LogTrace("HistogramManager") << "HistogramManager: Step2 offline\n";
  // Debug output
  for (auto& s : specs) {
    edm::LogInfo log("HistogramManager");
    log << "Specs for " << name << " ";
    s.dump(log, geometryInterface);
  }

  // This is esentially the booking code if step1, to reconstruct the ME names.
  // Once we have a name we load the ME and put it into the table.
  for (unsigned int i = 0; i < specs.size(); i++) {
    auto& s = specs[i];
    auto& t = tables[i];
    for (auto iq : geometryInterface.allModules()) {
      std::string name = this->name;
      GeometryInterface::Values significantvalues;
      geometryInterface.extractColumns(s.steps[0].columns, iq, significantvalues);
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
	  case SummationStep::NO_TYPE:
	    assert(!"step1 spec step is NYI.");
	  }
	}
      }

      AbstractHistogram& histo = t[significantvalues];
      std::ostringstream dir(topFolderName + "/", std::ostringstream::ate);
      for (auto c : s.steps[0].columns) {
	  auto entry = significantvalues.get(c);
	  // col[0] = 0 implies col[1] = 0 which means invalid colum. This one was not there.
	  if (entry.first[0] != 0) dir << geometryInterface.pretty(entry.first) << "_" << entry.second << "/";
      }
      histo.me = iGetter.get(dir.str() + name);
      if (!histo.me) {
	edm::LogError("HistogramManager") << "ME " << dir.str() + name << " not found\n";
      } else {
	histo.th1 = histo.me->getTH1();
      }
    }
    
    // now execute step2.
    for (SummationStep step : s.steps) {
      if (step.stage == SummationStep::STAGE2) {

	//TODO: change labels, dimensionality, range, colums as fits
	
	// SAVE: traverse the table, book a ME for every TH1.
	if (step.type == SummationStep::SAVE) {
	  for (auto& e : t) {
	    if (e.second.me) continue; // if there is a ME already, nothing to do
	    assert(e.second.th1 || !"Missing histogram. Something is broken.");

	    GeometryInterface::Values vals(e.first);
	    std::ostringstream dir("");
	    for (auto c : s.steps[0].columns) {
		auto entry = vals.get(c);
		// col[0] = 0 implies col[1] = 0 which means invalid colum. This one was not there.
		if (entry.first[0] != 0) dir << geometryInterface.pretty(entry.first) << "_" << entry.second << "/";
	    }
	    iBooker.setCurrentFolder(topFolderName + "/" + dir.str());

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

	// Simple grouping. Drop colums, add histos if one is already present.
	if (step.type == SummationStep::GROUPBY) {
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

        if (step.type == SummationStep::REDUCE) {
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

        if (step.type == SummationStep::EXTEND_X) {
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
	    GeometryInterface::Values new_vals(old_vals);
	    //GeometryInterface::Value xval = new_vals[step.columns.at(0)]; // We rely on correct ordering
	    GeometryInterface::Column col0 = new_vals.get(step.columns.at(0)).first;
	    std::string colname = geometryInterface.pretty(col0);
	    new_vals.erase(col0);
	    TH1 *th1 = e.second.th1;

	    AbstractHistogram& new_histo = out[new_vals];
	    if (!new_histo.th1) {
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
      }
    }
  }
}


