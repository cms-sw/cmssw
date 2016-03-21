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
  return SummationSpecificationBuilder(specs[specs.size()-1]);
}

// note that this will be pretty hot. Ideally it should be malloc-free.
void HistogramManager::fill(double value, DetId sourceModule, const edm::Event *sourceEvent, int col, int row) {
  auto iq = GeometryInterface::InterestingQuantities{
              sourceModule, sourceEvent, col, row
	    };							    
  for (unsigned int i = 0; i < specs.size(); i++) {
    auto& s = specs[i];
    auto& t = tables[i];
    auto significantvalues = geometryInterface.extractColumns(s.steps[0].columns, iq);
    // TODO: more step1 steps must be excuted here. Step1 steps can be applied per-sample.
    // Step1 steps are those that have s.stage = SummationStep::STAGE1, and step[0].
    t[significantvalues].fill(value);
  }
}
  
void HistogramManager::book(DQMStore::IBooker& iBooker, edm::EventSetup const& iSetup) {
  if (!geometryInterface.loaded()) {
    geometryInterface.load(iSetup, iConfig);
  }

    // TODO: We need 2 passes, one to count the elements for EXTEND-ranges, one for actual booking.
  for (unsigned int i = 0; i < specs.size(); i++) {
    auto& s = specs[i];
    auto& t = tables[i];
    for (auto iq : geometryInterface.allModules()) {
      auto dimensions = this->dimensions;
      std::ostringstream name(this->name, std::ostringstream::ate);
      std::ostringstream dir("");
      std::ostringstream title(this->title, std::ostringstream::ate);
      std::ostringstream xlabel(this->xlabel, std::ostringstream::ate);
      std::ostringstream ylabel(this->ylabel, std::ostringstream::ate);
      auto significantvalues = geometryInterface.extractColumns(s.steps[0].columns, iq);
      for (SummationStep step : s.steps) {
	if (step.stage == SummationStep::STAGE1) {
	  //TODO: change labels, dimensionality, range, colums as fits
	  if (step.type == SummationStep::SAVE) continue; // this happens automatically
	  assert(!"step1 specs are NYI.");
	}
      }

      AbstractHistogram& histo = t[significantvalues];
      if (histo.me) continue;

      for (auto c : s.steps[0].columns) {
	dir << c << "_" << std::hex << significantvalues[c] << "/";
      }

      iBooker.setCurrentFolder(topFolderName + "/" + dir.str());

      if (dimensions == 1) {
	title << ";" << xlabel.str();
      	histo.me = iBooker.book1D(name.str().c_str(), title.str().c_str(), range_nbins, range_min, range_max);
      } else if (dimensions == 2) {
	title << ";" << xlabel.str() << ";" << ylabel.str();
	// TODO: proper 2D range.
	histo.me = iBooker.book2D(name.str().c_str(), title.str().c_str(), 
	                          range_nbins, range_min, range_max, range_nbins, range_min, range_max);
      } else {
	std::cout << "Booking " << dimensions << " dimensions not supported.\n";
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
  //std::cout << "+++ HistogramManager: Step2 offline\n";
  // Debug output
  //for (auto& s : specs) {
    //std::cout << "+++ " << name << " ";
    //s.dump(std::cout);
  //}

  // This is esentially the booking code if step1, to reconstruct the ME names.
  // Once we have a name we load the ME and put it into the table.
  for (unsigned int i = 0; i < specs.size(); i++) {
    auto& s = specs[i];
    auto& t = tables[i];
    for (auto iq : geometryInterface.allModules()) {
      std::ostringstream name(this->name, std::ostringstream::ate);
      std::ostringstream dir(topFolderName + "/", std::ostringstream::ate);
      auto significantvalues = geometryInterface.extractColumns(s.steps[0].columns, iq);
      for (SummationStep step : s.steps) {
	if (step.stage == SummationStep::STAGE1) {
	  //TODO: change labels, dimensionality, range, colums as fits
	}
      }

      AbstractHistogram& histo = t[significantvalues];
      for (auto c : s.steps[0].columns) {
	dir << c << "_" << std::hex << significantvalues[c] << "/";
      }
      histo.me = iGetter.get(dir.str() + name.str());
      if (!histo.me) {
	std::cout << "+++ ME " << dir.str() + name.str() << " not found\n";
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
	      if (vals.find(c) != vals.end())
		dir << c << "_" << std::hex << vals[c] << "/";
	    }
	    iBooker.setCurrentFolder(topFolderName + "/" + dir.str());

	    if (e.second.th1->GetDimension() == 1) {
	      TAxis* ax = e.second.th1->GetXaxis();
	      e.second.me = iBooker.book1D(e.second.th1->GetName(), e.second.th1->GetTitle(), ax->GetNbins(), ax->GetXmin(), ax->GetXmax());
	      e.second.me->setAxisTitle(ax->GetTitle());
	      e.second.me->setAxisTitle(e.second.th1->GetYaxis()->GetTitle(), 2);
	    } else {
	      assert(!"NIY");
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
	    for (auto col : step.columns) {
	      auto it = old_vals.find(col);
	      if (it != old_vals.end()) {
		new_vals.insert(*it);
	      }
	    }
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
              std::cout << "+++ Reduction '" << step.arg << " not yet implemented\n";
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
	    new_vals.erase(step.columns.at(0));
	    TH1 *th1 = e.second.th1;

	    AbstractHistogram& new_histo = out[new_vals];
	    if (!new_histo.th1) {
	      // We need to book. Two cases here: 1D or 2D.
	      if (th1->GetDimension() == 1) {
		// Output is 1D
	        new_histo.th1 = (TH1*) new TH1D(th1->GetName(), (std::string("") 
		                                + th1->GetYaxis()->GetTitle() + " per " + step.columns.at(0) 
		                                + ";" + step.columns.at(0) + "/" + th1->GetXaxis()->GetTitle()
						+ ";" + th1->GetYaxis()->GetTitle()).c_str(), 
		                                nbins[new_vals], 0, nbins[new_vals]);
		new_histo.count = 1; // used as a fill pointer. Assumes histograms are ordered correctly (map should provide that)
		// TODO: actually we should use the label of the input histograms. But for 0D we dont have anything there...
	      } else {
                // output is 2D, input is 2D histograms.
		assert(!"2D NIY");
	      }
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
	      assert(!"2D histo concat NIY");
	    }
	  }
	  t.swap(out);
	}
	// TODO: more.

      }
    }
  }
}


