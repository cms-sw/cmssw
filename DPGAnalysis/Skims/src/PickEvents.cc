// -*- C++ -*-
//
// Package:    PickEvents
// Class:      PickEvents
//
/**\class PickEvents PickEvents.cc DPGAnalysis/PickEvents/src/PickEvents.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Michael Henry Schmitt
//         Created:  Mon Sep 15 19:36:37 CEST 2008
// $Id: PickEvents.cc,v 1.4 2010/08/07 14:55:55 wmtan Exp $
//         Modified: 27/03/2009 Luca Malgeri
//                   reading external file, defining selection syntax
//         Modified: 30/06/2014 Giovanni Franzoni
//                   reading run-lumisection list from json
//
//

// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <limits>
#include <cassert>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"

// ordering function to sort LuminosityBlockRange based on the starting run number
bool orderLuminosityBlockRange(edm::LuminosityBlockRange u, edm::LuminosityBlockRange v) {
  return (u.startRun() < v.startRun());
}

//
// class declaration
//

class PickEvents : public edm::EDFilter {
public:
  explicit PickEvents(const edm::ParameterSet&);
  ~PickEvents() override;

private:
  void beginJob() override;
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  std::string listrunevents_;
  std::string listruneventsinpath_;
  bool isRunLsBased_;
  std::vector<edm::LuminosityBlockRange> luminositySectionsBlockRanges_;

  std::vector<bool> whattodo;
  std::vector<edm::RunNumber_t> startrun;
  std::vector<edm::RunNumber_t> endrun;
  std::vector<edm::EventNumber_t> startevent;
  std::vector<edm::EventNumber_t> endevent;

  int nEventsAnalyzed;
  int nEventsSelected;
};

PickEvents::PickEvents(const edm::ParameterSet& iConfig) {
  isRunLsBased_ = iConfig.getParameter<bool>("IsRunLsBased");
  luminositySectionsBlockRanges_ =
      iConfig.getUntrackedParameter<std::vector<edm::LuminosityBlockRange> >("LuminositySectionsBlockRange");

  listruneventsinpath_ = iConfig.getUntrackedParameter<std::string>("RunEventList", "");
  edm::FileInPath listruneventstmp(listruneventsinpath_);
  listrunevents_ = listruneventstmp.fullPath();

  // sanity checks
  if (isRunLsBased_ && luminositySectionsBlockRanges_.empty()) {
    assert("ERROR: selection based on run/Lumisection from json file, but LuminositySectionsBlockRange is emptpy." ==
           nullptr);
  }
  if ((!isRunLsBased_) && !luminositySectionsBlockRanges_.empty()) {
    assert("ERROR: selection based on run/event from txt file, but LuminositySectionsBlockRange is not emptpy." ==
           nullptr);
  }

  if (isRunLsBased_) {
    std::cout << "Selection based on run/luminositySection; file with run/event list: " << std::endl;
  } else {
    std::cout << "Selection based on run/event; file with run/event list: " << listrunevents_ << std::endl;
  }
}

PickEvents::~PickEvents() {}

bool PickEvents::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  RunNumber_t kRun = iEvent.id().run();
  EventNumber_t kEvent = iEvent.id().event();
  LuminosityBlockNumber_t kLumi = iEvent.id().luminosityBlock();

  bool selectThisEvent = false;

  // two alternative definition of the filter selection are possible, according to isRunLsBased_

  if (isRunLsBased_) {
    // std::cout << "GF DEBUG: kRun is " << kRun << " kLumi is: " << kLumi << std::endl;

    for (std::vector<edm::LuminosityBlockRange>::iterator oneLumiRange = luminositySectionsBlockRanges_.begin();
         oneLumiRange != luminositySectionsBlockRanges_.end();
         ++oneLumiRange) {
      // luminositySectionsBlockRanges_ is sorted according to startRun()
      // => if kRun below it, you can stop the loop and return false
      if (kRun < (*oneLumiRange).startRun()) {
        // std::cout << "GF DEBUG: LS has NOT PASSED (early bail-out) ! ***" << std::endl;
        break;
      }

      // if endRun() below kRun, go to the next iteration
      if ((*oneLumiRange).endRun() < kRun)
        continue;

      // if the run number and lumi section match => exit from the loop
      if ((*oneLumiRange).startLumi() <= kLumi && kLumi <= (*oneLumiRange).endLumi()) {
        selectThisEvent = true;
        // std::cout << "GF DEBUG: LS HAS PASSED ! ***" << std::endl;
        break;
      }
    }

  }       // end of isRunLsBased_
  else {  // !isRunLsBased_

    for (unsigned int cond = 0; cond < whattodo.size(); cond++) {
      //       std::string what;
      if (kRun >= startrun[cond] && kRun <= endrun[cond] && kEvent >= startevent[cond] &&
          kEvent <= endevent[cond]) {  // it's in the range, use
        selectThisEvent = whattodo[cond];
      }
    }  // loop on whattodo

  }  // !isRunLsBased_

  nEventsAnalyzed++;
  if (selectThisEvent)
    nEventsSelected++;
  //   if (selectThisEvent) std::cout << "Event selected: " << kRun << " " << kEvent << std::endl;

  return selectThisEvent;
}

void PickEvents::beginJob() {
  using namespace std;

  std::string line;
  std::string buf;

  std::stringstream ss;
  std::vector<std::string> tokens;

  nEventsAnalyzed = 0;
  nEventsSelected = 0;

  if (isRunLsBased_) {
    // sorting luminositySectionsBlockRanges_ according to the starting run of the block allows the speedup the search by an average factor 2
    std::sort(luminositySectionsBlockRanges_.begin(), luminositySectionsBlockRanges_.end(), orderLuminosityBlockRange);
  }  // if isRunLsBased_

  else {  // !isRunLsBased_

    // open file listevent file
    std::ifstream listfile;
    listfile.open(listrunevents_.c_str());
    if (listfile.is_open()) {
      while (!listfile.eof()) {
        getline(listfile, line);
        ss.clear();
        ss.str(line);
        tokens.clear();
        while (ss >> buf) {
          tokens.push_back(buf);
          //	      std::cout << buf << std::endl;
        }
        // std::cout << tokens.size() << std::endl;
        if (tokens.size() < 3) {
          //	      std::cout << "strange selection line:" << line << std::endl;
          //	      std::cout << "skipping it" << std::endl;
          continue;
        }
        if (tokens[0] == "-" || tokens[0] == "+") {
          // it's a selection line, use it
          if (tokens[0] == "-")
            whattodo.push_back(false);
          else
            whattodo.push_back(true);

          // start with run selecion
          int loc = tokens[1].find(':', 0);

          std::string first = tokens[1].substr(0, loc);
          startrun.push_back((edm::RunNumber_t)atoi(first.c_str()));

          std::string last = tokens[1].substr(loc + 1, tokens[1].size());
          if (last == "infty")
            endrun.push_back(std::numeric_limits<unsigned int>::max());
          else
            endrun.push_back((edm::RunNumber_t)atoi(last.c_str()));

          // then event number selecion
          loc = tokens[2].find(':', 0);

          first = tokens[2].substr(0, loc);
          startevent.push_back((edm::EventNumber_t)atoi(first.c_str()));

          last = tokens[2].substr(loc + 1, tokens[2].size());
          if (last == "infty")
            endevent.push_back(edm::EventID::maxEventNumber());
          //		endevent.push_back(std::numeric_limits<long long int>::max());
          else
            endevent.push_back((edm::EventNumber_t)atoi(last.c_str()));
        }
      }
      listfile.close();
      // printout summary
      std::cout << "Summary from list of run/event number selection" << std::endl;
      for (unsigned int cond = 0; cond < whattodo.size(); cond++) {
        std::string what;
        if (whattodo[cond])
          what = "select";
        else
          what = "reject";
        std::cout << what << " ";
        std::cout << "from run " << startrun[cond] << " to run " << endrun[cond] << " ";
        std::cout << "from eve " << startevent[cond] << " to eve " << endevent[cond] << std::endl;
      }
    }

    else
      std::cout << "Unable to open file";

  }  // !isRunLsBased_
}
void PickEvents::endJob() {
  using namespace std;
  std::cout << "================================================\n"
            << "  n Events Analyzed ............... " << nEventsAnalyzed << std::endl
            << "  n Events Selected ............... " << nEventsSelected << std::endl
            << "================================================\n\n";
}

DEFINE_FWK_MODULE(PickEvents);
