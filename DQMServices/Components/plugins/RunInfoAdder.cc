#include "DQMServices/Components/plugins/RunInfoAdder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"


#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

RunInfoAdder::RunInfoAdder(const edm::ParameterSet& ps)
{

  // Get parameters from configuration file
  addRunNumber_  =  ps.getParameter<bool>("addRunNumber");
  addLumi_       =  ps.getParameter<bool>("addLumi");
  folder_        =  ps.getParameter<std::string>("folder");

  edm::LogInfo("RunInfoAdder") <<  "Constructor RunInfoAdder addRunNumber_ " 
      << addRunNumber_ << " addLumi_ " << addLumi_ << std::endl;
}

RunInfoAdder::~RunInfoAdder() {};

void RunInfoAdder::dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_)
{
  // TODO: add recursive search here.
  std::vector<std::string> dirs;
  igetter_.getContents(dirs, false);
  for (auto& d : dirs) {

    // limit to folder
    if (d.substr(0, folder_.size()) != folder_) continue;

    auto dir = d.substr(0, d.size() - 1); // getContents appends a ':'.
    std::vector<MonitorElement*> mes = igetter_.getContents(dir);

    for (auto me : mes) {
      // This is what the ME uses to check for a root thing...
      if (me->kind() < MonitorElement::DQM_KIND_TH1F) continue;
      TH1* th1 = me->getTH1();
      if (!th1) {
        edm::LogError("RunInfoAdder") << "Not a TH1: " << me->getFullname() << std::endl;
        continue;
      }

      std::string title(th1->GetTitle());
      if (title.find("(Run") == std::string::npos) {
        std::stringstream newtitle;
        newtitle << title;
        if (addRunNumber_) {
          // we use data from the ME, we could also track it in endlumi.
          newtitle << " (Run " << me->run();
        }
        if (addLumi_) {
          newtitle << " Lumi " << me->lumi();
        }
        newtitle << ")";
        th1->SetTitle(newtitle.str().c_str());
      }
    }
  }
}

void RunInfoAdder::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker_, DQMStore::IGetter & igetter_, edm::LuminosityBlock const & iLumi, edm::EventSetup const& iSetup) 
{
  // TODO: See what we can do here in terms of harvesting.
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RunInfoAdder);
