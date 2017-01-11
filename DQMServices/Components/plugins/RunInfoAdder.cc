#include "DQMServices/Components/plugins/RunInfoAdder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include <iostream>
#include <string>
#include <sstream>

RunInfoAdder::RunInfoAdder(const edm::ParameterSet& ps)
{

  // Get parameters from configuration file
  addRunNumber_  =  ps.getParameter<bool>("addRunNumber");
  addLumi_       =  ps.getParameter<bool>("addLumi");
  folder_        =  ps.getParameter<std::vector<std::string>>("folder");

  run_ = 0;

  edm::LogInfo("RunInfoAdder") <<  "Constructor RunInfoAdder addRunNumber_ " 
      << addRunNumber_ << " addLumi_ " << addLumi_ << std::endl;
}

RunInfoAdder::~RunInfoAdder() {};

void RunInfoAdder::dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_)
{
  std::vector<std::string> dirs;
  igetter_.getContents(dirs, false);
  for (auto& d : dirs) {

    // limit to given folders
    bool apply = false;
    for (auto& f : folder_)
      if (d.substr(0, f.size()) == f) apply = true;
    if (!apply) continue;

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
      // Check to prevent double adding. This _should_ not be needed.
      if (title.find("(Run") == std::string::npos) {
        std::stringstream newtitle;
        newtitle << title;
        if (addRunNumber_) {
          // we use data from endLumi, value in ME seems broken.
          newtitle << " (Run " << run_;
        }
        if (addLumi_) {
          // does this work? Not sure.
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
  // Disable in multi-run case (maybe use ME value then?)
  if (run_ != 0 && iLumi.run() != run_) addRunNumber_ = false; 
  run_ = iLumi.run();
  // TODO: See what we can do here in terms of harvesting.
  // Maybe call dqmEndJob to do the work when we do online harvesting.
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RunInfoAdder);
