#include "DQMOffline/PFTau/plugins/PFDQMEventSelector.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>

//
// -- Constructor
//
PFDQMEventSelector::PFDQMEventSelector(const edm::ParameterSet &pset) {
  usesResource("DQMStore");
  verbose_ = pset.getParameter<bool>("DebugOn");
  inputFileName_ = pset.getParameter<std::string>("InputFileName");
  folderNames_ = pset.getParameter<std::vector<std::string>>("FolderNames");

  nEvents_ = 0;
  nSelectedEvents_ = 0;
  fileOpened_ = false;
}
//
// -- Destructor
//
PFDQMEventSelector::~PFDQMEventSelector() {}

//
// -- BeginJob
//
void PFDQMEventSelector::beginJob() {
  dqmStore_ = edm::Service<DQMStore>().operator->();
  fileOpened_ = openInputFile();
}
//
// -- Event Filtering
//
bool PFDQMEventSelector::filter(edm::Event &iEvent, edm::EventSetup const &iSetup) {
  nEvents_++;
  if (!fileOpened_)
    return false;

  edm::RunNumber_t runNb = iEvent.id().run();
  edm::EventNumber_t evtNb = iEvent.id().event();
  edm::LuminosityBlockNumber_t lumiNb = iEvent.id().luminosityBlock();
  std::ostringstream eventid_str;
  eventid_str << runNb << "_" << evtNb << "_" << lumiNb;

  for (std::vector<std::string>::const_iterator ifolder = folderNames_.begin(); ifolder != folderNames_.end();
       ifolder++) {
    std::string path = "ParticleFlow/" + (*ifolder) + "/BadEvents";
    MonitorElement *me = dqmStore_->get(path + "/" + eventid_str.str());
    if (me) {
      nSelectedEvents_++;
      if (verbose_)
        std::cout << " Total Events " << nEvents_ << " Selected Events " << nSelectedEvents_ << " Run # : " << runNb
                  << " Event # : " << evtNb << " Luminosity Block # : " << lumiNb << std::endl;
      return true;
    }
  }
  return false;
}
//
// -- End Job
//
void PFDQMEventSelector::endJob() {
  if (verbose_)
    std::cout << " Total Events " << nEvents_ << " Selected Events " << nSelectedEvents_ << std::endl;
}
//
// -- Open Input File
//
bool PFDQMEventSelector::openInputFile() {
  if (inputFileName_.empty())
    return false;
  edm::LogInfo("SiStripOfflineDQM") << "SiStripOfflineDQM::openInputFile: Accessing root File" << inputFileName_;
  dqmStore_->open(inputFileName_, false, "", "", DQMStore::StripRunDirs);
  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFDQMEventSelector);
