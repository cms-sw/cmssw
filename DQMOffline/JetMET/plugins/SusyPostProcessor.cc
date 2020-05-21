#include <iostream>

#include "DQMOffline/JetMET/plugins/SusyPostProcessor.h"
#include "DQMOffline/JetMET/interface/SusyDQM/Quantile.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

const char* SusyPostProcessor::messageLoggerCatregory = "SusyDQMPostProcessor";

SusyPostProcessor::SusyPostProcessor(const edm::ParameterSet& pSet) {
  iConfig = pSet;

  SUSYFolder = iConfig.getParameter<string>("folderName");
  _quantile = iConfig.getParameter<double>("quantile");
}

SusyPostProcessor::~SusyPostProcessor() {}

void SusyPostProcessor::QuantilePlots(MonitorElement*& ME, double q_value, DQMStore::IBooker& ibooker_) {
  if (ME->getTH1()->GetEntries() > 0.) {
    Quantile q(static_cast<const TH1*>(ME->getTH1()));
    Float_t mean = q[q_value].first;
    Float_t RMS = q[q_value].second;

    Float_t xLow = -5.5;
    Float_t xUp = 9.5;
    Int_t NBin = 15;

    if (mean > 0.) {
      Float_t DBin = RMS * TMath::Sqrt(12.) / 2.;
      xLow = mean - int(mean / DBin + 2) * DBin;
      xUp = int(0.2 * mean / DBin) * DBin + mean + 5 * DBin;
      NBin = (xUp - xLow) / DBin;
    }

    ibooker_.setCurrentFolder(ME->getPathname());
    TString name = ME->getTH1()->GetName();
    name += "_quant";
    ME = ibooker_.book1D(name, "", NBin, xLow, xUp);
    ME->Fill(mean - RMS);
    ME->Fill(mean + RMS);
  }
}

void SusyPostProcessor::dqmEndJob(DQMStore::IBooker& ibook_, DQMStore::IGetter& iget_) {
  // MET
  //----------------------------------------------------------------------------
  iget_.setCurrentFolder("JetMET/MET");

  Dirs = iget_.getSubdirs();

  std::vector<std::string> metFolders;

  metFolders.push_back("Uncleaned/");
  metFolders.push_back("Cleaned/");

  //Need our own copy for thread safety
  TF1 mygaus("mygaus", "gaus");

  for (auto& Dir : Dirs) {
    std::string prefix = "dummy";

    if (size_t(Dir.find("met")) != string::npos)
      prefix = "met";
    if (size_t(Dir.find("pfMet")) != string::npos)
      prefix = "pfMET";

    for (const auto& metFolder : metFolders) {
      std::string dirName = Dir + "/" + metFolder;

      MEx = iget_.get(dirName + "/" + "MEx");
      MEy = iget_.get(dirName + "/" + "MEy");

      if (MEx && MEx->kind() == MonitorElement::Kind::TH1F) {
        if (MEx->getTH1F()->GetEntries() > 50)
          MEx->getTH1F()->Fit(&mygaus, "q");
      }

      if (MEy && MEy->kind() == MonitorElement::Kind::TH1F) {
        if (MEy->getTH1F()->GetEntries() > 50)
          MEy->getTH1F()->Fit(&mygaus, "q");
      }
    }
  }

  // SUSY
  //----------------------------------------------------------------------------
  iget_.setCurrentFolder(SUSYFolder);
  Dirs = iget_.getSubdirs();
  for (auto& Dir : Dirs) {
    size_t found = Dir.find("Alpha");
    if (found != string::npos)
      continue;
    if (!iget_.dirExists(Dir)) {
      edm::LogError(messageLoggerCatregory) << "Directory " << Dir << " doesn't exist!!";
      continue;
    }
    vector<MonitorElement*> histoVector = iget_.getContents(Dir);
    for (auto& i : histoVector) {
      QuantilePlots(i, _quantile, ibook_);
    }
  }
}
