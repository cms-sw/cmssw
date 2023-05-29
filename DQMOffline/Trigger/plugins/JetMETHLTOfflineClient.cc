// Migrated to use DQMEDHarvester by: Jyothsna Rani Komaragiri, Oct 2014

#include "DQMOffline/Trigger/interface/JetMETHLTOfflineClient.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"

JetMETHLTOfflineClient::JetMETHLTOfflineClient(const edm::ParameterSet& iConfig) : conf_(iConfig) {
  debug_ = false;
  verbose_ = false;

  processname_ = iConfig.getParameter<std::string>("processname");

  hltTag_ = iConfig.getParameter<std::string>("hltTag");
  if (debug_)
    std::cout << hltTag_ << std::endl;

  dirName_ = iConfig.getParameter<std::string>("DQMDirName");
}

JetMETHLTOfflineClient::~JetMETHLTOfflineClient() = default;

void JetMETHLTOfflineClient::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  ibooker.setCurrentFolder(dirName_);

  LogDebug("JetMETHLTOfflineClient") << "dqmEndJob" << std::endl;
  if (debug_)
    std::cout << "dqmEndJob" << std::endl;

  std::vector<MonitorElement*> hltMEs;

  // Look at all folders, go to the subfolder which includes the string "Eff"
  std::vector<std::string> fullPathHLTFolders = igetter.getSubdirs();
  for (auto& fullPathHLTFolder : fullPathHLTFolders) {
    // Move on only if the folder name contains "Eff" Or "Trigger Summary"
    if (debug_)
      std::cout << fullPathHLTFolder << std::endl;
    if ((fullPathHLTFolder.find("Eff") != std::string::npos)) {
      ibooker.setCurrentFolder(fullPathHLTFolder);
    } else {
      continue;
    }

    // Look at all subfolders, go to the subfolder which includes the string "Eff"
    std::vector<std::string> fullSubPathHLTFolders = igetter.getSubdirs();
    for (auto& fullSubPathHLTFolder : fullSubPathHLTFolders) {
      if (debug_)
        std::cout << fullSubPathHLTFolder << std::endl;
      ibooker.setCurrentFolder(fullSubPathHLTFolder);

      // Look at all MonitorElements in this folder
      hltMEs = igetter.getContents(fullSubPathHLTFolder);
      LogDebug("JetMETHLTOfflineClient") << "Number of MEs for this HLT path = " << hltMEs.size() << std::endl;

      for (unsigned int k = 0; k < hltMEs.size(); k++) {
        if (debug_)
          std::cout << hltMEs[k]->getName() << std::endl;

        //-----
        if ((hltMEs[k]->getName().find("ME_Numerator") != std::string::npos) &&
            hltMEs[k]->getName().find("ME_Numerator") == 0) {
          std::string name = hltMEs[k]->getName();
          name.erase(0, 12);  // Removed "ME_Numerator"
          if (debug_)
            std::cout << "==name==" << name << std::endl;
          //	  if( name.find("EtaPhi") !=std::string::npos ) continue; // do not consider EtaPhi 2D plots

          for (unsigned int l = 0; l < hltMEs.size(); l++) {
            if (hltMEs[l]->getName() == "ME_Denominator" + name) {
              // found denominator too
              if (name.find("EtaPhi") != std::string::npos) {
                TH2F* tNumerator = hltMEs[k]->getTH2F();
                TH2F* tDenominator = hltMEs[l]->getTH2F();

                std::string title = "Eff_" + hltMEs[k]->getTitle();

                auto* teff = (TH2F*)tNumerator->Clone(title.c_str());
                teff->Divide(tNumerator, tDenominator, 1, 1);
                ibooker.book2D("ME_Eff_" + name, teff);
                delete teff;
              } else {
                TH1F* tNumerator = hltMEs[k]->getTH1F();
                TH1F* tDenominator = hltMEs[l]->getTH1F();

                std::string title = "Eff_" + hltMEs[k]->getTitle();

                auto* teff = (TH1F*)tNumerator->Clone(title.c_str());
                teff->Divide(tNumerator, tDenominator, 1, 1);
                ibooker.book1D("ME_Eff_" + name, teff);
                delete teff;
              }
            }  // Denominator
          }    // Loop-l
        }      // Numerator

      }  // Loop-k
    }    // fullSubPath
  }      // fullPath
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetMETHLTOfflineClient);
