#include "DQM/HLTEvF/plugins/TriggerRatesMonitorClient.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// -------------------------------------- Constructor --------------------------------------------
//
TriggerRatesMonitorClient::TriggerRatesMonitorClient(const edm::ParameterSet& iConfig)
    : m_dqm_path(iConfig.getUntrackedParameter<std::string>("dqmPath")) {
  edm::LogInfo("TriggerRatesMonitorClient")
      << "Constructor  TriggerRatesMonitorClient::TriggerRatesMonitorClient " << std::endl;
}

//
// -------------------------------------- beginJob --------------------------------------------
//
void TriggerRatesMonitorClient::beginJob() {
  edm::LogInfo("TriggerRatesMonitorClient") << "TriggerRatesMonitorClient::beginJob " << std::endl;
}

//
// -------------------------------------- get and book in the endJob --------------------------------------------
//
void TriggerRatesMonitorClient::dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_) {
  // create and cd into new folder
  ibooker_.setCurrentFolder(m_dqm_path);

  //get available histograms
  std::vector<std::string> directories = igetter_.getSubdirs();
  m_hltXpd_counts.resize(directories.size());

  int i = 0;
  for (auto const& dir : directories) {
    //    std::cout << "dir: " << dir << std::endl;
    ibooker_.setCurrentFolder(m_dqm_path + "/" + dir);

    std::vector<std::string> const& all_mes = igetter_.getMEs();
    std::vector<std::string> mes;
    for (auto const& me : all_mes)
      if (me.find("accept") != std::string::npos)
        mes.push_back(me);

    int nbinsY = mes.size();
    double xminY = 0.;
    double xmaxY = xminY + 1. * nbinsY;
    int nbinsX = 0;
    int ibinY = 1;
    for (auto const& me : mes) {
      //      std::cout << "me: " << me << std::endl;
      ibooker_.setCurrentFolder(m_dqm_path + "/" + dir);
      TH1F* histo = igetter_.get(me)->getTH1F();

      if (m_hltXpd_counts[i] == nullptr) {
        // get range and binning for new MEs
        nbinsX = histo->GetNbinsX();
        double xminX = histo->GetXaxis()->GetXmin();
        double xmaxX = histo->GetXaxis()->GetXmax();

        //book new histogram
        std::string hname = dir + "_summary";
        ibooker_.setCurrentFolder(m_dqm_path);
        m_hltXpd_counts[i] = ibooker_.book2D(hname, hname, nbinsX, xminX, xmaxX, nbinsY, xminY, xmaxY)->getTH2F();
      } else {
        m_hltXpd_counts[i]->GetYaxis()->SetBinLabel(ibinY, me.c_str());
      }

      // handle mes
      for (int ibinX = 1; ibinX <= nbinsX; ++ibinX) {
        float rate = histo->GetBinContent(ibinX);
        m_hltXpd_counts[i]->SetBinContent(ibinX, ibinY, rate);
      }
      ibinY++;
    }

    i++;
  }
}

//
// -------------------------------------- get in the endLumi if needed --------------------------------------------
//
void TriggerRatesMonitorClient::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker_,
                                                      DQMStore::IGetter& igetter_,
                                                      edm::LuminosityBlock const& iLumi,
                                                      edm::EventSetup const& iSetup) {
  edm::LogInfo("TriggerRatesMonitorClient") << "TriggerRatesMonitorClient::endLumi " << std::endl;
}

void TriggerRatesMonitorClient::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("dqmPath", "HLT/Datasets");
  descriptions.add("triggerRatesMonitorClient", desc);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TriggerRatesMonitorClient);
