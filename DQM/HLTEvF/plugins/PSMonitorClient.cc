// C++ headers
#include <string>
#include <cstring>

// boost headers
#include <boost/regex.hpp>

// Root headers
#include <TH1F.h>

// CMSSW headers
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

struct MEPSet {
  std::string name;
  std::string folder;
};

class PSMonitorClient : public DQMEDHarvester {
public:
  explicit PSMonitorClient(edm::ParameterSet const &);
  ~PSMonitorClient() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  static void fillMePSetDescription(edm::ParameterSetDescription &pset);

private:
  static MEPSet getHistoPSet(edm::ParameterSet pset);

  std::string m_dqm_path;

  void dqmEndLuminosityBlock(DQMStore::IBooker &booker,
                             DQMStore::IGetter &getter,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override;
  void dqmEndJob(DQMStore::IBooker &booker, DQMStore::IGetter &getter) override;

  void check(DQMStore::IBooker &booker, DQMStore::IGetter &getter);

  MEPSet psColumnVSlumiPSet;
};

PSMonitorClient::PSMonitorClient(edm::ParameterSet const &config)
    : m_dqm_path(config.getUntrackedParameter<std::string>("dqmPath")),
      psColumnVSlumiPSet(getHistoPSet(config.getParameter<edm::ParameterSet>("me"))) {}

MEPSet PSMonitorClient::getHistoPSet(edm::ParameterSet pset) {
  return MEPSet{
      pset.getParameter<std::string>("name"),
      pset.getParameter<std::string>("folder"),
  };
}

void PSMonitorClient::fillMePSetDescription(edm::ParameterSetDescription &pset) {
  pset.add<std::string>("folder", "HLT/PSMonitoring");
  pset.add<std::string>("name", "psColumnVSlumi");
}

void PSMonitorClient::dqmEndJob(DQMStore::IBooker &booker, DQMStore::IGetter &getter) { check(booker, getter); }

void PSMonitorClient::dqmEndLuminosityBlock(DQMStore::IBooker &booker,
                                            DQMStore::IGetter &getter,
                                            edm::LuminosityBlock const &lumi,
                                            edm::EventSetup const &setup) {
  check(booker, getter);
}

#include "FWCore/MessageLogger/interface/MessageLogger.h"
void PSMonitorClient::check(DQMStore::IBooker &booker, DQMStore::IGetter &getter) {
  std::string folder = psColumnVSlumiPSet.folder;
  std::string name = psColumnVSlumiPSet.name;

  getter.setCurrentFolder(folder);
  MonitorElement *psColumnVSlumi = getter.get(psColumnVSlumiPSet.folder + "/" + psColumnVSlumiPSet.name);
  // if no ME available, return
  if (!psColumnVSlumi) {
    edm::LogWarning("PSMonitorClient") << "no " << psColumnVSlumiPSet.name << " ME is available in "
                                       << psColumnVSlumiPSet.folder << std::endl;
    return;
  }

  /*
  TH2F* h = psColumnVSlumi->getTH2F();
  size_t nbinsX = psColumnVSlumi->getNbinsX();
  size_t nbinsY = psColumnVSlumi->getNbinsY();

  for ( size_t ibinY=1; ibinY < nbinsY; ++ibinY )
    std::cout << h->GetXaxis()->GetBinLabel(ibinY) << std::endl;
    for ( size_t ibinX=1; ibinX< nbinsX; ++ibinX )
      if ( psColumnVSlumi->getBinContent(ibinX) )
	std::cout << "ibinX: " << psColumnVSlumi->getBinContent(ibinX) << std::endl;
  */
}

void PSMonitorClient::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("dqmPath", "HLT/PSMonitoring");

  edm::ParameterSetDescription mePSet;
  fillMePSetDescription(mePSet);
  desc.add<edm::ParameterSetDescription>("me", mePSet);

  descriptions.add("psMonitorClient", desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PSMonitorClient);
