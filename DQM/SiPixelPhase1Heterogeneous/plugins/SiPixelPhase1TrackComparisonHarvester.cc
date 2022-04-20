#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
class SiPixelPhase1TrackComparisonHarvester : public DQMEDHarvester {
public:
  explicit SiPixelPhase1TrackComparisonHarvester(const edm::ParameterSet&);
  ~SiPixelPhase1TrackComparisonHarvester() override = default;
  void dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // ----------member data ---------------------------
  const std::string topFolder_;
};

SiPixelPhase1TrackComparisonHarvester::SiPixelPhase1TrackComparisonHarvester(const edm::ParameterSet& iConfig)
    : topFolder_(iConfig.getParameter<std::string>("topFolderName")) {}

void SiPixelPhase1TrackComparisonHarvester::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  MonitorElement* hpt_eta_tkAllCPU = igetter.get(topFolder_ + "/ptetatrkAllCPU");
  MonitorElement* hpt_eta_tkAllCPUmatched = igetter.get(topFolder_ + "/ptetatrkAllCPUmatched");
  MonitorElement* hphi_z_tkAllCPU = igetter.get(topFolder_ + "/phiztrkAllCPU");
  MonitorElement* hphi_z_tkAllCPUmatched = igetter.get(topFolder_ + "/phiztrkAllCPUmatched");

  if (hpt_eta_tkAllCPU == nullptr or hpt_eta_tkAllCPUmatched == nullptr or hphi_z_tkAllCPU == nullptr or
      hphi_z_tkAllCPUmatched == nullptr) {
    edm::LogError("SiPixelPhase1TrackComparisonHarvester")
        << "MEs needed for this module are not found in the input file. Skipping.";
    return;
  }

  ibooker.cd();
  ibooker.setCurrentFolder(topFolder_);
  MonitorElement* hpt_eta_matchRatio = ibooker.book2D(
      "matchingeff_pt_eta", "Efficiency of track matching; #eta; p_{T} [GeV];", 30, -M_PI, M_PI, 200, 0., 200.);
  MonitorElement* hphi_z_matchRatio = ibooker.book2D(
      "matchingeff_phi_z", "Efficiency of track matching; #phi; z [cm];", 30, -M_PI, M_PI, 30, -30., 30.);

  hpt_eta_matchRatio->divide(hpt_eta_tkAllCPUmatched, hpt_eta_tkAllCPU, 1., 1., "B");
  hphi_z_matchRatio->divide(hphi_z_tkAllCPUmatched, hphi_z_tkAllCPU, 1., 1., "B");
}

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
void SiPixelPhase1TrackComparisonHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("topFolderName", "SiPixelHeterogeneous/PixelTrackCompareGPUvsCPU/");
  descriptions.add("siPixelPhase1TrackComparisonHarvester", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelPhase1TrackComparisonHarvester);
