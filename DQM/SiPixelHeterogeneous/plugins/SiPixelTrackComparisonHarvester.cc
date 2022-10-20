#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// for string manipulations
#include <fmt/printf.h>

class SiPixelTrackComparisonHarvester : public DQMEDHarvester {
public:
  explicit SiPixelTrackComparisonHarvester(const edm::ParameterSet&);
  ~SiPixelTrackComparisonHarvester() override = default;
  void dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) override;
  void project2DalongDiagonal(MonitorElement* input2D, DQMStore::IBooker& ibooker);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // ----------member data ---------------------------
  const std::string topFolder_;
};

SiPixelTrackComparisonHarvester::SiPixelTrackComparisonHarvester(const edm::ParameterSet& iConfig)
    : topFolder_(iConfig.getParameter<std::string>("topFolderName")) {}

void SiPixelTrackComparisonHarvester::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  MonitorElement* hpt_eta_tkAllCPU = igetter.get(topFolder_ + "/ptetatrkAllCPU");
  MonitorElement* hpt_eta_tkAllCPUmatched = igetter.get(topFolder_ + "/ptetatrkAllCPUmatched");
  MonitorElement* hphi_z_tkAllCPU = igetter.get(topFolder_ + "/phiztrkAllCPU");
  MonitorElement* hphi_z_tkAllCPUmatched = igetter.get(topFolder_ + "/phiztrkAllCPUmatched");

  if (hpt_eta_tkAllCPU == nullptr or hpt_eta_tkAllCPUmatched == nullptr or hphi_z_tkAllCPU == nullptr or
      hphi_z_tkAllCPUmatched == nullptr) {
    edm::LogError("SiPixelTrackComparisonHarvester")
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

  // now create the 1D projection from the 2D histograms
  std::vector<std::string> listOfMEsToProject = {"nTracks",
                                                 "nLooseAndAboveTracks",
                                                 "nLooseAndAboveTracks_matched",
                                                 "nRecHits",
                                                 "nLayers",
                                                 "nChi2ndof",
                                                 "charge",
                                                 "pt",
                                                 "eta",
                                                 "phi",
                                                 "z",
                                                 "tip"};
  for (const auto& me : listOfMEsToProject) {
    MonitorElement* input2D = igetter.get(topFolder_ + "/" + me);
    this->project2DalongDiagonal(input2D, ibooker);
  }
}

void SiPixelTrackComparisonHarvester::project2DalongDiagonal(MonitorElement* input2D, DQMStore::IBooker& ibooker) {
  if (input2D == nullptr) {
    edm::LogError("SiPixelTrackComparisonHarvester")
        << "MEs needed for diagonal projection are not found in the input file. Skipping.";
    return;
  }

  ibooker.cd();
  ibooker.setCurrentFolder(topFolder_ + "/projectedDifferences");
  const auto& h_name = fmt::sprintf("%s_proj", input2D->getName());
  const auto& h_title = fmt::sprintf(";%s CPU -GPU difference", input2D->getTitle());
  const auto& span = (input2D->getAxisMax() - input2D->getAxisMin());
  const auto& b_w = span / input2D->getNbinsX();
  const auto& nbins = ((input2D->getNbinsX() % 2) == 0) ? input2D->getNbinsX() + 1 : input2D->getNbinsX();

  MonitorElement* diagonalized = ibooker.book1D(h_name, h_title, nbins, -span / 2., span / 2.);

  // collect all the entry on each diagonal of the 2D histogram
  for (int i = 1; i <= input2D->getNbinsX(); i++) {
    for (int j = 1; j <= input2D->getNbinsY(); j++) {
      diagonalized->Fill((i - j) * b_w, input2D->getBinContent(i, j));
    }
  }

  // zero the error on the bin as it's sort of meaningless for the way we fill it
  // by collecting the entry on each diagonal
  for (int bin = 1; bin <= diagonalized->getNbinsX(); bin++) {
    diagonalized->setBinError(bin, 0.f);
  }
}

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
void SiPixelTrackComparisonHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("topFolderName", "SiPixelHeterogeneous/PixelTrackCompareGPUvsCPU/");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelTrackComparisonHarvester);
