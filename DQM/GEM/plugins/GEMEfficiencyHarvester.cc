#include "DQM/GEM/plugins/GEMEfficiencyHarvester.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

GEMEfficiencyHarvester::GEMEfficiencyHarvester(const edm::ParameterSet& ps)
    : GEMDQMEfficiencyClientBase(ps), kFolders_(ps.getUntrackedParameter<std::vector<std::string> >("folders")) {}

GEMEfficiencyHarvester::~GEMEfficiencyHarvester() {}

void GEMEfficiencyHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // GEMDQMEfficiencyClientBase
  desc.addUntracked<double>("confidenceLevel", 0.683);  // 1-sigma
  desc.addUntracked<std::string>("logCategory", "GEMEfficiencyHarvester");

  // GEMEfficiencyHarvester
  desc.addUntracked<std::vector<std::string> >("folders", {"GEM/Efficiency/muonSTA"});
  descriptions.add("gemEfficiencyHarvester", desc);
}

// boook MEs for
//   - efficiency vs camber id
//   - efficiency vs ieta
// by projecting MEs used for 2d chamber-ieta efficiency.
void GEMEfficiencyHarvester::bookDetector1DEfficiency(DQMStore::IBooker& ibooker,
                                                      DQMStore::IGetter& igetter,
                                                      const std::string& folder) {
  const std::map<std::string, MEPair> me_pairs = makeEfficiencySourcePair(ibooker, igetter, folder, "chamber_ieta_");

  for (const auto& [key, value] : me_pairs) {
    // numerator and denominator
    const auto& [me_num, me_den] = value;

    if (me_num->kind() != MonitorElement::Kind::TH2F) {
      edm::LogError(kLogCategory_) << key << "expected TH2F but got ";  // TODO
      continue;
    }

    const TH2F* h_num = me_num->getTH2F();
    if (h_num == nullptr) {
      edm::LogError(kLogCategory_) << "numerator: failed to get TH2F from MonitorElement" << key;
      continue;
    }

    const TH2F* h_den = me_den->getTH2F();
    if (h_den == nullptr) {
      edm::LogError(kLogCategory_) << "denominator: failed to get TH2F from MonitorElement" << key;
      continue;
    }

    const auto [parsing_success, var_name, gem_name, is_matched] = parseEfficiencySourceName(me_den->getName());
    if (not parsing_success) {
      edm::LogError(kLogCategory_) << "failed to parse " << me_den->getName();
      continue;
    }

    // TODO sanity-check
    const TH1F* h_chamber_num = projectHistogram(h_num, TH1::kXaxis);
    const TH1F* h_chamber_den = projectHistogram(h_den, TH1::kXaxis);
    const char* eff_chamber_name = Form("eff_chamber_%s", gem_name.c_str());
    if (TH1F* eff = makeEfficiency(h_chamber_num, h_chamber_den, eff_chamber_name)) {
      ibooker.book1D(eff_chamber_name, eff);
    }

    const TH1F* h_ieta_num = projectHistogram(h_num, TH1::kYaxis);
    const TH1F* h_ieta_den = projectHistogram(h_den, TH1::kYaxis);
    const char* eff_ieta_name = Form("eff_ieta_%s", gem_name.c_str());
    if (TH1F* eff = makeEfficiency(h_ieta_num, h_ieta_den, eff_ieta_name)) {
      ibooker.book1D(eff_ieta_name, eff);
    }
  }  // pairs
}

void GEMEfficiencyHarvester::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  for (const std::string& folder : kFolders_) {
    bookEfficiencyAuto(ibooker, igetter, folder);
    bookDetector1DEfficiency(ibooker, igetter, folder);
  }
}
