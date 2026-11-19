#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoTracker/MkFit/interface/MkFitGeometry.h"

// mkFit includes
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"

class MkFitIterationConfigESProducer : public edm::ESProducer {
public:
  MkFitIterationConfigESProducer(const edm::ParameterSet &iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  std::unique_ptr<mkfit::IterationConfig> produce(const TrackerRecoGeometryRecord &iRecord);

private:
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> geomToken_;
  const std::string configFile_;
  const float minPtCut_;
  const unsigned int maxClusterSize_;
  const int backwardSearchMinPixelLayers_;
  const float backwardSearchPromptMaxD0_;
  const float dcFracSharedHitsLowPt_;
  const float dcLowPtRampStart_;
  const float dcLowPtRampEnd_;
  const float dcLowPtMaxRelDiffInvPt_;
  const bool dcLowPtRequireSameCharge_;
  const float dcLowPtMaxD0_;
  const float dcLowPtMaxDRTimesPt_;
  const int dcMinUniqueHitsToKeep_;
  const float dcMinPtUniqueHitsToKeep_;
};

MkFitIterationConfigESProducer::MkFitIterationConfigESProducer(const edm::ParameterSet &iConfig)
    : geomToken_{setWhatProduced(this, iConfig.getParameter<std::string>("ComponentName")).consumes()},
      configFile_{iConfig.getParameter<edm::FileInPath>("config").fullPath()},
      minPtCut_{(float)iConfig.getParameter<double>("minPt")},
      maxClusterSize_{iConfig.getParameter<unsigned int>("maxClusterSize")},
      backwardSearchMinPixelLayers_{iConfig.getParameter<int>("backwardSearchMinPixelLayers")},
      backwardSearchPromptMaxD0_{(float)iConfig.getParameter<double>("backwardSearchPromptMaxD0")},
      dcFracSharedHitsLowPt_{(float)iConfig.getParameter<double>("duplicateCleaningLowPtFraction")},
      dcLowPtRampStart_{(float)iConfig.getParameter<double>("duplicateCleaningLowPtRampStart")},
      dcLowPtRampEnd_{(float)iConfig.getParameter<double>("duplicateCleaningLowPtRampEnd")},
      dcLowPtMaxRelDiffInvPt_{(float)iConfig.getParameter<double>("duplicateCleaningLowPtMaxRelDiffInvPt")},
      dcLowPtRequireSameCharge_{iConfig.getParameter<bool>("duplicateCleaningLowPtRequireSameCharge")},
      dcLowPtMaxD0_{(float)iConfig.getParameter<double>("duplicateCleaningLowPtMaxD0")},
      dcLowPtMaxDRTimesPt_{(float)iConfig.getParameter<double>("duplicateCleaningLowPtMaxDRTimesPt")},
      dcMinUniqueHitsToKeep_{iConfig.getParameter<int>("duplicateCleaningMinUniqueHitsToKeep")},
      dcMinPtUniqueHitsToKeep_{(float)iConfig.getParameter<double>("duplicateCleaningMinPtUniqueHitsToKeep")} {}

void MkFitIterationConfigESProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName", "")->setComment("Product label");
  desc.add<edm::FileInPath>("config", edm::FileInPath())
      ->setComment("Path to the JSON file for the mkFit configuration parameters");
  desc.add<double>("minPt", 0.0)->setComment("min pT cut applied during track building");
  desc.add<unsigned int>("maxClusterSize", 8)->setComment("Max cluster size of SiStrip hits");
  desc.add<int>("backwardSearchMinPixelLayers", 0)
      ->setComment("If > 0, keep a backward-search extension only if its new hits lie on this many pixel layers");
  desc.add<double>("backwardSearchPromptMaxD0", 0.0)
      ->setComment("Candidates with |d0| to the beam spot below this (cm) need new hits on only one pixel layer");
  // parameters of phase2:clean_duplicates_sharedhits_pixelpriority_ptadaptive (ignored by other cleaners)
  desc.add<double>("duplicateCleaningLowPtFraction", 0.10)
      ->setComment(
          "Shared-hit fraction for pairs whose harder track is below duplicateCleaningLowPtRampStart (< 0: off)");
  desc.add<double>("duplicateCleaningLowPtRampStart", 5.0)
      ->setComment("pT (GeV) of the harder track below which the low-pT fraction applies fully");
  desc.add<double>("duplicateCleaningLowPtRampEnd", 10.0)
      ->setComment(
          "pT (GeV) of the harder track above which the standard fraction applies (linear in 1/pT in between)");
  desc.add<double>("duplicateCleaningLowPtMaxRelDiffInvPt", 0.10)
      ->setComment("Low-pT fraction only if |1/pT1 - 1/pT2| <= this * max(1/pT1, 1/pT2) (<= 0: no condition)");
  desc.add<bool>("duplicateCleaningLowPtRequireSameCharge", true)
      ->setComment("Low-pT fraction only if the two tracks have the same charge");
  desc.add<double>("duplicateCleaningLowPtMaxD0", 2.0)
      ->setComment(
          "Low-pT fraction only if both tracks have |d0| to the beam spot below this, in cm (<= 0: no condition)");
  desc.add<double>("duplicateCleaningLowPtMaxDRTimesPt", 0.10)
      ->setComment(
          "Low-pT fraction only if dR(eta, phi) < this / pT of the harder track, i.e. a cone shrinking with pT, in GeV "
          "(<= 0: no condition)");
  desc.add<int>("duplicateCleaningMinUniqueHitsToKeep", 9)
      ->setComment(
          "Keep a would-be duplicate above duplicateCleaningMinPtUniqueHitsToKeep with at least this many found hits "
          "not shared with the other track (<= 0: off)");
  desc.add<double>("duplicateCleaningMinPtUniqueHitsToKeep", 10.0)
      ->setComment("Minimum pT (GeV) of a would-be duplicate for the unique-hit rule");
  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<mkfit::IterationConfig> MkFitIterationConfigESProducer::produce(
    const TrackerRecoGeometryRecord &iRecord) {
  mkfit::ConfigJson cj;
  auto it_conf = cj.load_File(configFile_);
  it_conf->m_params.minPtCut = minPtCut_;
  it_conf->m_backward_params.minPtCut = minPtCut_;
  it_conf->m_params.maxClusterSize = maxClusterSize_;
  it_conf->m_backward_params.maxClusterSize = maxClusterSize_;
  it_conf->m_backward_search_min_pixel_layers = backwardSearchMinPixelLayers_;
  it_conf->m_backward_search_prompt_max_d0 = backwardSearchPromptMaxD0_;
  it_conf->dc_fracSharedHitsLowPt = dcFracSharedHitsLowPt_;
  it_conf->dc_lowPtRampStart = dcLowPtRampStart_;
  it_conf->dc_lowPtRampEnd = dcLowPtRampEnd_;
  it_conf->dc_lowPtMaxRelDiffInvPt = dcLowPtMaxRelDiffInvPt_;
  it_conf->dc_lowPtRequireSameCharge = dcLowPtRequireSameCharge_;
  it_conf->dc_lowPtMaxD0 = dcLowPtMaxD0_;
  it_conf->dc_lowPtMaxDRTimesPt = dcLowPtMaxDRTimesPt_;
  it_conf->dc_minUniqueHitsToKeep = dcMinUniqueHitsToKeep_;
  it_conf->dc_minPtUniqueHitsToKeep = dcMinPtUniqueHitsToKeep_;
  it_conf->setupStandardFunctionsFromNames();
  return it_conf;
}

DEFINE_FWK_EVENTSETUP_MODULE(MkFitIterationConfigESProducer);
