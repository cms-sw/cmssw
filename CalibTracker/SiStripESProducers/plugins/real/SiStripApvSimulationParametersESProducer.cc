#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvSimulationParameters.h"
#include <fstream>
#include <boost/range/adaptor/indexed.hpp>

class SiStripApvSimulationParametersESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  explicit SiStripApvSimulationParametersESSource(const edm::ParameterSet& conf);
  ~SiStripApvSimulationParametersESSource() override {}

  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue& iov,
                      edm::ValidityInterval& iValidity) override;

  std::unique_ptr<SiStripApvSimulationParameters> produce(const SiStripApvSimulationParametersRcd& record);

private:
  std::vector<edm::FileInPath> baselineFiles_TOB_;
  std::vector<edm::FileInPath> baselineFiles_TIB_;
  std::vector<edm::FileInPath> baselineFiles_TID_;
  std::vector<edm::FileInPath> baselineFiles_TEC_;
  unsigned int baseline_nBins_;
  float baseline_min_;
  float baseline_max_;
  std::vector<float> puBinEdges_;
  std::vector<float> zBinEdges_;
  std::vector<float> rBinEdgesTID_;
  std::vector<float> rBinEdgesTEC_;

  SiStripApvSimulationParameters::LayerParameters makeLayerParameters(const std::string& apvBaselinesFileName,
                                                                      const std::vector<float>& rzBinEdges) const;
};

SiStripApvSimulationParametersESSource::SiStripApvSimulationParametersESSource(const edm::ParameterSet& conf)
    : baseline_nBins_(conf.getUntrackedParameter<unsigned int>("apvBaselines_nBinsPerBaseline")),
      baseline_min_(conf.getUntrackedParameter<double>("apvBaselines_minBaseline")),
      baseline_max_(conf.getUntrackedParameter<double>("apvBaselines_maxBaseline")) {
  setWhatProduced(this);
  findingRecord<SiStripApvSimulationParametersRcd>();
  for (const auto x : conf.getUntrackedParameter<std::vector<double>>("apvBaselines_puBinEdges")) {
    puBinEdges_.push_back(x);
  }
  for (const auto x : conf.getUntrackedParameter<std::vector<double>>("apvBaselines_zBinEdges")) {
    zBinEdges_.push_back(x);
  }
  for (const auto x : conf.getUntrackedParameter<std::vector<double>>("apvBaselines_rBinEdges_TID")) {
    rBinEdgesTID_.push_back(x);
  }
  for (const auto x : conf.getUntrackedParameter<std::vector<double>>("apvBaselines_rBinEdges_TEC")) {
    rBinEdgesTEC_.push_back(x);
  }
  baselineFiles_TIB_ = {conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tib1"),
                        conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tib2"),
                        conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tib3"),
                        conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tib4")};
  baselineFiles_TOB_ = {conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tob1"),
                        conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tob2"),
                        conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tob3"),
                        conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tob4"),
                        conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tob5"),
                        conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tob6")};

  if (!rBinEdgesTID_.empty()) {
    baselineFiles_TID_ = {conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tid1"),
                          conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tid2"),
                          conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tid3")};
  }

  if (!rBinEdgesTEC_.empty()) {
    baselineFiles_TEC_ = {conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tec1"),
                          conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tec2"),
                          conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tec3"),
                          conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tec4"),
                          conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tec5"),
                          conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tec6"),
                          conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tec7"),
                          conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tec8"),
                          conf.getUntrackedParameter<edm::FileInPath>("apvBaselinesFile_tec9")};
  }
}

void SiStripApvSimulationParametersESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                                            const edm::IOVSyncValue& iov,
                                                            edm::ValidityInterval& iValidity) {
  iValidity = edm::ValidityInterval{iov.beginOfTime(), iov.endOfTime()};
}

SiStripApvSimulationParameters::LayerParameters SiStripApvSimulationParametersESSource::makeLayerParameters(
    const std::string& apvBaselinesFileName, const std::vector<float>& rzBinEdges) const {
  // Prepare histograms
  unsigned int nZBins = rzBinEdges.size();
  unsigned int nPUBins = puBinEdges_.size();

  if (nPUBins == 0 || nZBins == 0 || baseline_nBins_ == 0) {
    throw cms::Exception("MissingInput") << "The parameters for the APV simulation are not correctly configured\n";
  }
  std::vector<float> baselineBinEdges{};
  const auto baseline_binWidth = (baseline_max_ - baseline_min_) / baseline_nBins_;
  for (unsigned i{0}; i != baseline_nBins_; ++i) {
    baselineBinEdges.push_back(baseline_min_ + i * baseline_binWidth);
  }
  baselineBinEdges.push_back(baseline_max_);

  SiStripApvSimulationParameters::LayerParameters layerParams{baselineBinEdges, puBinEdges_, rzBinEdges};

  // Read apv baselines from text files
  std::vector<double> theAPVBaselines;
  std::ifstream apvBaselineFile(apvBaselinesFileName.c_str());
  if (!apvBaselineFile.good()) {
    throw cms::Exception("FileError") << "Problem opening APV baselines file: " << apvBaselinesFileName;
  }
  std::string line;
  while (std::getline(apvBaselineFile, line)) {
    if (!line.empty()) {
      std::istringstream lStr{line};
      double value;
      while (lStr >> value) {
        theAPVBaselines.push_back(value);
      }
    }
  }
  if (theAPVBaselines.empty()) {
    throw cms::Exception("WrongAPVBaselines")
        << "Problem reading from APV baselines file " << apvBaselinesFileName << ": no values read in";
  }

  if (theAPVBaselines.size() != nZBins * nPUBins * baseline_nBins_) {
    throw cms::Exception("WrongAPVBaselines") << "Problem reading from APV baselines file " << apvBaselinesFileName
                                              << ": number of baselines read different to that expected i.e. nZBins * "
                                                 "nPUBins * apvBaselines_nBinsPerBaseline_";
  }

  // Put baselines into histograms
  for (auto const& apvBaseline : theAPVBaselines | boost::adaptors::indexed(0)) {
    unsigned int binInCurrentHistogram = apvBaseline.index() % baseline_nBins_ + 1;
    unsigned int binInZ = int(apvBaseline.index()) / (nPUBins * baseline_nBins_);
    unsigned int binInPU = int(apvBaseline.index() - binInZ * (nPUBins)*baseline_nBins_) / baseline_nBins_;

    layerParams.setBinContent(binInCurrentHistogram, binInPU + 1, binInZ + 1, apvBaseline.value());
  }

  return layerParams;
}

std::unique_ptr<SiStripApvSimulationParameters> SiStripApvSimulationParametersESSource::produce(
    const SiStripApvSimulationParametersRcd& record) {
  auto apvSimParams = std::make_unique<SiStripApvSimulationParameters>(
      baselineFiles_TIB_.size(), baselineFiles_TOB_.size(), baselineFiles_TID_.size(), baselineFiles_TEC_.size());
  for (unsigned int i{0}; i != baselineFiles_TIB_.size(); ++i) {
    if (!apvSimParams->putTIB(i + 1, makeLayerParameters(baselineFiles_TIB_[i].fullPath(), zBinEdges_))) {
      throw cms::Exception("SiStripApvSimulationParameters") << "Could not add parameters for TIB layer " << (i + 1);
    } else {
      LogDebug("SiStripApvSimulationParameters") << "Added parameters for TIB layer " << (i + 1);
    }
  }
  for (unsigned int i{0}; i != baselineFiles_TOB_.size(); ++i) {
    if (!apvSimParams->putTOB(i + 1, makeLayerParameters(baselineFiles_TOB_[i].fullPath(), zBinEdges_))) {
      throw cms::Exception("SiStripApvSimulationParameters") << "Could not add parameters for TOB layer " << (i + 1);
    } else {
      LogDebug("SiStripApvSimulationParameters") << "Added parameters for TOB layer " << (i + 1);
    }
  }
  for (unsigned int i{0}; i != baselineFiles_TID_.size(); ++i) {
    if (!apvSimParams->putTID(i + 1, makeLayerParameters(baselineFiles_TID_[i].fullPath(), rBinEdgesTID_))) {
      throw cms::Exception("SiStripApvSimulationParameters") << "Could not add parameters for TID wheel " << (i + 1);
    } else {
      LogDebug("SiStripApvSimulationParameters") << "Added parameters for TID wheel " << (i + 1);
    }
  }
  for (unsigned int i{0}; i != baselineFiles_TEC_.size(); ++i) {
    if (!apvSimParams->putTEC(i + 1, makeLayerParameters(baselineFiles_TEC_[i].fullPath(), rBinEdgesTEC_))) {
      throw cms::Exception("SiStripApvSimulationParameters") << "Could not add parameters for TEC wheel " << (i + 1);
    } else {
      LogDebug("SiStripApvSimulationParameters") << "Added parameters for TEC wheel " << (i + 1);
    }
  }
  return apvSimParams;
}

#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripApvSimulationParametersESSource);
