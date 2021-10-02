#ifndef SiPixelBadModuleByHandBuilder_H
#define SiPixelBadModuleByHandBuilder_H

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include <vector>

class SiPixelBadModuleByHandBuilder : public ConditionDBWriter<SiPixelQuality> {
public:
  explicit SiPixelBadModuleByHandBuilder(const edm::ParameterSet&);
  ~SiPixelBadModuleByHandBuilder() override;

private:
  std::unique_ptr<SiPixelQuality> getNewObject() override;

  void algoBeginRun(const edm::Run& run, const edm::EventSetup& es) override {
    if (!tTopo_) {
      tTopo_ = std::make_unique<TrackerTopology>(es.getData(tkTopoToken_));
    }
  };

private:
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tkTopoToken_;
  bool printdebug_;
  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters BadModuleList_;
  std::string ROCListFile_;
  std::unique_ptr<TrackerTopology> tTopo_;
};

#endif
