#ifndef SiPixelBadModuleByHandBuilder_H
#define SiPixelBadModuleByHandBuilder_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
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
  ~SiPixelBadModuleByHandBuilder();

private:
  std::unique_ptr<SiPixelQuality> getNewObject() override;
  void algoBeginJob(const edm::EventSetup& es) override {
    edm::ESHandle<TrackerTopology> htopo;
    es.get<TrackerTopologyRcd>().get(htopo);
    tTopo_ = htopo.product();
  };

private:
  bool printdebug_;
  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters BadModuleList_;
  std::string ROCListFile_;
  const TrackerTopology* tTopo_;
};

#endif
