#ifndef SiStripBadModuleByHandBuilder_H
#define SiStripBadModuleByHandBuilder_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <vector>

#include <ext/hash_map>

class SiStripBadModuleByHandBuilder : public ConditionDBWriter<SiStripBadStrip> {
public:
  explicit SiStripBadModuleByHandBuilder(const edm::ParameterSet&);
  ~SiStripBadModuleByHandBuilder() override;

private:
  std::unique_ptr<SiStripBadStrip> getNewObject() override;

private:
  edm::FileInPath fp_;
  bool printdebug_;
  std::vector<uint32_t> BadModuleList_;
  SiStripDetInfoFileReader* reader;
};
#endif
