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

#include <vector>
#include <ext/hash_map>

class SiPixelBadModuleByHandBuilder : public ConditionDBWriter<SiPixelQuality> {

public:

  explicit SiPixelBadModuleByHandBuilder(const edm::ParameterSet&);
  ~SiPixelBadModuleByHandBuilder();


private:

  std::unique_ptr<SiPixelQuality> getNewObject();

private:
  bool printdebug_;
  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters BadModuleList_;


};

#endif
