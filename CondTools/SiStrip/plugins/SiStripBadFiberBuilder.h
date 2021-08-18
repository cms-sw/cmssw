#ifndef SiStripBadFiberBuilder_H
#define SiStripBadFiberBuilder_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <vector>

class SiStripBadFiberBuilder : public ConditionDBWriter<SiStripBadStrip> {
public:
  explicit SiStripBadFiberBuilder(const edm::ParameterSet&);
  ~SiStripBadFiberBuilder() override;

private:
  std::unique_ptr<SiStripBadStrip> getNewObject() override;

  bool printdebug_;

  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters BadComponentList_;
};
#endif
