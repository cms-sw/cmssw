#include "CondFormats/DataRecord/interface/TotemAnalysisMaskRcd.h"
#include "CondFormats/DataRecord/interface/TotemReadoutRcd.h"
#include "CondFormats/PPSObjects/interface/TotemAnalysisMask.h"
#include "CondFormats/PPSObjects/interface/TotemDAQMapping.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <fstream>
#include <iostream>

//----------------------------------------------------------------------------------------------------

/**
 *\brief Writes to file the DAQ mapping loaded by TotemDAQMappingESSourceXML.
 **/
class WriteTotemDAQMapping : public edm::one::EDAnalyzer<> {
public:
  WriteTotemDAQMapping(const edm::ParameterSet &ps);
  ~WriteTotemDAQMapping() override = default;

private:
  /// label of the CTPPS sub-system
  const std::string subSystemName_;
  std::ofstream outStream_;
  const bool readMap_;
  const bool readMask_;
  edm::ESGetToken<TotemDAQMapping, TotemReadoutRcd> mappingToken_;
  edm::ESGetToken<TotemAnalysisMask, TotemAnalysisMaskRcd> maskToken_;
  void analyze(const edm::Event &e, const edm::EventSetup &es) override;
};

WriteTotemDAQMapping::WriteTotemDAQMapping(const edm::ParameterSet &ps)
    : subSystemName_(ps.getUntrackedParameter<std::string>("subSystem")),
      outStream_(ps.getUntrackedParameter<std::string>("fileName"), std::ios_base::app),
      readMap_(ps.getUntrackedParameter<bool>("readMap")),
      readMask_(ps.getUntrackedParameter<bool>("readMask")) {
  if (readMap_ == true) {
    mappingToken_ = esConsumes(edm::ESInputTag("", subSystemName_));
  }
  if (readMask_ == true) {
    maskToken_ = esConsumes(edm::ESInputTag("", subSystemName_));
  }
}

//----------------------------------------------------------------------------------------------------

void WriteTotemDAQMapping::analyze(const edm::Event &, edm::EventSetup const &es) {
  // get mapping
  if (readMap_ == true) {
    auto mappingHandle = es.getHandle(mappingToken_);
    if (mappingHandle.isValid() && !mappingHandle.failedToGet()) {
      auto const &mapping = *mappingHandle;
      mapping.print(outStream_, subSystemName_);
    } else {
      edm::LogError("WriteTotemDAQMapping mapping") << "WriteTotemDAQMapping: No mapping found";
    }
  }

  // get analysis mask to mask channels
  if (readMask_ == true) {
    auto analysisMaskHandle = es.getHandle(maskToken_);
    if (analysisMaskHandle.isValid() && !analysisMaskHandle.failedToGet()) {
      auto const &analysisMask = *analysisMaskHandle;
      outStream_ << analysisMask;
    } else {
      edm::LogError("WriteTotemDAQMapping mask") << "WriteTotemDAQMapping: No analysis mask found";
    }
  }

  outStream_.close();
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(WriteTotemDAQMapping);
