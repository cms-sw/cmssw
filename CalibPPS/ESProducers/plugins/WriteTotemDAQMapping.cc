/****************************************************************************
 *
 ****************************************************************************/

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
  ~WriteTotemDAQMapping() override {}

private:
  /// label of the CTPPS sub-system
  std::string subSystemName;
  std::ofstream outStream;
  const edm::ESGetToken<TotemDAQMapping, TotemReadoutRcd> mappingToken;
  const edm::ESGetToken<TotemAnalysisMask, TotemAnalysisMaskRcd> maskToken;
  void analyze(const edm::Event &e, const edm::EventSetup &es) override;
};

WriteTotemDAQMapping::WriteTotemDAQMapping(const edm::ParameterSet &ps)
    : subSystemName(ps.getUntrackedParameter<std::string>("subSystem")),
      outStream(ps.getUntrackedParameter<std::string>("fileName"), std::ios_base::app),
      mappingToken(esConsumes(edm::ESInputTag("", subSystemName))),
      maskToken(esConsumes(edm::ESInputTag("", subSystemName))) {}

//----------------------------------------------------------------------------------------------------

void WriteTotemDAQMapping::analyze(const edm::Event &, edm::EventSetup const &es) {
  // get mapping
  if (auto mappingHandle = es.getHandle(mappingToken)) {
    auto const &mapping = *mappingHandle;
    outStream << mapping;
  } else {
    edm::LogError("WriteTotemDAQMapping mapping") << "No mapping found";
  }

  // get analysis mask to mask channels
  if (auto analysisMaskHandle = es.getHandle(maskToken)) {
    auto const &analysisMask = *analysisMaskHandle;
    outStream << analysisMask;
  } else {
    edm::LogError("WriteTotemDAQMapping mask") << "No analysis mask found";
  }

  outStream.close();
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(WriteTotemDAQMapping);
