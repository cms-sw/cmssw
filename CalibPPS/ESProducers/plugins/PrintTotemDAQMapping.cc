/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors: 
 *  Jan Kašpar (jan.kaspar@gmail.com) 
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

//----------------------------------------------------------------------------------------------------

/**
 *\brief Prints the DAQ mapping loaded by TotemDAQMappingESSourceXML.
 **/
class PrintTotemDAQMapping : public edm::one::EDAnalyzer<> {
public:
  PrintTotemDAQMapping(const edm::ParameterSet &ps);
  ~PrintTotemDAQMapping() override {}

private:
  /// label of the CTPPS sub-system
  const std::string subSystemName_;
  const edm::ESGetToken<TotemDAQMapping, TotemReadoutRcd> mappingToken_;
  const edm::ESGetToken<TotemAnalysisMask, TotemAnalysisMaskRcd> maskToken_;
  void analyze(const edm::Event &e, const edm::EventSetup &es) override;
};

PrintTotemDAQMapping::PrintTotemDAQMapping(const edm::ParameterSet &ps)
    : subSystemName_(ps.getUntrackedParameter<std::string>("subSystem")),
      mappingToken_(esConsumes(edm::ESInputTag("", subSystemName_))),
      maskToken_(esConsumes(edm::ESInputTag("", subSystemName_))) {}

//----------------------------------------------------------------------------------------------------

void PrintTotemDAQMapping::analyze(const edm::Event &, edm::EventSetup const &es) {
  // get mapping
  if (auto mappingHandle = es.getHandle(mappingToken_)) {
    auto const &mapping = *mappingHandle;
    edm::LogInfo("PrintTotemDAQMapping mapping");
    mapping.print(std::cout, subSystemName_);
  } else {
    edm::LogError("PrintTotemDAQMapping mapping") << "PrintTotemDAQMapping: No mapping found";
  }

  // get analysis mask to mask channels
  if (auto analysisMaskHandle = es.getHandle(maskToken_)) {
    auto const &analysisMask = *analysisMaskHandle;
    edm::LogPrint("PrintTotemDAQMapping") << analysisMask;
  } else {
    edm::LogError("PrintTotemDAQMapping mask") << "PrintTotemDAQMapping: No analysis mask found";
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(PrintTotemDAQMapping);
