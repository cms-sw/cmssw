/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors: 
 *  Jan Kašpar (jan.kaspar@gmail.com) 
 *  Leszek Grzanka (leszek.grzanka@cern.ch)
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/TotemReadoutRcd.h"
#include "CondFormats/PPSObjects/interface/TotemDAQMapping.h"
#include "CondFormats/PPSObjects/interface/TotemAnalysisMask.h"

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
  std::string subSystemName;
  edm::ESGetToken<TotemDAQMapping, TotemReadoutRcd> mappingToken_;
  edm::ESGetToken<TotemAnalysisMask, TotemReadoutRcd> maskToken_;
  void analyze(const edm::Event &e, const edm::EventSetup &es) override;
};

PrintTotemDAQMapping::PrintTotemDAQMapping(const edm::ParameterSet &ps)
    : subSystemName(ps.getUntrackedParameter<std::string>("subSystem")),
      mappingToken_(esConsumes(edm::ESInputTag("", subSystemName))),
      maskToken_(esConsumes(edm::ESInputTag("", subSystemName))) {}

//----------------------------------------------------------------------------------------------------

void PrintTotemDAQMapping::analyze(const edm::Event &, edm::EventSetup const &es) {

  // get mapping
  if (auto mappingHandle = es.getHandle(mappingToken_)) {
    auto const &mapping = *mappingHandle;
    // print mapping
    for (const auto &p : mapping.VFATMapping)
      edm::LogInfo("PrintTotemDAQMapping mapping") << "    " << p.first << " -> " << p.second;
  } else {
    edm::LogError("PrintTotemDAQMapping mapping") << "No mapping found";
  }

  // get analysis mask to mask channels
  if (auto analysisMaskHandle = es.getHandle(maskToken_)) {
    auto const &analysisMask = *analysisMaskHandle;
    // print mapping
    for (const auto &p : analysisMask.analysisMask)
      edm::LogInfo("PrintTotemDAQMapping mask") << "    " << p.first << ": fullMask=" << p.second.fullMask
                                                << ", number of masked channels " << p.second.maskedChannels.size();
  } else {
    edm::LogError("PrintTotemDAQMapping mapping") << "No analysis mask found";
  }

}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(PrintTotemDAQMapping);
