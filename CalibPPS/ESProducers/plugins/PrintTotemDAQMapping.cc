/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors: 
 *  Jan Kašpar (jan.kaspar@gmail.com) 
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
  auto const &mapping = es.getData(mappingToken_);

  // get analysis mask to mask channels
  auto const &analysisMask = es.getData(maskToken_);

  // print mapping
  for (const auto &p : mapping.VFATMapping)
    edm::LogInfo("PrintTotemDAQMapping mapping") << "    " << p.first << " -> " << p.second;

  // print mapping
  for (const auto &p : analysisMask.analysisMask)
    edm::LogInfo("PrintTotemDAQMapping mask") << "    " << p.first << ": fullMask=" << p.second.fullMask
                                              << ", number of masked channels " << p.second.maskedChannels.size();
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(PrintTotemDAQMapping);
