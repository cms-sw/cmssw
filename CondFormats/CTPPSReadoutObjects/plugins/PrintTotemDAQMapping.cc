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
#include "CondFormats/CTPPSReadoutObjects/interface/TotemDAQMapping.h"
#include "CondFormats/CTPPSReadoutObjects/interface/TotemAnalysisMask.h"

//----------------------------------------------------------------------------------------------------

/**
 *\brief Prints the DAQ mapping loaded by TotemDAQMappingESSourceXML.
 **/
class PrintTotemDAQMapping : public edm::one::EDAnalyzer<>
{
  public:
    PrintTotemDAQMapping(const edm::ParameterSet &ps);
    ~PrintTotemDAQMapping() {}

  private:
    /// label of the CTPPS sub-system
    std::string subSystemName;

    virtual void analyze(const edm::Event &e, const edm::EventSetup &es) override;
};

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

PrintTotemDAQMapping::PrintTotemDAQMapping(const edm::ParameterSet &ps) :
  subSystemName(ps.getUntrackedParameter<string>("subSystem"))
{
}

//----------------------------------------------------------------------------------------------------

void PrintTotemDAQMapping::analyze(const edm::Event&, edm::EventSetup const& es)
{
  // get mapping
  ESHandle<TotemDAQMapping> mapping;
  es.get<TotemReadoutRcd>().get(subSystemName, mapping);

  // get analysis mask to mask channels
  ESHandle<TotemAnalysisMask> analysisMask;
  es.get<TotemReadoutRcd>().get(subSystemName, analysisMask);

  // print mapping
  printf("* DAQ mapping\n");
  for (const auto &p : mapping->VFATMapping)
    cout << "    " << p.first << " -> " << p.second << endl;

  // print mapping
  printf("* mask\n");
  for (const auto &p : analysisMask->analysisMask)
    cout << "    " << p.first
      << ": fullMask=" << p.second.fullMask
      << ", number of masked channels " << p.second.maskedChannels.size() << endl;
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(PrintTotemDAQMapping);
