#include "L1Trigger/ME0TriggerPrimitives/src/ME0TriggerPrimitivesBuilder.h"
#include "L1Trigger/ME0TriggerPrimitives/src/ME0Motherboard.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//------------------
// Static variables
//------------------
const int ME0TriggerPrimitivesBuilder::min_endcap  = ME0DetId::minRegionId;
const int ME0TriggerPrimitivesBuilder::max_endcap  = ME0DetId::maxRegionId;
const int ME0TriggerPrimitivesBuilder::min_chamber = ME0DetId::minChamberId;
const int ME0TriggerPrimitivesBuilder::max_chamber = ME0DetId::maxChamberId;

//-------------
// Constructor
//-------------
ME0TriggerPrimitivesBuilder::ME0TriggerPrimitivesBuilder(const edm::ParameterSet& conf)
{
  for (int endc = min_endcap; endc <= max_endcap; endc++)
  {
    for (int cham = min_chamber; cham <= max_chamber; cham++)
    {
      if ((endc <= 0 || endc > MAX_ENDCAPS)    ||
	  (cham <= 0 || cham > MAX_CHAMBERS))
      {
	edm::LogError("L1ME0TPEmulatorSetupError")
	  << "+++ trying to instantiate TMB of illegal ME0 id ["
	  << " endcap = "  << endc 
	  << " chamber = " << cham 
	  << "]; skipping it... +++\n";
	continue;
      }
      tmb_[endc-1][cham-1].reset( new ME0Motherboard(endc, cham, conf) );
    }
  }
}

//------------
// Destructor
//------------
ME0TriggerPrimitivesBuilder::~ME0TriggerPrimitivesBuilder()
{
}

void ME0TriggerPrimitivesBuilder::build(const ME0PadDigiCollection* gemPads,
					ME0LCTDigiCollection& oc_lct)
{
  for (int endc = min_endcap; endc <= max_endcap; endc++)
  {
    for (int cham = min_chamber; cham <= max_chamber; cham++)
    {
      ME0Motherboard* tmb = tmb_[endc-1][cham-1].get();
      
      // 0th layer means whole chamber.
      ME0DetId detid(endc, 0, cham, 0);
      
      tmb->run(gemPads);
      
      std::vector<ME0LCTDigi> lctV = tmb->readoutLCTs();
      
      if (!lctV.empty()) {
	LogTrace("L1ME0Trigger")
	  << "ME0TriggerPrimitivesBuilder got results in " <<detid;
	
	LogTrace("L1ME0Trigger")
	  << "Put " << lctV.size() << " LCT digi"
	  << ((lctV.size() > 1) ? "s " : " ") << "in collection\n";
	oc_lct.put(std::make_pair(lctV.begin(),lctV.end()), detid);
      }
    } 
  }
}
