#include "L1Trigger/ME0Trigger/src/ME0TriggerBuilder.h"
#include "L1Trigger/ME0Trigger/src/ME0Motherboard.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

const int ME0TriggerBuilder::min_endcap  = ME0DetId::minRegionId;
const int ME0TriggerBuilder::max_endcap  = ME0DetId::maxRegionId;
const int ME0TriggerBuilder::min_chamber = ME0DetId::minChamberId;
const int ME0TriggerBuilder::max_chamber = ME0DetId::maxChamberId;

ME0TriggerBuilder::ME0TriggerBuilder(const edm::ParameterSet& conf)
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

ME0TriggerBuilder::~ME0TriggerBuilder()
{
}

void ME0TriggerBuilder::build(const ME0PadDigiCollection* me0Pads,
			      ME0TriggerDigiCollection& oc_trig)
{
  for (int endc = min_endcap; endc <= max_endcap; endc++)
  {
    for (int cham = min_chamber; cham <= max_chamber; cham++)
    {
      ME0Motherboard* tmb = tmb_[endc-1][cham-1].get();
      tmb->setME0Geometry(me0_g);

      // 0th layer means whole chamber.
      ME0DetId detid(endc, 0, cham, 0);

      // Run processors only if chamber exists in geometry.
      if (tmb == 0 || me0_g->chamber(detid) == 0) continue;
      
      tmb->run(me0Pads);
      
      std::vector<ME0TriggerDigi> trigV = tmb->readoutTriggers();
      
      if (!trigV.empty()) {
	LogTrace("L1ME0Trigger")
	  << "ME0TriggerBuilder got results in " <<detid
	  << std::endl 
	  << "Put " << trigV.size() << " Trigger digi"
	  << ((trigV.size() > 1) ? "s " : " ") << "in collection\n";
	oc_trig.put(std::make_pair(trigV.begin(),trigV.end()), detid);
      }
    } 
  }
}
