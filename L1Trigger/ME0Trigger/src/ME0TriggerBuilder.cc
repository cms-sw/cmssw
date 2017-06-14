#include "L1Trigger/ME0Trigger/src/ME0TriggerBuilder.h"
#include "L1Trigger/ME0Trigger/src/ME0Motherboard.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

ME0TriggerBuilder::ME0TriggerBuilder(const edm::ParameterSet& conf)
{
  config_ = conf;

  for (int endc = 0; endc < 2; endc++)
  {
    for (int cham = 0; cham < 18; cham++)
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
      tmb_[endc][cham].reset( new ME0Motherboard(endc, cham, config_) );
    }
  }
}

ME0TriggerBuilder::~ME0TriggerBuilder()
{
}

void ME0TriggerBuilder::build(const ME0PadDigiClusterCollection* me0Clusters,
			      ME0TriggerDigiCollection& oc_trig)
{
  std::unique_ptr<ME0PadDigiCollection> me0Pads(new ME0PadDigiCollection());
  declusterize(me0Clusters, *me0Pads);
  build(me0Pads.get(), oc_trig);
}

void ME0TriggerBuilder::build(const ME0PadDigiCollection* me0Pads,
			      ME0TriggerDigiCollection& oc_trig)
{
  for (int endc = 0; endc < 2; endc++)
  {
    for (int cham = 0; cham < 18; cham++)
    {
      ME0Motherboard* tmb = tmb_[endc][cham].get();
      tmb->setME0Geometry(me0_g);

      // 0th layer means whole chamber.
      const int region(endc == 0 ? -1 : 1);
      ME0DetId detid(region, 0, cham, 0);

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

void
ME0TriggerBuilder::declusterize(const ME0PadDigiClusterCollection* in_clusters,
				ME0PadDigiCollection& out_pads)
{
  ME0PadDigiClusterCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt = in_clusters->begin();detUnitIt != in_clusters->end(); ++detUnitIt) {
    const ME0DetId& id = (*detUnitIt).first;
    const ME0PadDigiClusterCollection::Range& range = (*detUnitIt).second;
    for (ME0PadDigiClusterCollection::const_iterator digiIt = range.first; digiIt!=range.second; ++digiIt) {
      for (auto p: digiIt->pads()){
	out_pads.insertDigi(id, ME0PadDigi(p, digiIt->bx()));
      }
    }
  }
}
