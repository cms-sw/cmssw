#include "L1Trigger/ME0Trigger/src/ME0TriggerPseudoBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

ME0TriggerPseudoBuilder::ME0TriggerPseudoBuilder(const edm::ParameterSet& conf)
{
  config_ = conf;
  info_ = config_.getUntrackedParameter<int>("info", 0);
  phiresolution_ = config_.getUntrackedParameter<double>("PhiResolution", 0.25);
  dphiresolution_ = config_.getUntrackedParameter<double>("DeltaPhiResolution", 0.25);

}

ME0TriggerPseudoBuilder::~ME0TriggerPseudoBuilder()
{
}

void ME0TriggerPseudoBuilder::build(const ME0SegmentCollection* me0Segments,
			      ME0TriggerDigiCollection& oc_trig)
{
  for (int endc = 0; endc < 2; endc++)
  {
    for (int cham = 0; cham < 18; cham++)
    {
      // 0th layer means whole chamber.
      const int region(endc == 0 ? -1 : 1);
      ME0DetId detid(region, 0, cham, 0);

      
      const auto& drange = me0Segments->get(detid);
      std::vector<ME0TriggerDigi> trigV;
      for (auto digiIt = drange.first; digiIt != drange.second; digiIt++) {
            ME0TriggerDigi trig = segmentConversion(*digiIt);
            if (trig.isValid()) trigV.push_back(trig);
      }
      
      if (!trigV.empty()) {
	LogTrace("L1ME0Trigger")
	  << "ME0TriggerPseudoBuilder got results in " <<detid
	  << std::endl 
	  << "Put " << trigV.size() << " Trigger digi"
	  << ((trigV.size() > 1) ? "s " : " ") << "in collection\n";
	oc_trig.put(std::make_pair(trigV.begin(),trigV.end()), detid);
      }
    } 
  }
}


ME0TriggerDigi ME0TriggerPseudoBuilder::segmentConversion(const ME0Segment segment){
  return ME0TriggerDigi();
}
