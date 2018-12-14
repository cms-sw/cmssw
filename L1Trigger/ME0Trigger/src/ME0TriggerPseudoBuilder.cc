#include "L1Trigger/ME0Trigger/src/ME0TriggerPseudoBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"



const unsigned int ME0TriggerPseudoBuilder::ME0KeyLayer = 3;
const int ME0TriggerPseudoBuilder::ME0TriggerCentralBX = 8;

ME0TriggerPseudoBuilder::ME0TriggerPseudoBuilder(const edm::ParameterSet& conf)
{
  config_ = conf;
  info_ = config_.getUntrackedParameter<int>("info", 0);
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
            if (info_ > 1 ) LogTrace("L1ME0Trigger") << "ME0TriggerPseudoBuilder id "<< detid
                              <<" ME0 segment "<< *digiIt <<" to be converted\n";
            ME0TriggerDigi trig = segmentConversion(*digiIt);
            if (trig.isValid()) trigV.push_back(trig);
            if (info_ > 1 and trig.isValid()) LogTrace("L1ME0Trigger") <<" ME0trigger "<< trig <<"\n";
            else if (info_ > 1) LogTrace("L1ME0Trigger") <<" ME0trigger is not valid. Conversion failed \n";
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
  auto detid = segment.me0DetId();
  const ME0Chamber* chamber = me0_g->chamber(detid);
  const ME0Layer* keylayer = me0_g->chamber(detid)->layer(ME0TriggerPseudoBuilder::ME0KeyLayer);
  int chamberid = detid.chamber()%2;
  int totRolls = keylayer->nEtaPartitions();
  float dphi = chamber->computeDeltaPhi(segment.localPosition(), segment.localDirection());
  float time = segment.time();
  int sign_time = (time > 0) ? 1 : -1;
  int nrechits = segment.nRecHits();
  std::vector<int> rolls;
  for (auto rechit : segment.specificRecHits()){
      if (std::find(rolls.begin(), rolls.end(), rechit.me0Id().roll()) == rolls.end()) 
        rolls.push_back(rechit.me0Id().roll());
  }
  if (rolls.size() > 2 or rolls.size() == 0) LogTrace("L1ME0Trigger") << " ME0 segment is crossing "<< rolls.size() <<" roll !!! \n";
  if (rolls.size() == 0 ) return ME0TriggerDigi();
  int partition = (rolls[0] << 1) -1;//counting from 1
  if (rolls.size() == 2 and rolls[0] > rolls[1]) partition = partition+1;
  else if (rolls.size() == 2 and rolls[0] < rolls[1]) partition = partition-1;
  
  if (partition <= 0 or partition >= 2*totRolls){
     LogTrace("L1ME0Trigger") << " ME0 segment roll "<<  rolls[0] <<" and ME0 trigger roll is "
                              << partition-1 <<" max expected "<< 2*totRolls <<"\n";
     return ME0TriggerDigi();
  }
  
  const ME0EtaPartition* etapart = keylayer->etaPartition(rolls[0]);
  float strippitch = etapart->localPitch(segment.localPosition());
  float strip = etapart->strip(segment.localPosition());
  int totstrip = etapart->nstrips();
  int phiposition = static_cast<int>(strip);
  if (phiposition > totstrip)  LogTrace("L1ME0Trigger")<<" ME0 segment strip number is "<< phiposition <<" larger than nstrip "<< totstrip <<" !!! \n";
  int phiposition2 = (static_cast<int>((strip - phiposition)/(strippitch)) & 3);
  phiposition = (phiposition << 2) | phiposition2;
  int idphi = static_cast<int>(fabs(dphi)/(strippitch*dphiresolution_));
  int quality = nrechits;// attention: not the same as discussed in meeting
  int BX = (static_cast<int>(fabs(time)/25.0))*sign_time + ME0TriggerPseudoBuilder::ME0TriggerCentralBX;
  int bend = (dphi > 0.0) ? 0 : 1;
  
  return ME0TriggerDigi(chamberid, quality, phiposition, partition-1, idphi, bend, BX);
}
