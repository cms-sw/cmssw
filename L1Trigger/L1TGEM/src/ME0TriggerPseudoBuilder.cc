#include "L1Trigger/L1TGEM/interface/ME0TriggerPseudoBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

#include <iostream>
#include <cassert>

const unsigned int ME0TriggerPseudoBuilder::ME0KeyLayer = 3;
const int ME0TriggerPseudoBuilder::ME0TriggerCentralBX = 8;

ME0TriggerPseudoBuilder::ME0TriggerPseudoBuilder(const edm::ParameterSet& conf) {
  config_ = conf;
  info_ = config_.getUntrackedParameter<int>("info", 0);
  dphiresolution_ = config_.getUntrackedParameter<double>("DeltaPhiResolution", 0.25);
}

ME0TriggerPseudoBuilder::~ME0TriggerPseudoBuilder() {}

void ME0TriggerPseudoBuilder::build(const ME0SegmentCollection* me0Segments, ME0TriggerDigiCollection& oc_trig) {
  if (info_ > 2)
    dumpAllME0Segments(*me0Segments);

  for (int endc = 0; endc < static_cast<int>(trig_me0s::MAX_ENDCAPS); endc++) {
    for (int cham = 0; cham < static_cast<int>(trig_me0s::MAX_CHAMBERS); cham++) {
      // 0th layer means whole chamber.
      // chamber counts from 1 to 18 in ME0ID
      const int region(endc == 0 ? -1 : 1);
      ME0DetId detid(region, 0, cham + 1, 0);

      const auto& drange = me0Segments->get(detid);
      std::vector<ME0TriggerDigi> trigV;
      for (auto digiIt = drange.first; digiIt != drange.second; digiIt++) {
        if (info_ > 1)
          LogTrace("L1ME0Trigger") << "ME0TriggerPseudoBuilder id " << detid << " ME0 segment " << *digiIt
                                   << " to be converted into trigger digi\n";
        ME0TriggerDigi trig = segmentConversion(*digiIt);
        if (trig.isValid())
          trigV.push_back(trig);
        if (info_ > 1 and trig.isValid())
          LogTrace("L1ME0Trigger") << " ME0trigger " << trig << "\n";
        else if (info_ > 1)
          LogTrace("L1ME0Trigger") << " ME0trigger is not valid. Conversion failed \n";
      }

      if (!trigV.empty()) {
        LogTrace("L1ME0Trigger") << "ME0TriggerPseudoBuilder got results in " << detid << std::endl
                                 << "Put " << trigV.size() << " Trigger digi" << ((trigV.size() > 1) ? "s " : " ")
                                 << "in collection\n";
        oc_trig.put(std::make_pair(trigV.begin(), trigV.end()), detid);
      }
    }
  }
}

ME0TriggerDigi ME0TriggerPseudoBuilder::segmentConversion(const ME0Segment segment) {
  auto detid = segment.me0DetId();
  const ME0Chamber* chamber = me0_g->chamber(detid);
  const ME0Layer* keylayer = me0_g->chamber(detid)->layer(ME0TriggerPseudoBuilder::ME0KeyLayer);
  int chamberid = detid.chamber() % 2;
  int totRolls = keylayer->nEtaPartitions();
  float dphi = chamber->computeDeltaPhi(segment.localPosition(), segment.localDirection());
  float time = segment.time();
  int sign_time = (time > 0) ? 1 : -1;
  int nrechits = segment.nRecHits();
  std::vector<int> rolls;
  for (auto rechit : segment.specificRecHits()) {
    if (std::find(rolls.begin(), rolls.end(), rechit.me0Id().roll()) == rolls.end())
      rolls.push_back(rechit.me0Id().roll());
  }
  if (rolls.size() > 2 or rolls.empty())
    LogTrace("L1ME0Trigger") << " ME0 segment is crossing " << rolls.size() << " roll !!! \n";
  //assert(rolls.size() <= 2);   // we did found very few ME0 segments crossing 3 rolls!!! this cut is applied offline
  if (rolls.empty())
    return ME0TriggerDigi();
  if (rolls[0] < 1)
    LogTrace("L1ME0Trigger") << " ME0 segment has wrong roll number " << rolls[0] << " which should be >= 1 \n !!!";
  assert(rolls[0] >= 1);
  int partition = (rolls[0] - 1) << 1;  //roll from detid counts from 1
  if (rolls.size() == 2 and rolls[0] > rolls[1])
    partition = partition - 1;
  else if (rolls.size() == 2 and rolls[0] < rolls[1])
    partition = partition + 1;

  if (partition < 0 or partition >= 2 * totRolls) {
    LogTrace("L1ME0Trigger") << " ME0 segment rolls size of all hits " << rolls.size() << " rolls[0] " << rolls[0]
                             << " rolls.back() " << rolls.back() << " and ME0 trigger roll is " << partition
                             << " max expected " << 2 * totRolls - 1 << "\n";
    return ME0TriggerDigi();
  }

  //globalpoint from ME0 segment
  GlobalPoint gp = me0_g->idToDet(segment.me0DetId())->surface().toGlobal(segment.localPosition());
  const ME0EtaPartition* etapart = keylayer->etaPartition(rolls[0]);
  LocalPoint segment_lp = etapart->surface().toLocal(gp);  // convert segment gp into lp in etapartition coordinate
  float strippitch = etapart->localPitch(segment_lp);
  float strip = etapart->strip(segment_lp);
  int totstrip = etapart->nstrips();
  int istrip = static_cast<int>(strip);
  int phiposition = istrip;
  if (phiposition > totstrip)
    LogTrace("L1ME0Trigger") << " ME0 segment strip number is " << phiposition << " larger than nstrip " << totstrip
                             << " !!! \n";
  float phi_resolution = 0.5;                                                         //halfstrip
  int phiposition2 = (static_cast<int>((strip - phiposition) / phi_resolution) & 1);  // half-strip resolution
  phiposition = (phiposition << 1) | phiposition2;

  //gloablpoint from ME0 trigger digi
  float centreOfStrip = istrip + 0.25 + phiposition2 * 0.5;
  GlobalPoint gp_digi = etapart->toGlobal(etapart->centreOfStrip(centreOfStrip));

  float strippitch_rad = strippitch / gp.perp();  //unit in rad

  int idphi = static_cast<int>(fabs(dphi) / (strippitch_rad * dphiresolution_));
  const int max_idphi = 512;
  if (idphi >= max_idphi) {
    LogTrace("L1ME0Trigger") << " ME0 segment dphi " << dphi << " and int type: " << idphi
                             << " larger than max allowed: " << max_idphi << " !!! \n";
    idphi = max_idphi - 1;
  }
  int quality = nrechits;  // attention: not the same as discussed in meeting
  int BX = (static_cast<int>(fabs(time) / 25.0)) * sign_time + ME0TriggerPseudoBuilder::ME0TriggerCentralBX;
  int bend = (dphi > 0.0) ? 0 : 1;
  if (info_ > 2)
    LogTrace("L1ME0Trigger") << " ME0trigger in conversion function:\n"
                             << "\t chamber(1-18) " << detid.chamber() << " chamber id " << chamberid << " \n"
                             << "\t rolls size of all hits " << rolls.size() << " rolls[0] " << rolls[0]
                             << " rolls.back() " << rolls.back() << " roll " << partition << " \n"
                             << "\t nRechits " << nrechits << " quality " << quality << " \n"
                             << "\t strip(float) " << strip << " (int) " << istrip << " phiposition " << phiposition
                             << " resolution (in term of strip) " << phi_resolution << " \n"
                             << "\t deltaphi(float) " << dphi << " (int) " << idphi << " resolution "
                             << strippitch_rad * dphiresolution_ << " bend " << bend << " \n"
                             << "\t global point eta " << gp.eta() << " phi " << gp.phi() << " trigger digi eta "
                             << gp_digi.eta() << " phi " << gp_digi.phi() << " \n"
                             << "\t time (ns, float) " << time << " BX " << BX << " \n";

  ME0TriggerDigi result = ME0TriggerDigi(chamberid, quality, phiposition, partition, idphi, bend, BX);
  result.setStrip(istrip);
  return result;
}

void ME0TriggerPseudoBuilder::dumpAllME0Segments(const ME0SegmentCollection& segments) const {
  LogTrace("L1ME0Trigger") << "dumpt all ME0 Segments" << std::endl;
  for (auto iC = segments.id_begin(); iC != segments.id_end(); ++iC) {
    auto ch_segs = segments.get(*iC);
    for (auto iS = ch_segs.first; iS != ch_segs.second; ++iS) {
      GlobalPoint gp = me0_g->idToDet(iS->me0DetId())->surface().toGlobal(iS->localPosition());
      LogTrace("L1ME0Trigger") << "ME0Detid " << iS->me0DetId() << " segment " << *iS << " eta " << gp.eta() << " phi "
                               << gp.phi() << std::endl;
      auto recHits(iS->recHits());
      LogTrace("L1ME0Trigger") << "\t has " << recHits.size() << " me0 rechits" << std::endl;
      for (auto& rh : recHits) {
        const ME0RecHit* me0rh(dynamic_cast<const ME0RecHit*>(rh));
        LogTrace("L1ME0Trigger") << "\t  detid " << me0rh->me0Id() << " rechit " << *me0rh << std::endl;
      }
    }
  }
}
