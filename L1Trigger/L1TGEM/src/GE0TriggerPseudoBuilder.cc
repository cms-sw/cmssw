#include "L1Trigger/L1TGEM/interface/GE0TriggerPseudoBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include <iostream>
#include <cassert>

const unsigned int GE0TriggerPseudoBuilder::ME0KeyLayer = 3;
const int GE0TriggerPseudoBuilder::ME0TriggerCentralBX = 8;

GE0TriggerPseudoBuilder::GE0TriggerPseudoBuilder(const edm::ParameterSet& conf) {
  config_ = conf;
  info_ = config_.getUntrackedParameter<int>("info", 0);
  dphiresolution_ = config_.getUntrackedParameter<double>("DeltaPhiResolution", 0.25);
}

GE0TriggerPseudoBuilder::~GE0TriggerPseudoBuilder() {}

void GE0TriggerPseudoBuilder::build(const GEMSegmentCollection& me0Segments, GE0TriggerDigiCollection& oc_trig) {
  if (info_ > 2)
    dumpAllME0Segments(me0Segments);

  for (unsigned int endc = 0; endc < static_cast<unsigned int>(trig_me0s::MAX_ENDCAPS); endc++) {
    for (unsigned int cham = 0; cham < static_cast<unsigned int>(trig_me0s::MAX_CHAMBERS); cham++) {
      // 0th layer means whole chamber.
      // chamber counts from 1 to 18 in ME0ID
      const int region(endc == 0 ? -1 : 1);
      //  constexpr GEMDetId(int region, int ring, int station, int layer, int chamber, int ieta)
      GEMDetId detid(region, 1, 0, 0, cham + 1, 0);

      const auto& drange = me0Segments.get(detid);
      std::vector<ME0TriggerDigi> trigV;
      for (auto digiIt = drange.first; digiIt != drange.second; digiIt++) {
        if (info_ > 1)
          LogTrace("L1ME0Trigger") << "GE0TriggerPseudoBuilder id " << detid << " ME0 segment " << *digiIt
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
        LogTrace("L1ME0Trigger") << "GE0TriggerPseudoBuilder got results in " << detid << std::endl
                                 << "Put " << trigV.size() << " Trigger digi" << ((trigV.size() > 1) ? "s " : " ")
                                 << "in collection\n";
        oc_trig.put(std::make_pair(trigV.begin(), trigV.end()), detid);
      }
    }
  }
}

ME0TriggerDigi GE0TriggerPseudoBuilder::segmentConversion(const GEMSegment segment) {
  auto detid = segment.gemDetId();
  const GEMSuperChamber* chamber = me0_g->superChamber(detid);
  const GEMChamber* keylayer = chamber->chamber(GE0TriggerPseudoBuilder::ME0KeyLayer);
  int chamberid = detid.superChamberId() % 2;
  int totRolls = keylayer->nEtaPartitions();
  float dphi = chamber->computeDeltaPhi(segment.localPosition(), segment.localDirection());

  int nrechits = segment.nRecHits();
  std::vector<int> rolls;
  for (const auto& rechit : segment.specificRecHits()) {
    if (std::find(rolls.begin(), rolls.end(), rechit.gemId().roll()) == rolls.end())
      rolls.push_back(rechit.gemId().roll());
  }
  if (rolls.size() > 2 or rolls.empty())
    LogTrace("L1ME0Trigger") << " ME0 segment is crossing " << rolls.size() << " roll !!! \n";
  //assert(rolls.size() <= 2);   // we did found very few ME0 segments crossing 3 rolls!!! this cut is applied offline
  if (rolls.empty())
    return ME0TriggerDigi();
  if (rolls[0] < 1) {
    LogTrace("L1ME0Trigger") << " ME0 segment has wrong roll number " << rolls[0] << " which should be >= 1 \n !!!";
    throw edm::Exception(edm::errors::LogicError, "ME0 should have at least one roll");
  }
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
  GlobalPoint gp = me0_g->idToDet(segment.gemDetId())->surface().toGlobal(segment.localPosition());
  const GEMEtaPartition* etapart = keylayer->etaPartition(rolls[0]);
  LocalPoint segment_lp = etapart->surface().toLocal(gp);  // convert segment gp into lp in etapartition coordinate
  float strippitch = etapart->localPitch(segment_lp);
  float strip = etapart->strip(segment_lp);
  int totstrip = etapart->nstrips();
  int istrip = static_cast<int>(strip);
  int phiposition = istrip;
  if (phiposition > totstrip)
    LogTrace("L1ME0Trigger") << " ME0 segment strip number is " << phiposition << " larger than nstrip " << totstrip
                             << " !!! \n";
  constexpr float phi_resolution = 0.5;                                               //halfstrip
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
  // !!!!TODO!!!! int BX = (static_cast<int>(fabs(time) / 25.0)) * sign_time + GE0TriggerPseudoBuilder::ME0TriggerCentralBX;
  int BX = GE0TriggerPseudoBuilder::ME0TriggerCentralBX;
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
        // << "\t time (ns, float) " << time << " BX " << BX << " \n"
        ;

  ME0TriggerDigi result = ME0TriggerDigi(chamberid, quality, phiposition, partition, idphi, bend, BX);
  result.setStrip(istrip);
  return result;

  return ME0TriggerDigi();
}

void GE0TriggerPseudoBuilder::dumpAllME0Segments(const GEMSegmentCollection& segments) const {
  LogTrace("L1GE0Trigger") << "dumpt all ME0 Segments" << std::endl;
  for (auto iC = segments.id_begin(); iC != segments.id_end(); ++iC) {
    auto ch_segs = segments.get(*iC);
    for (auto iS = ch_segs.first; iS != ch_segs.second; ++iS) {
      if (iS->gemDetId().station() != 0)  // only dump GE0 segments
        continue;
      GlobalPoint gp = me0_g->idToDet(iS->gemDetId())->surface().toGlobal(iS->localPosition());
      LogTrace("L1ME0Trigger") << "ME0Detid " << iS->gemDetId() << " segment " << *iS << " eta " << gp.eta() << " phi "
                               << gp.phi() << std::endl;
      auto recHits(iS->recHits());
      LogTrace("L1GE0Trigger") << "\t has " << recHits.size() << " me0 rechits" << std::endl;
      for (auto& rh : recHits) {
        const GEMRecHit* me0rh(dynamic_cast<const GEMRecHit*>(rh));
        LogTrace("L1GEMTrigger") << "\t  detid " << me0rh->gemId() << " rechit " << *me0rh << std::endl;
      }
    }
  }
}
