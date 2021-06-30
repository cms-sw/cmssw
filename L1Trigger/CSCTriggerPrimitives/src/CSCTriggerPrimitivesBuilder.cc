#include <memory>

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCTriggerPrimitivesBuilder.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCMotherboard.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCGEMMotherboardME11.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCGEMMotherboardME21.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCMuonPortCard.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

const int CSCTriggerPrimitivesBuilder::min_endcap = CSCDetId::minEndcapId();
const int CSCTriggerPrimitivesBuilder::max_endcap = CSCDetId::maxEndcapId();
const int CSCTriggerPrimitivesBuilder::min_station = CSCDetId::minStationId();
const int CSCTriggerPrimitivesBuilder::max_station = CSCDetId::maxStationId();
const int CSCTriggerPrimitivesBuilder::min_sector = CSCTriggerNumbering::minTriggerSectorId();
const int CSCTriggerPrimitivesBuilder::max_sector = CSCTriggerNumbering::maxTriggerSectorId();
const int CSCTriggerPrimitivesBuilder::min_subsector = CSCTriggerNumbering::minTriggerSubSectorId();
const int CSCTriggerPrimitivesBuilder::max_subsector = CSCTriggerNumbering::maxTriggerSubSectorId();
const int CSCTriggerPrimitivesBuilder::min_chamber = CSCTriggerNumbering::minTriggerCscId();
const int CSCTriggerPrimitivesBuilder::max_chamber = CSCTriggerNumbering::maxTriggerCscId();

CSCTriggerPrimitivesBuilder::CSCTriggerPrimitivesBuilder(const edm::ParameterSet& conf) {
  // special configuration parameters for ME11 treatment
  edm::ParameterSet commonParams = conf.getParameter<edm::ParameterSet>("commonParam");
  runPhase2_ = commonParams.getParameter<bool>("runPhase2");
  infoV = commonParams.getParameter<int>("verbosity");
  disableME1a_ = commonParams.getParameter<bool>("disableME1a");
  disableME42_ = commonParams.getParameter<bool>("disableME42");

  checkBadChambers_ = conf.getParameter<bool>("checkBadChambers");

  runME11Up_ = commonParams.getParameter<bool>("runME11Up");
  runME21Up_ = commonParams.getParameter<bool>("runME21Up");
  runME31Up_ = commonParams.getParameter<bool>("runME31Up");
  runME41Up_ = commonParams.getParameter<bool>("runME41Up");

  runME11ILT_ = commonParams.getParameter<bool>("runME11ILT");
  runME21ILT_ = commonParams.getParameter<bool>("runME21ILT");

  // Initializing boards.
  for (int endc = min_endcap; endc <= max_endcap; endc++) {
    for (int stat = min_station; stat <= max_station; stat++) {
      int numsubs = ((stat == 1) ? max_subsector : 1);
      for (int sect = min_sector; sect <= max_sector; sect++) {
        for (int subs = min_subsector; subs <= numsubs; subs++) {
          for (int cham = min_chamber; cham <= max_chamber; cham++) {
            if ((endc <= 0 || endc > MAX_ENDCAPS) || (stat <= 0 || stat > MAX_STATIONS) ||
                (sect <= 0 || sect > MAX_SECTORS) || (subs <= 0 || subs > MAX_SUBSECTORS) ||
                (cham <= 0 || cham > MAX_CHAMBERS)) {
              edm::LogError("CSCTriggerPrimitivesBuilder|SetupError")
                  << "+++ trying to instantiate TMB of illegal CSC id ["
                  << " endcap = " << endc << " station = " << stat << " sector = " << sect << " subsector = " << subs
                  << " chamber = " << cham << "]; skipping it... +++\n";
              continue;
            }
            int ring = CSCTriggerNumbering::ringFromTriggerLabels(stat, cham);
            // When the motherboard is instantiated, it instantiates ALCT
            // and CLCT processors.

            // go through all possible cases
            if (runPhase2_ and ring == 1 and stat == 1 and runME11Up_ and runME11ILT_)
              tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1] =
                  std::make_unique<CSCGEMMotherboardME11>(endc, stat, sect, subs, cham, conf);
            else if (runPhase2_ and ring == 1 and stat == 2 and runME21Up_ and runME21ILT_)
              tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1] =
                  std::make_unique<CSCGEMMotherboardME21>(endc, stat, sect, subs, cham, conf);
            else if (runPhase2_ and ring == 1 and
                     ((stat == 2 and runME21Up_ and !runME21ILT_) || (stat == 3 and runME31Up_) ||
                      (stat == 4 and runME41Up_)))
              tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1] =
                  std::make_unique<CSCUpgradeMotherboard>(endc, stat, sect, subs, cham, conf);
            else
              tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1] =
                  std::make_unique<CSCMotherboard>(endc, stat, sect, subs, cham, conf);
          }
        }
        // Init MPC
        mpc_[endc - 1][stat - 1][sect - 1] = std::make_unique<CSCMuonPortCard>(endc, stat, sect, conf);
      }
    }
  }
}

//------------
// Destructor
//------------
CSCTriggerPrimitivesBuilder::~CSCTriggerPrimitivesBuilder() {}

void CSCTriggerPrimitivesBuilder::setConfigParameters(const CSCDBL1TPParameters* conf) {
  // Receives CSCDBL1TPParameters percolated down from ESProducer.

  for (int endc = min_endcap; endc <= max_endcap; endc++) {
    for (int stat = min_station; stat <= max_station; stat++) {
      int numsubs = ((stat == 1) ? max_subsector : 1);
      for (int sect = min_sector; sect <= max_sector; sect++) {
        for (int subs = min_subsector; subs <= numsubs; subs++) {
          for (int cham = min_chamber; cham <= max_chamber; cham++) {
            tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1]->setConfigParameters(conf);
          }
        }
      }
    }
  }
}

void CSCTriggerPrimitivesBuilder::build(const CSCBadChambers* badChambers,
                                        const CSCWireDigiCollection* wiredc,
                                        const CSCComparatorDigiCollection* compdc,
                                        const GEMPadDigiClusterCollection* gemClusters,
                                        CSCALCTDigiCollection& oc_alct,
                                        CSCCLCTDigiCollection& oc_clct,
                                        CSCALCTPreTriggerDigiCollection& oc_alctpretrigger,
                                        CSCCLCTPreTriggerDigiCollection& oc_pretrigger,
                                        CSCCLCTPreTriggerCollection& oc_pretrig,
                                        CSCCorrelatedLCTDigiCollection& oc_lct,
                                        CSCCorrelatedLCTDigiCollection& oc_sorted_lct,
                                        CSCShowerDigiCollection& oc_shower,
                                        CSCShowerDigiCollection& oc_shower_anode,
                                        GEMCoPadDigiCollection& oc_gemcopad) {
  // CSC geometry.
  for (int endc = min_endcap; endc <= max_endcap; endc++) {
    for (int stat = min_station; stat <= max_station; stat++) {
      int numsubs = ((stat == 1) ? max_subsector : 1);
      for (int sect = min_sector; sect <= max_sector; sect++) {
        for (int subs = min_subsector; subs <= numsubs; subs++) {
          for (int cham = min_chamber; cham <= max_chamber; cham++) {
            // extract the ring number
            int ring = CSCTriggerNumbering::ringFromTriggerLabels(stat, cham);

            // case when you want to ignore ME42
            if (disableME42_ && stat == 4 && ring == 2)
              continue;

            CSCMotherboard* tmb = tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1].get();

            tmb->setCSCGeometry(csc_g);

            // actual chamber number =/= trigger chamber number
            int chid = CSCTriggerNumbering::chamberFromTriggerLabels(sect, subs, stat, cham);

            // 0th layer means whole chamber.
            CSCDetId detid(endc, stat, ring, chid, 0);

            // Run processors only if chamber exists in geometry.
            if (tmb == nullptr || csc_g->chamber(detid) == nullptr)
              continue;

            // Skip chambers marked as bad (usually includes most of ME4/2 chambers;
            // also, there's no ME1/a-1/b separation, it's whole ME1/1)
            if (checkBadChambers_ && badChambers->isInBadChamber(detid))
              continue;

            /// running upgraded ME1/1 TMBs with GEMs
            if (stat == 1 && ring == 1 && runPhase2_ && runME11ILT_) {
              // run the TMB
              CSCGEMMotherboardME11* tmb11GEM = static_cast<CSCGEMMotherboardME11*>(tmb);
              tmb11GEM->setCSCGeometry(csc_g);
              tmb11GEM->setGEMGeometry(gem_g);
              if (infoV > 1)
                LogTrace("CSCTriggerPrimitivesBuilder")
                    << "CSCTriggerPrimitivesBuilder::build in E:" << endc << " S:" << stat << " R:" << ring;

              tmb11GEM->run(wiredc, compdc, gemClusters);

              // 0th layer means whole chamber.
              GEMDetId gemId(detid.zendcap(), 1, 1, 0, chid, 0);

              // get the collections
              const std::vector<CSCCorrelatedLCTDigi>& lctV = tmb11GEM->readoutLCTs1b();
              const std::vector<CSCCorrelatedLCTDigi>& lctV1a = tmb11GEM->readoutLCTs1a();
              const std::vector<CSCALCTDigi>& alctV = tmb11GEM->alctProc->readoutALCTs();
              const std::vector<CSCCLCTDigi>& clctV = tmb11GEM->clctProc->readoutCLCTs();
              const std::vector<int>& preTriggerBXs = tmb11GEM->clctProc->preTriggerBXs();
              const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV = tmb11GEM->clctProc->preTriggerDigis();
              const std::vector<GEMCoPadDigi>& copads = tmb11GEM->coPadProcessor->readoutCoPads();
              const std::vector<CSCALCTPreTriggerDigi>& alctpretriggerV = tmb11GEM->alctProc->preTriggerDigis();
              const CSCShowerDigi& shower = tmb11GEM->readoutShower();
              const CSCShowerDigi& anodeShower = tmb11GEM->alctProc->readoutShower();

              // ME1/b
              if (!(lctV.empty() && alctV.empty() && clctV.empty()) and infoV > 1) {
                LogTrace("L1CSCTrigger") << "CSCTriggerPrimitivesBuilder results in " << detid;
              }

              // put collections in event
              put(lctV, oc_lct, detid, " ME1b LCT digi");
              put(lctV1a, oc_lct, detid, " ME1a LCT digi");
              put(alctV, oc_alct, detid, " ME1b ALCT digi");
              put(clctV, oc_clct, detid, " ME1b CLCT digi");
              put(pretriggerV, oc_pretrigger, detid, " ME1b CLCT pre-trigger digi");
              put(preTriggerBXs, oc_pretrig, detid, " ME1b CLCT pre-trigger BX");
              put(copads, oc_gemcopad, gemId, " GEM coincidence pad");
              put(alctpretriggerV, oc_alctpretrigger, detid, " ME1b ALCT pre-trigger digi");
              if (shower.isValid())
                oc_shower.insertDigi(detid, shower);
              if (anodeShower.isValid())
                oc_shower_anode.insertDigi(detid, anodeShower);
            }

            // running upgraded ME2/1 TMBs
            else if (stat == 2 && ring == 1 && runPhase2_ && runME21ILT_) {
              // run the TMB
              CSCGEMMotherboardME21* tmb21GEM = static_cast<CSCGEMMotherboardME21*>(tmb);
              tmb21GEM->setCSCGeometry(csc_g);
              tmb21GEM->setGEMGeometry(gem_g);

              tmb21GEM->run(wiredc, compdc, gemClusters);

              // 0th layer means whole chamber.
              GEMDetId gemId(detid.zendcap(), 1, 2, 0, chid, 0);

              // get the collections
              const std::vector<CSCCorrelatedLCTDigi>& lctV = tmb21GEM->readoutLCTs();
              const std::vector<CSCALCTDigi>& alctV = tmb21GEM->alctProc->readoutALCTs();
              const std::vector<CSCCLCTDigi>& clctV = tmb21GEM->clctProc->readoutCLCTs();
              const std::vector<int>& preTriggerBXs = tmb21GEM->clctProc->preTriggerBXs();
              const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV = tmb21GEM->clctProc->preTriggerDigis();
              const std::vector<GEMCoPadDigi>& copads = tmb21GEM->coPadProcessor->readoutCoPads();
              const std::vector<CSCALCTPreTriggerDigi>& alctpretriggerV = tmb21GEM->alctProc->preTriggerDigis();
              const CSCShowerDigi& shower = tmb21GEM->readoutShower();
              const CSCShowerDigi& anodeShower = tmb21GEM->alctProc->readoutShower();

              if (!(alctV.empty() && clctV.empty() && lctV.empty()) and infoV > 1) {
                LogTrace("L1CSCTrigger") << "CSCTriggerPrimitivesBuilder got results in " << detid;
              }

              // put collections in event
              put(lctV, oc_lct, detid, " ME21 LCT digi");
              put(alctV, oc_alct, detid, " ME21 ALCT digi");
              put(clctV, oc_clct, detid, " ME21 CLCT digi");
              put(pretriggerV, oc_pretrigger, detid, " ME21 CLCT pre-trigger digi");
              put(preTriggerBXs, oc_pretrig, detid, " ME21 CLCT pre-trigger BX");
              put(copads, oc_gemcopad, gemId, " GEM coincidence pad");
              put(alctpretriggerV, oc_alctpretrigger, detid, " ME21 ALCT pre-trigger digi");
              if (shower.isValid())
                oc_shower.insertDigi(detid, shower);
              if (anodeShower.isValid())
                oc_shower_anode.insertDigi(detid, anodeShower);
            }
            // running upgraded ME2/1-ME3/1-ME4/1 TMBs (without GEMs or RPCs)
            else if (runPhase2_ and ring == 1 and
                     ((stat == 2 and runME21Up_ and !runME21ILT_) || (stat == 3 and runME31Up_) ||
                      (stat == 4 and runME41Up_))) {
              // run the TMB
              CSCUpgradeMotherboard* utmb = static_cast<CSCUpgradeMotherboard*>(tmb);
              utmb->setCSCGeometry(csc_g);
              utmb->run(wiredc, compdc);

              // get the collections
              const std::vector<CSCCorrelatedLCTDigi>& lctV = utmb->readoutLCTs();
              const std::vector<CSCALCTDigi>& alctV = utmb->alctProc->readoutALCTs();
              const std::vector<CSCCLCTDigi>& clctV = utmb->clctProc->readoutCLCTs();
              const std::vector<int>& preTriggerBXs = utmb->clctProc->preTriggerBXs();
              const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV = utmb->clctProc->preTriggerDigis();
              const std::vector<CSCALCTPreTriggerDigi>& alctpretriggerV = utmb->alctProc->preTriggerDigis();
              const CSCShowerDigi& shower = utmb->readoutShower();
              const CSCShowerDigi& anodeShower = utmb->alctProc->readoutShower();

              if (!(alctV.empty() && clctV.empty() && lctV.empty()) and infoV > 1) {
                LogTrace("L1CSCTrigger") << "CSCTriggerPrimitivesBuilder got results in " << detid;
              }

              // put collections in event
              put(lctV, oc_lct, detid, " LCT digi");
              put(alctV, oc_alct, detid, " ALCT digi");
              put(clctV, oc_clct, detid, " CLCT digi");
              put(pretriggerV, oc_pretrigger, detid, " CLCT pre-trigger digi");
              put(preTriggerBXs, oc_pretrig, detid, " CLCT pre-trigger BX");
              put(alctpretriggerV, oc_alctpretrigger, detid, " ALCT pre-trigger digi");
              if (shower.isValid())
                oc_shower.insertDigi(detid, shower);
              if (anodeShower.isValid())
                oc_shower_anode.insertDigi(detid, anodeShower);
            }

            // running non-upgraded TMB
            else {
              // run the TMB
              tmb->run(wiredc, compdc);

              // get the collections
              const std::vector<CSCCorrelatedLCTDigi>& lctV = tmb->readoutLCTs();
              const std::vector<CSCALCTDigi>& alctV = tmb->alctProc->readoutALCTs();
              const std::vector<CSCCLCTDigi>& clctV = tmb->clctProc->readoutCLCTs();
              const std::vector<int>& preTriggerBXs = tmb->clctProc->preTriggerBXs();
              const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV = tmb->clctProc->preTriggerDigis();
              const std::vector<CSCALCTPreTriggerDigi>& alctpretriggerV = tmb->alctProc->preTriggerDigis();
              const CSCShowerDigi& shower = tmb->readoutShower();
              const CSCShowerDigi& anodeShower = tmb->alctProc->readoutShower();

              if (!(alctV.empty() && clctV.empty() && lctV.empty()) and infoV > 1) {
                LogTrace("L1CSCTrigger") << "CSCTriggerPrimitivesBuilder got results in " << detid;
              }

              // put collections in event
              put(lctV, oc_lct, detid, tmb->getCSCName() + " LCT digi");
              put(alctV, oc_alct, detid, tmb->getCSCName() + " ALCT digi");
              put(clctV, oc_clct, detid, tmb->getCSCName() + " CLCT digi");
              put(pretriggerV, oc_pretrigger, detid, tmb->getCSCName() + " CLCT pre-trigger digi");
              put(preTriggerBXs, oc_pretrig, detid, tmb->getCSCName() + " CLCT pre-trigger BX");
              put(alctpretriggerV, oc_alctpretrigger, detid, tmb->getCSCName() + " ALCT pre-trigger digi");
              if (shower.isValid())
                oc_shower.insertDigi(detid, shower);
              if (anodeShower.isValid())
                oc_shower_anode.insertDigi(detid, anodeShower);
            }  // non-upgraded TMB
          }
        }
      }
    }
  }

  // run MPC simulation
  // there are 2 x 4 x 6 MPC VME cards
  for (int endc = min_endcap; endc <= max_endcap; endc++) {
    for (int stat = min_station; stat <= max_station; stat++) {
      for (int sect = min_sector; sect <= max_sector; sect++) {
        auto mpc = mpc_[endc - 1][stat - 1][sect - 1].get();

        // load the LCTs relevant for this MPC
        mpc->loadLCTs(oc_lct);

        // sort and select the LCTs (if applicable)
        mpc->sortLCTs();

        // get sorted+selected LCTs
        const auto& result = mpc->getLCTs();

        // now convert csctf::TrackStub back into CSCCorrelatedLCTDigi
        // put MPC stubs into the event
        for (const auto& lct : result) {
          oc_sorted_lct.insertDigi(CSCDetId(lct.getDetId().rawId()), *(lct.getDigi()));
          if (infoV > 1)
            LogDebug("CSCTriggerPrimitivesBuilder")
                << "MPC " << *(lct.getDigi()) << " found in ME" << ((lct.endcap() == 1) ? "+" : "-") << lct.station()
                << "/" << CSCDetId(lct.getDetId().rawId()).ring() << "/" << CSCDetId(lct.getDetId().rawId()).chamber()
                << " (sector " << lct.sector() << " trig id. " << lct.cscid() << ")"
                << "\n";
        }
      }
    }
  }
}
