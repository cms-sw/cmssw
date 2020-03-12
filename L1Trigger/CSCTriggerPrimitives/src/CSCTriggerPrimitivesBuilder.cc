#include "L1Trigger/CSCTriggerPrimitives/interface/CSCTriggerPrimitivesBuilder.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCMotherboard.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCMotherboardME11.h"
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
  isSLHC_ = commonParams.getParameter<bool>("isSLHC");
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

  useClusters_ = commonParams.getParameter<bool>("useClusters");

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
            if (isSLHC_ and ring == 1 and stat == 1 and runME11Up_ and !runME11ILT_)
              tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1].reset(
                  new CSCMotherboardME11(endc, stat, sect, subs, cham, conf));
            else if (isSLHC_ and ring == 1 and stat == 1 and runME11Up_ and runME11ILT_)
              tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1].reset(
                  new CSCGEMMotherboardME11(endc, stat, sect, subs, cham, conf));
            else if (isSLHC_ and ring == 1 and stat == 2 and runME21Up_ and !runME21ILT_)
              tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1].reset(
                  new CSCUpgradeMotherboard(endc, stat, sect, subs, cham, conf));
            else if (isSLHC_ and ring == 1 and stat == 2 and runME21Up_ and runME21ILT_)
              tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1].reset(
                  new CSCGEMMotherboardME21(endc, stat, sect, subs, cham, conf));
            else if (isSLHC_ and ring == 1 and ((stat == 3 and runME31Up_) || (stat == 4 and runME41Up_)))
              tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1].reset(
                  new CSCUpgradeMotherboard(endc, stat, sect, subs, cham, conf));
            else
              tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1].reset(
                  new CSCMotherboard(endc, stat, sect, subs, cham, conf));
          }
        }
      }
    }
  }

  // Get min and max BX to sort LCTs in MPC.
  m_minBX_ = conf.getParameter<int>("MinBX");
  m_maxBX_ = conf.getParameter<int>("MaxBX");

  // Init MPC
  m_muonportcard.reset(new CSCMuonPortCard(conf));
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
                                        const GEMPadDigiCollection* gemPads,
                                        const GEMPadDigiClusterCollection* gemClusters,
                                        CSCALCTDigiCollection& oc_alct,
                                        CSCALCTDigiCollection& oc_alct_all,
                                        CSCCLCTDigiCollection& oc_clct,
                                        CSCCLCTDigiCollection& oc_clct_all,
                                        CSCALCTPreTriggerDigiCollection& oc_alctpretrigger,
                                        CSCCLCTPreTriggerDigiCollection& oc_pretrigger,
                                        CSCCLCTPreTriggerCollection& oc_pretrig,
                                        CSCCorrelatedLCTDigiCollection& oc_lct,
                                        CSCCorrelatedLCTDigiCollection& oc_sorted_lct,
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

            // running upgraded ME1/1 TMBs
            if (stat == 1 && ring == 1 && isSLHC_ && !runME11ILT_) {
              // run the TMB
              CSCMotherboardME11* tmb11 = static_cast<CSCMotherboardME11*>(tmb);
              if (infoV > 1)
                LogTrace("CSCTriggerPrimitivesBuilder")
                    << "CSCTriggerPrimitivesBuilder::build in E:" << endc << " S:" << stat << " R:" << ring;
              tmb11->run(wiredc, compdc);

              // get all collections
              // all ALCTs, CLCTs, LCTs are written with detid ring = 1, as data did
              // but CLCTs and LCTs are written sperately in ME1a and ME1b, considering whether ME1a is disabled or not
              const std::vector<CSCCorrelatedLCTDigi>& lctV = tmb11->readoutLCTs1b();
              const std::vector<CSCCorrelatedLCTDigi>& lctV1a = tmb11->readoutLCTs1a();
              const std::vector<CSCALCTDigi>& alctV = tmb11->alctProc->readoutALCTs();
              const std::vector<CSCALCTDigi>& alctV_all = tmb11->alctProc->getALCTs();
              const std::vector<CSCCLCTDigi>& clctV = tmb11->clctProc->readoutCLCTsME1b();
              const std::vector<CSCCLCTDigi>& clctV_all = tmb11->clctProc->getCLCTs();
              const std::vector<int> preTriggerBXs = tmb11->clctProc->preTriggerBXs();
              const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV = tmb11->clctProc->preTriggerDigisME1b();
              const std::vector<CSCCLCTDigi>& clctV1a = tmb11->clctProc->readoutCLCTsME1a();
              const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV1a = tmb11->clctProc->preTriggerDigisME1a();
              const std::vector<CSCALCTPreTriggerDigi>& alctpretriggerV = tmb11->alctProc->preTriggerDigis();

              if (infoV > 1)
                LogTrace("CSCTriggerPrimitivesBuilder")
                    << "CSCTriggerPrimitivesBuilder:: a=" << alctV.size() << " c=" << clctV.size()
                    << " l=" << lctV.size() << " c=" << clctV1a.size() << " l=" << lctV1a.size();

              // ME1/b

              if (!(lctV.empty() && alctV.empty() && clctV.empty()) and infoV > 1) {
                LogTrace("L1CSCTrigger") << "CSCTriggerPrimitivesBuilder results in " << detid;
              }
              // put collections in event
              put(lctV, oc_lct, detid, " ME1b LCT digi");
              put(alctV, oc_alct, detid, " ME1b ALCT digi");
              put(alctV_all, oc_alct_all, detid, " ME1b ALCT digi");
              put(clctV, oc_clct, detid, " ME1b CLCT digi");
              put(clctV_all, oc_clct_all, detid, " ME1b CLCT digi");
              put(pretriggerV, oc_pretrigger, detid, " ME1b CLCT pre-trigger digi");
              put(preTriggerBXs, oc_pretrig, detid, " ME1b CLCT pre-trigger BX");
              put(alctpretriggerV, oc_alctpretrigger, detid, " ME1b ALCT pre-trigger digi");

              // ME1/a

              if (disableME1a_)
                continue;

              CSCDetId detid1a(endc, stat, 4, chid, 0);

              if (!(lctV1a.empty() && clctV1a.empty()) and infoV > 1) {
                LogTrace("L1CSCTrigger") << "CSCTriggerPrimitivesBuilder results in " << detid1a;
              }

              // put collections in event, still use detid ring =1
              put(lctV1a, oc_lct, detid, " ME1a LCT digi");
              put(clctV1a, oc_clct, detid, " ME1a CLCT digi");
              put(pretriggerV1a, oc_pretrigger, detid, " ME1a CLCT pre-trigger digi");
            }  // upgraded TMB

            // running upgraded ME1/1 TMBs with GEMs
            else if (stat == 1 && ring == 1 && isSLHC_ && runME11ILT_) {
              // run the TMB
              CSCGEMMotherboardME11* tmb11GEM = static_cast<CSCGEMMotherboardME11*>(tmb);
              tmb11GEM->setCSCGeometry(csc_g);
              tmb11GEM->setGEMGeometry(gem_g);
              if (infoV > 1)
                LogTrace("CSCTriggerPrimitivesBuilder")
                    << "CSCTriggerPrimitivesBuilder::build in E:" << endc << " S:" << stat << " R:" << ring;

              if (useClusters_) {
                tmb11GEM->run(wiredc, compdc, gemClusters);
              } else {
                tmb11GEM->run(wiredc, compdc, gemPads);
              }

              // 0th layer means whole chamber.
              GEMDetId gemId(detid.zendcap(), 1, 1, 0, chid, 0);

              // get the collections
              const std::vector<CSCCorrelatedLCTDigi>& lctV = tmb11GEM->readoutLCTs1b();
              const std::vector<CSCCorrelatedLCTDigi>& lctV1a = tmb11GEM->readoutLCTs1a();
              const std::vector<CSCALCTDigi>& alctV = tmb11GEM->alctProc->readoutALCTs();
              const std::vector<CSCALCTDigi>& alctV_all = tmb11GEM->alctProc->getALCTs();
              const std::vector<CSCCLCTDigi>& clctV = tmb11GEM->clctProc->readoutCLCTsME1b();
              const std::vector<CSCCLCTDigi>& clctV_all = tmb11GEM->clctProc->getCLCTs();
              const std::vector<int>& preTriggerBXs = tmb11GEM->clctProc->preTriggerBXs();
              const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV = tmb11GEM->clctProc->preTriggerDigisME1b();
              const std::vector<CSCCLCTDigi>& clctV1a = tmb11GEM->clctProc->readoutCLCTsME1a();
              const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV1a = tmb11GEM->clctProc->preTriggerDigisME1a();
              const std::vector<GEMCoPadDigi>& copads = tmb11GEM->coPadProcessor->readoutCoPads();
              const std::vector<CSCALCTPreTriggerDigi>& alctpretriggerV = tmb11GEM->alctProc->preTriggerDigis();

              // ME1/b
              if (!(lctV.empty() && alctV.empty() && clctV.empty()) and infoV > 1) {
                LogTrace("L1CSCTrigger") << "CSCTriggerPrimitivesBuilder results in " << detid;
              }

              // put collections in event
              put(lctV, oc_lct, detid, " ME1b LCT digi");
              put(alctV, oc_alct, detid, " ME1b ALCT digi");
              put(alctV_all, oc_alct_all, detid, " ME1b ALCT digi");
              put(clctV, oc_clct, detid, " ME1b CLCT digi");
              put(clctV_all, oc_clct_all, detid, " ME1b CLCT digi");
              put(pretriggerV, oc_pretrigger, detid, " ME1b CLCT pre-trigger digi");
              put(preTriggerBXs, oc_pretrig, detid, " ME1b CLCT pre-trigger BX");
              put(copads, oc_gemcopad, gemId, " GEM coincidence pad");
              put(alctpretriggerV, oc_alctpretrigger, detid, " ME1b ALCT pre-trigger digi");

              // ME1/a
              if (disableME1a_)
                continue;

              CSCDetId detid1a(endc, stat, 4, chid, 0);

              if (!(lctV1a.empty() && clctV1a.empty()) and infoV > 1) {
                LogTrace("L1CSCTrigger") << "CSCTriggerPrimitivesBuilder results in " << detid1a;
              }

              // put collections in event, still use detid ring =1
              put(lctV1a, oc_lct, detid, " ME1a LCT digi");
              put(clctV1a, oc_clct, detid, " ME1a CLCT digi");
              put(pretriggerV1a, oc_pretrigger, detid, " ME1a CLCT pre-trigger digi");
            }

            // running upgraded ME2/1 TMBs
            else if (stat == 2 && ring == 1 && isSLHC_ && runME21ILT_) {
              // run the TMB
              CSCGEMMotherboardME21* tmb21GEM = static_cast<CSCGEMMotherboardME21*>(tmb);
              tmb21GEM->setCSCGeometry(csc_g);
              tmb21GEM->setGEMGeometry(gem_g);

              if (useClusters_) {
                tmb21GEM->run(wiredc, compdc, gemClusters);
              } else {
                tmb21GEM->run(wiredc, compdc, gemPads);
              }

              // 0th layer means whole chamber.
              GEMDetId gemId(detid.zendcap(), 1, 2, 0, chid, 0);

              // get the collections
              const std::vector<CSCCorrelatedLCTDigi>& lctV = tmb21GEM->readoutLCTs();
              const std::vector<CSCALCTDigi>& alctV = tmb21GEM->alctProc->readoutALCTs();
              const std::vector<CSCALCTDigi>& alctV_all = tmb21GEM->alctProc->getALCTs();
              const std::vector<CSCCLCTDigi>& clctV = tmb21GEM->clctProc->readoutCLCTs();
              const std::vector<CSCCLCTDigi>& clctV_all = tmb21GEM->clctProc->getCLCTs();
              const std::vector<int>& preTriggerBXs = tmb21GEM->clctProc->preTriggerBXs();
              const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV = tmb21GEM->clctProc->preTriggerDigis();
              const std::vector<GEMCoPadDigi>& copads = tmb21GEM->coPadProcessor->readoutCoPads();
              const std::vector<CSCALCTPreTriggerDigi>& alctpretriggerV = tmb21GEM->alctProc->preTriggerDigis();

              if (!(alctV.empty() && clctV.empty() && lctV.empty()) and infoV > 1) {
                LogTrace("L1CSCTrigger") << "CSCTriggerPrimitivesBuilder got results in " << detid;
              }

              // put collections in event
              put(lctV, oc_lct, detid, " ME21 LCT digi");
              put(alctV, oc_alct, detid, " ME21 ALCT digi");
              put(alctV_all, oc_alct_all, detid, " ME21 ALCT digi");
              put(clctV, oc_clct, detid, " ME21 CLCT digi");
              put(clctV_all, oc_clct_all, detid, " ME21 CLCT digi");
              put(pretriggerV, oc_pretrigger, detid, " ME21 CLCT pre-trigger digi");
              put(preTriggerBXs, oc_pretrig, detid, " ME21 CLCT pre-trigger BX");
              put(copads, oc_gemcopad, gemId, " GEM coincidence pad");
              put(alctpretriggerV, oc_alctpretrigger, detid, " ME21 ALCT pre-trigger digi");
            }
            // running upgraded ME2/1-ME3/1-ME4/1 TMBs (without GEMs or RPCs)
            else if ((stat == 2 or stat == 3 or stat == 4) && ring == 1 && isSLHC_) {
              // run the TMB
              CSCUpgradeMotherboard* utmb = static_cast<CSCUpgradeMotherboard*>(tmb);
              utmb->setCSCGeometry(csc_g);
              utmb->run(wiredc, compdc);

              // get the collections
              const std::vector<CSCCorrelatedLCTDigi>& lctV = utmb->readoutLCTs();
              const std::vector<CSCALCTDigi>& alctV = utmb->alctProc->readoutALCTs();
              const std::vector<CSCALCTDigi>& alctV_all = utmb->alctProc->getALCTs();
              const std::vector<CSCCLCTDigi>& clctV = utmb->clctProc->readoutCLCTs();
              const std::vector<CSCCLCTDigi>& clctV_all = utmb->clctProc->getCLCTs();
              const std::vector<int>& preTriggerBXs = utmb->clctProc->preTriggerBXs();
              const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV = utmb->clctProc->preTriggerDigis();
              const std::vector<CSCALCTPreTriggerDigi>& alctpretriggerV = utmb->alctProc->preTriggerDigis();

              if (!(alctV.empty() && clctV.empty() && lctV.empty()) and infoV > 1) {
                LogTrace("L1CSCTrigger") << "CSCTriggerPrimitivesBuilder got results in " << detid;
              }

              // put collections in event
              put(lctV, oc_lct, detid, " LCT digi");
              put(alctV, oc_alct, detid, " ALCT digi");
              put(alctV_all, oc_alct_all, detid, " ALCT digi");
              put(clctV, oc_clct, detid, " CLCT digi");
              put(clctV_all, oc_clct_all, detid, tmb->getCSCName() + " CLCT digi");
              put(pretriggerV, oc_pretrigger, detid, " CLCT pre-trigger digi");
              put(preTriggerBXs, oc_pretrig, detid, " CLCT pre-trigger BX");
              put(alctpretriggerV, oc_alctpretrigger, detid, " ALCT pre-trigger digi");
            }

            // running non-upgraded TMB
            else {
              // run the TMB
              tmb->run(wiredc, compdc);

              // get the collections
              const std::vector<CSCCorrelatedLCTDigi>& lctV = tmb->readoutLCTs();
              const std::vector<CSCALCTDigi>& alctV = tmb->alctProc->readoutALCTs();
              const std::vector<CSCALCTDigi>& alctV_all = tmb->alctProc->getALCTs();
              const std::vector<CSCCLCTDigi>& clctV = tmb->clctProc->readoutCLCTs();
              const std::vector<CSCCLCTDigi>& clctV_all = tmb->clctProc->getCLCTs();
              const std::vector<int>& preTriggerBXs = tmb->clctProc->preTriggerBXs();
              const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV = tmb->clctProc->preTriggerDigis();
              const std::vector<CSCALCTPreTriggerDigi>& alctpretriggerV = tmb->alctProc->preTriggerDigis();

              if (!(alctV.empty() && clctV.empty() && lctV.empty()) and infoV > 1) {
                LogTrace("L1CSCTrigger") << "CSCTriggerPrimitivesBuilder got results in " << detid;
              }

              // put collections in event
              put(lctV, oc_lct, detid, tmb->getCSCName() + " LCT digi");
              put(alctV, oc_alct, detid, tmb->getCSCName() + " ALCT digi");
              put(alctV_all, oc_alct_all, detid, tmb->getCSCName() + " ALCT digi");
              put(clctV, oc_clct, detid, tmb->getCSCName() + " CLCT digi");
              put(clctV_all, oc_clct_all, detid, tmb->getCSCName() + " CLCT digi");
              put(pretriggerV, oc_pretrigger, detid, tmb->getCSCName() + " CLCT pre-trigger digi");
              put(preTriggerBXs, oc_pretrig, detid, tmb->getCSCName() + " CLCT pre-trigger BX");
              put(alctpretriggerV, oc_alctpretrigger, detid, tmb->getCSCName() + " ALCT pre-trigger digi");
            }  // non-upgraded TMB
          }
        }
      }
    }
  }

  // run MPC simulation
  m_muonportcard->loadDigis(oc_lct);

  // sort the LCTs per sector
  // insert them into the result vector
  std::vector<csctf::TrackStub> result;
  for (int bx = m_minBX_; bx <= m_maxBX_; ++bx)
    for (int e = min_endcap; e <= max_endcap; ++e)
      for (int st = min_station; st <= max_station; ++st)
        for (int se = min_sector; se <= max_sector; ++se) {
          if (st == 1) {
            std::vector<csctf::TrackStub> subs1, subs2;
            subs1 = m_muonportcard->sort(e, st, se, 1, bx);
            subs2 = m_muonportcard->sort(e, st, se, 2, bx);
            result.insert(result.end(), subs1.begin(), subs1.end());
            result.insert(result.end(), subs2.begin(), subs2.end());
          } else {
            std::vector<csctf::TrackStub> sector;
            sector = m_muonportcard->sort(e, st, se, 0, bx);
            result.insert(result.end(), sector.begin(), sector.end());
          }
        }

  // now convert csctf::TrackStub back into CSCCorrelatedLCTDigi
  // put MPC stubs into the event
  std::vector<csctf::TrackStub>::const_iterator itr = result.begin();
  for (; itr != result.end(); itr++) {
    oc_sorted_lct.insertDigi(CSCDetId(itr->getDetId().rawId()), *(itr->getDigi()));
    if (infoV > 1)
      LogDebug("L1CSCTrigger") << "MPC " << *(itr->getDigi()) << " found in ME" << ((itr->endcap() == 1) ? "+" : "-")
                               << itr->station() << "/" << CSCDetId(itr->getDetId().rawId()).ring() << "/"
                               << CSCDetId(itr->getDetId().rawId()).chamber() << " (sector " << itr->sector()
                               << " trig id. " << itr->cscid() << ")"
                               << "\n";
  }
}
