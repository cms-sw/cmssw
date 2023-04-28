#include <memory>

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCTriggerPrimitivesBuilder.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCMotherboard.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCGEMMotherboard.h"
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
  disableME42_ = commonParams.getParameter<bool>("disableME42");

  checkBadChambers_ = conf.getParameter<bool>("checkBadChambers");
  selectedChambers_ = conf.getParameter<std::vector<std::string>>("selectedChambers");

  runME11Up_ = commonParams.getParameter<bool>("runME11Up");
  runME21Up_ = commonParams.getParameter<bool>("runME21Up");

  runME11ILT_ = commonParams.getParameter<bool>("runME11ILT");
  runME21ILT_ = commonParams.getParameter<bool>("runME21ILT");

  // Initializing boards.
  CSCBaseboard::Parameters baseparams(conf);
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

            const bool upgrade = runPhase2_ and ring == 1;
            const bool upgradeGE11 = upgrade and stat == 1 and runME11Up_ and runME11ILT_;
            const bool upgradeGE21 = upgrade and stat == 2 and runME21Up_ and runME21ILT_;

            // GE1/1-ME1/1 integrated local trigger or GE2/1-ME2/1 integrated local trigger
            if (upgradeGE11 or upgradeGE21)
              tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1] =
                  std::make_unique<CSCGEMMotherboard>(endc, stat, sect, subs, cham, baseparams);
            // default case
            else
              tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1] =
                  std::make_unique<CSCMotherboard>(endc, stat, sect, subs, cham, baseparams);
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

void CSCTriggerPrimitivesBuilder::setESLookupTables(const CSCL1TPLookupTableCCLUT* conf) {
  // Receives CSCL1TPLookupTableCCLUT percolated down from ESProducer.
  for (int endc = min_endcap; endc <= max_endcap; endc++) {
    for (int stat = min_station; stat <= max_station; stat++) {
      int numsubs = ((stat == 1) ? max_subsector : 1);
      for (int sect = min_sector; sect <= max_sector; sect++) {
        for (int subs = min_subsector; subs <= numsubs; subs++) {
          for (int cham = min_chamber; cham <= max_chamber; cham++) {
            tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1]->setESLookupTables(conf);
          }
        }
      }
    }
  }
}

void CSCTriggerPrimitivesBuilder::setESLookupTables(const CSCL1TPLookupTableME11ILT* conf) {
  for (int endc = min_endcap; endc <= max_endcap; endc++) {
    for (int stat = min_station; stat <= max_station; stat++) {
      int numsubs = ((stat == 1) ? max_subsector : 1);
      for (int sect = min_sector; sect <= max_sector; sect++) {
        for (int subs = min_subsector; subs <= numsubs; subs++) {
          for (int cham = min_chamber; cham <= max_chamber; cham++) {
            if (tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1]->id().isME11())
              tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1]->setESLookupTables(conf);
          }
        }
      }
    }
  }
}

void CSCTriggerPrimitivesBuilder::setESLookupTables(const CSCL1TPLookupTableME21ILT* conf) {
  for (int endc = min_endcap; endc <= max_endcap; endc++) {
    for (int stat = min_station; stat <= max_station; stat++) {
      int numsubs = ((stat == 1) ? max_subsector : 1);
      for (int sect = min_sector; sect <= max_sector; sect++) {
        for (int subs = min_subsector; subs <= numsubs; subs++) {
          for (int cham = min_chamber; cham <= max_chamber; cham++) {
            if (tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1]->id().isME21())
              tmb_[endc - 1][stat - 1][sect - 1][subs - 1][cham - 1]->setESLookupTables(conf);
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
                                        CSCShowerDigiCollection& oc_shower_anode,
                                        CSCShowerDigiCollection& oc_shower_cathode,
                                        CSCShowerDigiCollection& oc_shower,
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

            //only process the selected chambers when selectedChambers is not empty
            if (!selectedChambers_.empty()) {
              if (std::find(selectedChambers_.begin(), selectedChambers_.end(), detid.chamberName()) ==
                  selectedChambers_.end()) {
                continue;
              }
            }
            const bool upgrade = runPhase2_ and ring == 1;
            const bool upgradeGE11 = upgrade and stat == 1 and runME11Up_ and runME11ILT_;
            const bool upgradeGE21 = upgrade and stat == 2 and runME21Up_ and runME21ILT_;

            // GE1/1-ME1/1 integrated local trigger or GE2/1-ME2/1 integrated local trigger
            if (upgradeGE11 or upgradeGE21) {
              // run the TMB
              CSCGEMMotherboard* tmbGEM = static_cast<CSCGEMMotherboard*>(tmb);
              tmbGEM->setGEMGeometry(gem_g);
              tmbGEM->setCSCGeometry(csc_g);
              tmbGEM->run(wiredc, compdc, gemClusters);

              // 0th layer means whole chamber.
              GEMDetId gemId(detid.zendcap(), 1, stat, 0, chid, 0);
              const std::vector<GEMCoPadDigi>& copads = tmbGEM->clusterProc()->readoutCoPads();
              put(copads, oc_gemcopad, gemId, " GEM coincidence pad");
            }
            // default case: regular TMBs and OTMBs without GEMs
            else {
              // run the TMB
              tmb->run(wiredc, compdc);
            }

            // get the collections

            // trigger primitives
            const std::vector<CSCALCTDigi>& alctV = tmb->alctProc->readoutALCTs();
            const std::vector<CSCCLCTDigi>& clctV = tmb->clctProc->readoutCLCTs();
            const std::vector<CSCCorrelatedLCTDigi>& lctV = tmb->readoutLCTs();

            // pre-triggers
            const std::vector<int>& preTriggerBXs = tmb->clctProc->preTriggerBXs();
            const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV = tmb->clctProc->preTriggerDigis();
            const std::vector<CSCALCTPreTriggerDigi>& alctpretriggerV = tmb->alctProc->preTriggerDigis();

            // showers
            const std::vector<CSCShowerDigi>& shower = tmb->readoutShower();
            const std::vector<CSCShowerDigi>& anodeShower = tmb->alctProc->readoutShower();
            const std::vector<CSCShowerDigi>& cathodeShower = tmb->clctProc->readoutShower();

            put(alctV, oc_alct, detid, tmb->getCSCName() + " ALCT digi");
            put(clctV, oc_clct, detid, tmb->getCSCName() + " CLCT digi");
            put(lctV, oc_lct, detid, tmb->getCSCName() + " LCT digi");

            put(preTriggerBXs, oc_pretrig, detid, tmb->getCSCName() + " CLCT pre-trigger BX");
            put(pretriggerV, oc_pretrigger, detid, tmb->getCSCName() + " CLCT pre-trigger digi");
            put(alctpretriggerV, oc_alctpretrigger, detid, tmb->getCSCName() + " ALCT pre-trigger digi");

            put(shower, oc_shower, detid, tmb->getCSCName() + "TMB shower");
            put(anodeShower, oc_shower_anode, detid, tmb->getCSCName() + "Anode shower");
            put(cathodeShower, oc_shower_cathode, detid, tmb->getCSCName() + "Cathode shower");
            //if (shower.isValid())
            //  oc_shower.insertDigi(detid, shower);
            //if (anodeShower.isValid())
            //  oc_shower_anode.insertDigi(detid, anodeShower);
            //if (cathodeShower.isValid())
            //  oc_shower_cathode.insertDigi(detid, cathodeShower);

            if (!(alctV.empty() && clctV.empty() && lctV.empty()) and infoV > 1) {
              LogTrace("L1CSCTrigger") << "CSCTriggerPrimitivesBuilder got results in " << detid;
            }
          }  // end loop on cham
        }    // end loop on subs
      }      // end loop on sect
    }        // end loop on stat
  }          // end loop on endc

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
