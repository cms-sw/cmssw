//-----------------------------------------------------------------------------
//
//   Class: CSCTriggerPrimitivesBuilder
//
//   Description: Algorithm to build anode, cathode, and correlated LCTs
//                in each endcap muon CSC chamber from wire and comparator
//                digis.
//
//   Author List: S. Valuev, UCLA.
//
//   $Date: 2010/08/04 10:21:24 $
//   $Revision: 1.19 $
//
//   Modifications:
//
//-----------------------------------------------------------------------------

#include <L1Trigger/CSCTriggerPrimitives/src/CSCTriggerPrimitivesBuilder.h>
#include <L1Trigger/CSCTriggerPrimitives/src/CSCMotherboard.h>
#include <L1Trigger/CSCTriggerPrimitives/src/CSCMuonPortCard.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

//------------------
// Static variables
//------------------
const int CSCTriggerPrimitivesBuilder::min_endcap  = CSCDetId::minEndcapId();
const int CSCTriggerPrimitivesBuilder::max_endcap  = CSCDetId::maxEndcapId();
const int CSCTriggerPrimitivesBuilder::min_station = CSCDetId::minStationId();
const int CSCTriggerPrimitivesBuilder::max_station = CSCDetId::maxStationId();
const int CSCTriggerPrimitivesBuilder::min_sector  =
                                  CSCTriggerNumbering::minTriggerSectorId();
const int CSCTriggerPrimitivesBuilder::max_sector  =
                                  CSCTriggerNumbering::maxTriggerSectorId();
const int CSCTriggerPrimitivesBuilder::min_subsector =
                                  CSCTriggerNumbering::minTriggerSubSectorId();
const int CSCTriggerPrimitivesBuilder::max_subsector =
                                  CSCTriggerNumbering::maxTriggerSubSectorId();
const int CSCTriggerPrimitivesBuilder::min_chamber =
                                  CSCTriggerNumbering::minTriggerCscId();
const int CSCTriggerPrimitivesBuilder::max_chamber =
                                  CSCTriggerNumbering::maxTriggerCscId();

//-------------
// Constructor
//-------------
CSCTriggerPrimitivesBuilder::CSCTriggerPrimitivesBuilder(const edm::ParameterSet& conf) {
  // Receives ParameterSet percolated down from EDProducer.

  // ORCA way of initializing boards.
  for (int endc = min_endcap; endc <= max_endcap; endc++) {
    for (int stat = min_station; stat <= max_station; stat++) {
      int numsubs = ((stat == 1) ? max_subsector : 1);
      for (int sect = min_sector; sect <= max_sector; sect++) {
	for (int subs = min_subsector; subs <= numsubs; subs++) {
	  for (int cham = min_chamber; cham <= max_chamber; cham++) {
	    if ((endc <= 0 || endc > MAX_ENDCAPS)    ||
		(stat <= 0 || stat > MAX_STATIONS)   ||
		(sect <= 0 || sect > MAX_SECTORS)    ||
		(subs <= 0 || subs > MAX_SUBSECTORS) ||
		(cham <= 0 || stat > MAX_CHAMBERS)) {
	      edm::LogError("L1CSCTPEmulatorSetupError")
		<< "+++ trying to instantiate TMB of illegal CSC id ["
		<< " endcap = "  << endc << " station = "   << stat
		<< " sector = "  << sect << " subsector = " << subs
		<< " chamber = " << cham << "]; skipping it... +++\n";
	      continue;
	    }
	    // When the motherboard is instantiated, it instantiates ALCT
	    // and CLCT processors.
	    tmb_[endc-1][stat-1][sect-1][subs-1][cham-1] =
	      new CSCMotherboard(endc, stat, sect, subs, cham, conf);
	  }
	}
      }
    }
  }

  // Get min and max BX to sort LCTs in MPC.
  m_minBX = conf.getParameter<int>("MinBX");
  m_maxBX = conf.getParameter<int>("MaxBX");

  // Init MPC
  m_muonportcard = new CSCMuonPortCard();
}

//------------
// Destructor
//------------
CSCTriggerPrimitivesBuilder::~CSCTriggerPrimitivesBuilder() {
  for (int endc = min_endcap; endc <= max_endcap; endc++) {
    for (int stat = min_station; stat <= max_station; stat++) {
      int numsubs = ((stat == 1) ? max_subsector : 1);
      for (int sect = min_sector; sect <= max_sector; sect++) {
	for (int subs = min_subsector; subs <= numsubs; subs++) {
	  for (int cham = min_chamber; cham <= max_chamber; cham++) {
	    delete tmb_[endc-1][stat-1][sect-1][subs-1][cham-1];
	  }
	}
      }
    }
  }
  delete m_muonportcard;
}

//------------
// Operations
//------------
// Set configuration parameters obtained via EventSetup mechanism.
void CSCTriggerPrimitivesBuilder::setConfigParameters(const CSCDBL1TPParameters* conf) {
  // Receives CSCDBL1TPParameters percolated down from ESProducer.

  for (int endc = min_endcap; endc <= max_endcap; endc++) {
    for (int stat = min_station; stat <= max_station; stat++) {
      int numsubs = ((stat == 1) ? max_subsector : 1);
      for (int sect = min_sector; sect <= max_sector; sect++) {
	for (int subs = min_subsector; subs <= numsubs; subs++) {
	  for (int cham = min_chamber; cham <= max_chamber; cham++) {
	    CSCMotherboard* tmb = tmb_[endc-1][stat-1][sect-1][subs-1][cham-1];
	    tmb->setConfigParameters(conf);
	  }
	}
      }
    }
  }
}

// Build anode, cathode, and correlated LCTs in each chamber and fill them
// into output collections.  Pass collections of wire and comparator digis
// to Trigger MotherBoard (TMB) processors, which, in turn, pass them to
// ALCT and CLCT processors.  Up to 2 anode and 2 cathode LCTs can be found
// in each chamber during any bunch crossing.  The 2 projections are then
// combined into three-dimensional "correlated" LCTs in the TMB.  Finally,
// MPC processor sorts up to 18 LCTs from 9 TMBs and writes collections of
// up to 3 best LCTs per (sub)sector into Event (to be used by the Sector
// Receiver).
void CSCTriggerPrimitivesBuilder::build(const CSCBadChambers* badChambers,
					const CSCWireDigiCollection* wiredc,
					const CSCComparatorDigiCollection* compdc,
					CSCALCTDigiCollection& oc_alct,
					CSCCLCTDigiCollection& oc_clct,
                                        CSCCLCTPreTriggerCollection & oc_pretrig,
					CSCCorrelatedLCTDigiCollection& oc_lct,
					CSCCorrelatedLCTDigiCollection& oc_sorted_lct) {
  // CSC geometry.
  CSCTriggerGeomManager* theGeom = CSCTriggerGeometry::get();

  for (int endc = min_endcap; endc <= max_endcap; endc++) {
    for (int stat = min_station; stat <= max_station; stat++) {
      int numsubs = ((stat == 1) ? max_subsector : 1);
      for (int sect = min_sector; sect <= max_sector; sect++) {
	for (int subs = min_subsector; subs <= numsubs; subs++) {
	  for (int cham = min_chamber; cham <= max_chamber; cham++) {
	    CSCMotherboard* tmb = tmb_[endc-1][stat-1][sect-1][subs-1][cham-1];

	    // Run processors only if chamber exists in geometry.
	    if (tmb != 0 &&
		theGeom->chamber(endc, stat, sect, subs, cham) != 0) {

	      // Calculate DetId.
	      int ring =
		CSCTriggerNumbering::ringFromTriggerLabels(stat, cham);
	      int chid =
		CSCTriggerNumbering::chamberFromTriggerLabels(sect, subs,
							      stat, cham);
	      // 0th layer means whole chamber.
	      CSCDetId detid(endc, stat, ring, chid, 0);

	      // Skip chambers marked as bad (includes most of ME4/2 chambers).
	      if (badChambers->isInBadChamber(detid)) continue;

	      std::vector<CSCCorrelatedLCTDigi> lctV = tmb->run(wiredc,compdc);

	      std::vector<CSCALCTDigi> alctV = tmb->alct->readoutALCTs();
	      std::vector<CSCCLCTDigi> clctV = tmb->clct->readoutCLCTs();
              std::vector<int> preTriggerBXs = tmb->clct->preTriggerBXs();

	      // Skip to next chamber if there are no LCTs to save.
	      if (alctV.empty() && clctV.empty() && lctV.empty()) continue;

	      // Correlated LCTs.
	      if (!lctV.empty()) {
		LogTrace("L1CSCTrigger")
		  << "Put " << lctV.size() << " LCT digi"
		  << ((lctV.size() > 1) ? "s " : " ") << "in collection\n";
		oc_lct.put(std::make_pair(lctV.begin(),lctV.end()), detid);
	      }

	      // Anode LCTs.
	      if (!alctV.empty()) {
		LogTrace("L1CSCTrigger")
		  << "Put " << alctV.size() << " ALCT digi"
		  << ((alctV.size() > 1) ? "s " : " ") << "in collection\n";
		oc_alct.put(std::make_pair(alctV.begin(),alctV.end()), detid);
	      }

	      // Cathode LCTs.
	      if (!clctV.empty()) {
		LogTrace("L1CSCTrigger")
		  << "Put " << clctV.size() << " CLCT digi"
		  << ((clctV.size() > 1) ? "s " : " ") << "in collection\n";
		oc_clct.put(std::make_pair(clctV.begin(),clctV.end()), detid);
	      }

              // Cathode LCTs pretriggers
              if (!preTriggerBXs.empty()) {
                LogTrace("L1CSCTrigger")
                  << "Put " << preTriggerBXs.size() << " CLCT pretriggers"
                  << ((preTriggerBXs.size() > 1) ? "s " : " ") << "in collection\n";
                oc_pretrig.put(std::make_pair(preTriggerBXs.begin(),preTriggerBXs.end()), detid);
              }
	    }
	  }
	}
      }
    }
  }

  // run MPC simulation
  m_muonportcard->loadDigis(oc_lct);

  std::vector<csctf::TrackStub> result;
  for(int bx = m_minBX; bx <= m_maxBX; ++bx)
    for(int e = min_endcap; e <= max_endcap; ++e)
      for(int st = min_station; st <= max_station; ++st)
        for(int se = min_sector; se <= max_sector; ++se)
          {
            if(st == 1)
              {
		std::vector<csctf::TrackStub> subs1, subs2;
                subs1 = m_muonportcard->sort(e, st, se, 1, bx);
                subs2 = m_muonportcard->sort(e, st, se, 2, bx);
                result.insert(result.end(), subs1.begin(), subs1.end());
                result.insert(result.end(), subs2.begin(), subs2.end());
              }
            else
              {
		std::vector<csctf::TrackStub> sector;
                sector = m_muonportcard->sort(e, st, se, 0, bx);
                result.insert(result.end(), sector.begin(), sector.end());
              }
          }

  std::vector<csctf::TrackStub>::const_iterator itr = result.begin();
  for (; itr != result.end(); itr++) {
    oc_sorted_lct.insertDigi(CSCDetId(itr->getDetId().rawId()), *(itr->getDigi()));
    LogDebug("L1CSCTrigger")
      << "MPC " << *(itr->getDigi()) << " found in ME"
      << ((itr->endcap() == 1) ? "+" : "-") << itr->station() << "/"
      << CSCDetId(itr->getDetId().rawId()).ring() << "/"
      << CSCDetId(itr->getDetId().rawId()).chamber()
      << " (sector " << itr->sector()
      << " trig id. " << itr->cscid() << ")" << "\n";
  }
}
