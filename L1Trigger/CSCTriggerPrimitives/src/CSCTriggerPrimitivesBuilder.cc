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
//
//   Modifications:
//
//-----------------------------------------------------------------------------

#include "L1Trigger/CSCTriggerPrimitives/src/CSCMotherboard.h"
#include "L1Trigger/CSCTriggerPrimitives/src/CSCMotherboardME11.h"
#include "L1Trigger/CSCTriggerPrimitives/src/CSCGEMMotherboardME11.h"
#include "L1Trigger/CSCTriggerPrimitives/src/CSCGEMMotherboardME21.h"
#include "L1Trigger/CSCTriggerPrimitives/src/CSCMotherboardME3141.h"
#include "L1Trigger/CSCTriggerPrimitives/src/CSCMuonPortCard.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "L1Trigger/CSCTriggerPrimitives/src/CSCTriggerPrimitivesBuilder.h"

//------------------
// Static variables
//------------------
const int CSCTriggerPrimitivesBuilder::min_endcap  = CSCDetId::minEndcapId();
const int CSCTriggerPrimitivesBuilder::max_endcap  = CSCDetId::maxEndcapId();
const int CSCTriggerPrimitivesBuilder::min_station = CSCDetId::minStationId();
const int CSCTriggerPrimitivesBuilder::max_station = CSCDetId::maxStationId();
const int CSCTriggerPrimitivesBuilder::min_sector  = CSCTriggerNumbering::minTriggerSectorId();
const int CSCTriggerPrimitivesBuilder::max_sector  = CSCTriggerNumbering::maxTriggerSectorId();
const int CSCTriggerPrimitivesBuilder::min_subsector = CSCTriggerNumbering::minTriggerSubSectorId();
const int CSCTriggerPrimitivesBuilder::max_subsector = CSCTriggerNumbering::maxTriggerSubSectorId();
const int CSCTriggerPrimitivesBuilder::min_chamber = CSCTriggerNumbering::minTriggerCscId();
const int CSCTriggerPrimitivesBuilder::max_chamber = CSCTriggerNumbering::maxTriggerCscId();

//-------------
// Constructor
//-------------
CSCTriggerPrimitivesBuilder::CSCTriggerPrimitivesBuilder(const edm::ParameterSet& conf)
{
  // Receives ParameterSet percolated down from EDProducer.

  // special configuration parameters for ME11 treatment
  edm::ParameterSet commonParams = conf.getParameter<edm::ParameterSet>("commonParam");
  smartME1aME1b = commonParams.getParameter<bool>("smartME1aME1b");
  disableME1a = commonParams.getParameter<bool>("disableME1a");
  disableME42 = commonParams.getParameter<bool>("disableME42");

  checkBadChambers_ = conf.getParameter<bool>("checkBadChambers");

  runME11ILT_ = commonParams.existsAs<bool>("runME11ILT")?commonParams.getParameter<bool>("runME11ILT"):false;
  runME21ILT_ = commonParams.existsAs<bool>("runME21ILT")?commonParams.getParameter<bool>("runME21ILT"):false;
  runME3141ILT_ = commonParams.existsAs<bool>("runME3141ILT")?commonParams.getParameter<bool>("runME3141ILT"):false;
  useClusters_ = commonParams.existsAs<bool>("useClusters")?commonParams.getParameter<bool>("useClusters"):false;

  // Initializing boards.
  for (int endc = min_endcap; endc <= max_endcap; endc++)
  {
    for (int stat = min_station; stat <= max_station; stat++)
    {
      int numsubs = ((stat == 1) ? max_subsector : 1);
      for (int sect = min_sector; sect <= max_sector; sect++)
      {
        for (int subs = min_subsector; subs <= numsubs; subs++)
        {
          for (int cham = min_chamber; cham <= max_chamber; cham++)
          {
            if ((endc <= 0 || endc > MAX_ENDCAPS)    ||
                (stat <= 0 || stat > MAX_STATIONS)   ||
                (sect <= 0 || sect > MAX_SECTORS)    ||
                (subs <= 0 || subs > MAX_SUBSECTORS) ||
                (cham <= 0 || cham > MAX_CHAMBERS))
            {
              edm::LogError("L1CSCTPEmulatorSetupError")
                << "+++ trying to instantiate TMB of illegal CSC id ["
                << " endcap = "  << endc << " station = "   << stat
                << " sector = "  << sect << " subsector = " << subs
                << " chamber = " << cham << "]; skipping it... +++\n";
              continue;
            }
            int ring = CSCTriggerNumbering::ringFromTriggerLabels(stat, cham);
            // When the motherboard is instantiated, it instantiates ALCT
            // and CLCT processors.
            if (stat==1 && ring==1 && smartME1aME1b && !runME11ILT_)
              tmb_[endc-1][stat-1][sect-1][subs-1][cham-1].reset( new CSCMotherboardME11(endc, stat, sect, subs, cham, conf) );
            else if (stat==1 && ring==1 && smartME1aME1b && runME11ILT_)
              tmb_[endc-1][stat-1][sect-1][subs-1][cham-1].reset( new CSCGEMMotherboardME11(endc, stat, sect, subs, cham, conf) );
            else if (stat==2 && ring==1 && runME21ILT_)
              tmb_[endc-1][stat-1][sect-1][subs-1][cham-1].reset( new CSCGEMMotherboardME21(endc, stat, sect, subs, cham, conf) );
            else if ((stat==3 || stat==4) && ring==1 && runME3141ILT_)
              tmb_[endc-1][stat-1][sect-1][subs-1][cham-1].reset( new CSCMotherboardME3141(endc, stat, sect, subs, cham, conf) );
            else
              tmb_[endc-1][stat-1][sect-1][subs-1][cham-1].reset( new CSCMotherboard(endc, stat, sect, subs, cham, conf) );
          }
        }
      }
    }
  }

  // Get min and max BX to sort LCTs in MPC.
  m_minBX = conf.getParameter<int>("MinBX");
  m_maxBX = conf.getParameter<int>("MaxBX");

  // Init MPC
  m_muonportcard.reset( new CSCMuonPortCard(conf) );
}

//------------
// Destructor
//------------
CSCTriggerPrimitivesBuilder::~CSCTriggerPrimitivesBuilder()
{
}

//------------
// Operations
//------------
// Set configuration parameters obtained via EventSetup mechanism.
void CSCTriggerPrimitivesBuilder::setConfigParameters(const CSCDBL1TPParameters* conf)
{
  // Receives CSCDBL1TPParameters percolated down from ESProducer.

  for (int endc = min_endcap; endc <= max_endcap; endc++)
  {
    for (int stat = min_station; stat <= max_station; stat++)
    {
      int numsubs = ((stat == 1) ? max_subsector : 1);
      for (int sect = min_sector; sect <= max_sector; sect++)
      {
        for (int subs = min_subsector; subs <= numsubs; subs++)
        {
          for (int cham = min_chamber; cham <= max_chamber; cham++)
          {
            tmb_[endc-1][stat-1][sect-1][subs-1][cham-1]->setConfigParameters(conf);
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
                                        const GEMPadDigiCollection* gemPads,
                                        const GEMPadDigiClusterCollection* gemClusters,
                                        CSCALCTDigiCollection& oc_alct,
                                        CSCCLCTDigiCollection& oc_clct,
                                        CSCCLCTPreTriggerDigiCollection& oc_pretrigger,
                                        CSCCLCTPreTriggerCollection & oc_pretrig,
                                        CSCCorrelatedLCTDigiCollection& oc_lct,
                                        CSCCorrelatedLCTDigiCollection& oc_sorted_lct,
                                        GEMCoPadDigiCollection& oc_gemcopad)
{
  // CSC geometry.
  for (int endc = min_endcap; endc <= max_endcap; endc++)
  {
    for (int stat = min_station; stat <= max_station; stat++)
    {
      int numsubs = ((stat == 1) ? max_subsector : 1);
      for (int sect = min_sector; sect <= max_sector; sect++)
      {
        for (int subs = min_subsector; subs <= numsubs; subs++)
        {
          for (int cham = min_chamber; cham <= max_chamber; cham++)
          {
            // extract the ring number
            int ring = CSCTriggerNumbering::ringFromTriggerLabels(stat, cham);

            // case when you want to ignore ME42
            if (disableME42 && stat==4 && ring==2) continue;

            CSCMotherboard* tmb = tmb_[endc-1][stat-1][sect-1][subs-1][cham-1].get();

            tmb->setCSCGeometry(csc_g);

            // actual chamber number =/= trigger chamber number
            int chid = CSCTriggerNumbering::chamberFromTriggerLabels(sect, subs, stat, cham);

            // 0th layer means whole chamber.
            CSCDetId detid(endc, stat, ring, chid, 0);

            // Run processors only if chamber exists in geometry.
            if (tmb == nullptr || csc_g->chamber(detid) == nullptr) continue;

            // Skip chambers marked as bad (usually includes most of ME4/2 chambers;
            // also, there's no ME1/a-1/b separation, it's whole ME1/1)
            if (checkBadChambers_ && badChambers->isInBadChamber(detid)) continue;


            // running upgraded ME1/1 TMBs
            if (stat==1 && ring==1 && smartME1aME1b && !runME11ILT_)
            {
              // run the TMB
              CSCMotherboardME11* tmb11 = static_cast<CSCMotherboardME11*>(tmb);
              LogTrace("CSCTriggerPrimitivesBuilder")<<"CSCTriggerPrimitivesBuilder::build in E:"<<endc<<" S:"<<stat<<" R:"<<ring;
              tmb11->run(wiredc,compdc);

              // get all collections
              const std::vector<CSCCorrelatedLCTDigi>& lctV = tmb11->readoutLCTs1b();
              const std::vector<CSCCorrelatedLCTDigi>& lctV1a = tmb11->readoutLCTs1a();
              std::vector<CSCALCTDigi> alctV1a, alctV = tmb11->alct->readoutALCTs();
              const std::vector<CSCCLCTDigi>& clctV = tmb11->clct->readoutCLCTs();
              const std::vector<int> preTriggerBXs = tmb11->clct->preTriggerBXs();
              const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV = tmb11->clct->preTriggerDigis();
              const std::vector<CSCCLCTDigi>& clctV1a = tmb11->clct1a->readoutCLCTs();
              const std::vector<int> preTriggerBXs1a = tmb11->clct1a->preTriggerBXs();
              const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV1a = tmb11->clct1a->preTriggerDigis();

              // perform simple separation of ALCTs into 1/a and 1/b
              // for 'smart' case. Some duplication occurs for WG [10,15]
              std::vector<CSCALCTDigi> tmpV(alctV);
              alctV.clear();
              for (unsigned int al=0; al < tmpV.size(); al++)
              {
                if (tmpV[al].getKeyWG()<=15) alctV1a.push_back(tmpV[al]);
                if (tmpV[al].getKeyWG()>=10) alctV.push_back(tmpV[al]);
              }

              LogTrace("CSCTriggerPrimitivesBuilder")<<"CSCTriggerPrimitivesBuilder:: a="<<alctV.size()<<" c="<<clctV.size()<<" l="<<lctV.size()
                                                     <<"   1a: a="<<alctV1a.size()<<" c="<<clctV1a.size()<<" l="<<lctV1a.size();

              // ME1/b

              if (!(lctV.empty()&&alctV.empty()&&clctV.empty())) {
                LogTrace("L1CSCTrigger")
                  << "CSCTriggerPrimitivesBuilder results in " <<detid;
              }

              // put collections in event
              put(lctV, oc_lct, detid, " ME1b LCT digi");
              put(alctV, oc_alct, detid, " ME1b ALCT digi");
              put(clctV, oc_clct, detid, " ME1b CLCT digi");
              put(pretriggerV, oc_pretrigger, detid, " ME1b CLCT pre-trigger digi");
              put(preTriggerBXs, oc_pretrig, detid, " ME1b CLCT pre-trigger BX");

              // ME1/a

              if (disableME1a) continue;

              CSCDetId detid1a(endc, stat, 4, chid, 0);

              if (!(lctV1a.empty()&&alctV1a.empty()&&clctV1a.empty())){
                LogTrace("L1CSCTrigger") << "CSCTriggerPrimitivesBuilder results in " <<detid1a;
              }

              // put collections in event
              put(lctV1a, oc_lct, detid1a, " ME1a LCT digi");
              put(alctV1a, oc_alct, detid1a, " ME1a ALCT digi");
              put(clctV1a, oc_clct, detid1a, " ME1a CLCT digi");
              put(pretriggerV1a, oc_pretrigger, detid1a, " ME1a CLCT pre-trigger digi");
              put(preTriggerBXs1a, oc_pretrig, detid1a, " ME1a CLCT pre-trigger BX");
            } // upgraded TMB

            // running upgraded ME1/1 TMBs with GEMs
            else if (stat==1 && ring==1 && smartME1aME1b && runME11ILT_)
            {
              // run the TMB
              CSCGEMMotherboardME11* tmb11GEM = static_cast<CSCGEMMotherboardME11*>(tmb);
              tmb11GEM->setCSCGeometry(csc_g);
              tmb11GEM->setGEMGeometry(gem_g);
              LogTrace("CSCTriggerPrimitivesBuilder")<<"CSCTriggerPrimitivesBuilder::build in E:"<<endc<<" S:"<<stat<<" R:"<<ring;
              tmb11GEM->run(wiredc, compdc, gemPads);

              // 0th layer means whole chamber.
              GEMDetId gemId(detid.zendcap(), 1, 1, 0, chid, 0);

              // get the collections
              const std::vector<CSCCorrelatedLCTDigi>& lctV = tmb11GEM->readoutLCTs1b();
              const std::vector<CSCCorrelatedLCTDigi>& lctV1a = tmb11GEM->readoutLCTs1a();
              std::vector<CSCALCTDigi> alctV1a, alctV = tmb11GEM->alct->readoutALCTs();
              const std::vector<CSCCLCTDigi>& clctV = tmb11GEM->clct->readoutCLCTs();
              const std::vector<int>& preTriggerBXs = tmb11GEM->clct->preTriggerBXs();
              const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV = tmb11GEM->clct->preTriggerDigis();
              const std::vector<CSCCLCTDigi>& clctV1a = tmb11GEM->clct1a->readoutCLCTs();
              const std::vector<int>& preTriggerBXs1a = tmb11GEM->clct1a->preTriggerBXs();
              const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV1a = tmb11GEM->clct1a->preTriggerDigis();
              const std::vector<GEMCoPadDigi>& copads = tmb11GEM->coPadProcessor->readoutCoPads();

              // perform simple separation of ALCTs into 1/a and 1/b
              // for 'smart' case. Some duplication occurs for WG [10,15]
              std::vector<CSCALCTDigi> tmpV(alctV);
              alctV.clear();
              for (unsigned int al=0; al < tmpV.size(); al++)
                {
                  if (tmpV[al].getKeyWG()<=15) alctV1a.push_back(tmpV[al]);
                  if (tmpV[al].getKeyWG()>=10) alctV.push_back(tmpV[al]);
                }
              //LogTrace("CSCTriggerPrimitivesBuilder")<<"CSCTriggerPrimitivesBuilder:: a="<<alctV.size()<<" c="<<clctV.size()<<" l="<<lctV.size()
              //  <<"   1a: a="<<alctV1a.size()<<" c="<<clctV1a.size()<<" l="<<lctV1a.size();

              // ME1/b
              if (!(lctV.empty()&&alctV.empty()&&clctV.empty())) {
                LogTrace("L1CSCTrigger")
                  << "CSCTriggerPrimitivesBuilder results in " <<detid;
              }

              // put collections in event
              put(lctV, oc_lct, detid, " ME1b LCT digi");
              put(alctV, oc_alct, detid, " ME1b ALCT digi");
              put(clctV, oc_clct, detid, " ME1b CLCT digi");
              put(pretriggerV, oc_pretrigger, detid, " ME1b CLCT pre-trigger digi");
              put(preTriggerBXs, oc_pretrig, detid, " ME1b CLCT pre-trigger BX");
              put(copads, oc_gemcopad, gemId, " GEM coincidence pad");

              // ME1/a
              if (disableME1a) continue;

              CSCDetId detid1a(endc, stat, 4, chid, 0);

              if (!(lctV1a.empty()&&alctV1a.empty()&&clctV1a.empty())){
                LogTrace("L1CSCTrigger") << "CSCTriggerPrimitivesBuilder results in " <<detid1a;
              }

              // put collections in event
              put(lctV1a, oc_lct, detid1a, " ME1a LCT digi");
              put(alctV1a, oc_alct, detid1a, " ME1a ALCT digi");
              put(clctV1a, oc_clct, detid1a, " ME1a CLCT digi");
              put(pretriggerV1a, oc_pretrigger, detid1a, " ME1a CLCT pre-trigger digi");
              put(preTriggerBXs1a, oc_pretrig, detid1a, " ME1a CLCT pre-trigger BX");
            }

            // running upgraded ME2/1 TMBs
            else if (stat==2 && ring==1 && runME21ILT_)
            {
              // run the TMB
              CSCGEMMotherboardME21* tmb21GEM = static_cast<CSCGEMMotherboardME21*>(tmb);
              tmb21GEM->setCSCGeometry(csc_g);
              tmb21GEM->setGEMGeometry(gem_g);
              tmb21GEM->run(wiredc, compdc, gemPads);

              // 0th layer means whole chamber.
              GEMDetId gemId(detid.zendcap(), 1, 2, 0, chid, 0);

              // get the collections
              const std::vector<CSCCorrelatedLCTDigi>& lctV = tmb21GEM->readoutLCTs();
              const std::vector<CSCALCTDigi>& alctV = tmb21GEM->alct->readoutALCTs();
              const std::vector<CSCCLCTDigi>& clctV = tmb21GEM->clct->readoutCLCTs();
              const std::vector<int>& preTriggerBXs = tmb21GEM->clct->preTriggerBXs();
              const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV = tmb21GEM->clct->preTriggerDigis();
              const std::vector<GEMCoPadDigi>& copads = tmb21GEM->coPadProcessor->readoutCoPads();

              if (!(alctV.empty() && clctV.empty() && lctV.empty())) {
                LogTrace("L1CSCTrigger")
                  << "CSCTriggerPrimitivesBuilder got results in " <<detid;
              }

              // put collections in event
              put(lctV, oc_lct, detid, " ME21 LCT digi");
              put(alctV, oc_alct, detid, " ME21 ALCT digi");
              put(clctV, oc_clct, detid, " ME21 CLCT digi");
              put(pretriggerV, oc_pretrigger, detid, " ME21 CLCT pre-trigger digi");
              put(preTriggerBXs, oc_pretrig, detid, " ME21 CLCT pre-trigger BX");
              put(copads, oc_gemcopad, gemId, " GEM coincidence pad");
            }
            // running upgraded ME3/1-ME4/1 TMBs
            else if ((stat==3 or stat==4) && ring==1 && runME3141ILT_)
            {
              // run the TMB
              CSCMotherboardME3141* tmb3141 = static_cast<CSCMotherboardME3141*>(tmb);
              tmb3141->setCSCGeometry(csc_g);
              tmb3141->run(wiredc, compdc);

              // get the collections
              const std::vector<CSCCorrelatedLCTDigi>& lctV = tmb3141->readoutLCTs();
              const std::vector<CSCALCTDigi>& alctV = tmb3141->alct->readoutALCTs();
              const std::vector<CSCCLCTDigi>& clctV = tmb3141->clct->readoutCLCTs();
              const std::vector<int>& preTriggerBXs = tmb3141->clct->preTriggerBXs();
              const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV = tmb3141->clct->preTriggerDigis();

              if (!(alctV.empty() && clctV.empty() && lctV.empty())) {
                LogTrace("L1CSCTrigger")
                  << "CSCTriggerPrimitivesBuilder got results in " <<detid;
              }

              // put collections in event
              put(lctV, oc_lct, detid, " ME21 LCT digi");
              put(alctV, oc_alct, detid, " ME21 ALCT digi");
              put(clctV, oc_clct, detid, " ME21 CLCT digi");
              put(pretriggerV, oc_pretrigger, detid, " ME21 CLCT pre-trigger digi");
              put(preTriggerBXs, oc_pretrig, detid, " ME21 CLCT pre-trigger BX");
            }

            // running non-upgraded TMB
            else
            {
              // run the TMB
              tmb->run(wiredc,compdc);

              // get the collections
              const std::vector<CSCCorrelatedLCTDigi>& lctV = tmb->readoutLCTs();
              const std::vector<CSCALCTDigi>& alctV = tmb->alct->readoutALCTs();
              const std::vector<CSCCLCTDigi>& clctV = tmb->clct->readoutCLCTs();
              const std::vector<int>& preTriggerBXs = tmb->clct->preTriggerBXs();
              const std::vector<CSCCLCTPreTriggerDigi>& pretriggerV = tmb->clct->preTriggerDigis();

              if (!(alctV.empty() && clctV.empty() && lctV.empty())) {
                LogTrace("L1CSCTrigger")
                  << "CSCTriggerPrimitivesBuilder got results in " <<detid;
              }

              // put collections in event
              const std::string chamberString("ME" + std::to_string(stat) + "" + std::to_string(ring) + " ");
              put(lctV, oc_lct, detid, chamberString + " LCT digi");
              put(alctV, oc_alct, detid, chamberString + " ALCT digi");
              put(clctV, oc_clct, detid, chamberString + " CLCT digi");
              put(pretriggerV, oc_pretrigger, detid, chamberString + " CLCT pre-trigger digi");
              put(preTriggerBXs, oc_pretrig, detid, chamberString + " CLCT pre-trigger BX");
            } // non-upgraded TMB
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

  // now convert csctf::TrackStub back into CSCCorrelatedLCTDigi
  // put MPC stubs into the event
  std::vector<csctf::TrackStub>::const_iterator itr = result.begin();
  for (; itr != result.end(); itr++)
  {
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
