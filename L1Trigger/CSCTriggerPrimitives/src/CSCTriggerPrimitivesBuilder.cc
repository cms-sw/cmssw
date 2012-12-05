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
//   $Id: CSCTriggerPrimitivesBuilder.cc,v 1.21.2.3 2012/10/18 05:04:22 khotilov Exp $
//
//   Modifications:
//
//-----------------------------------------------------------------------------

#include <L1Trigger/CSCTriggerPrimitives/src/CSCTriggerPrimitivesBuilder.h>
#include <L1Trigger/CSCTriggerPrimitives/src/CSCMotherboard.h>
#include <L1Trigger/CSCTriggerPrimitives/src/CSCMotherboardME11.h>
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
  smartME1aME1b = commonParams.getUntrackedParameter<bool>("smartME1aME1b", false);
  disableME1a = commonParams.getUntrackedParameter<bool>("disableME1a", false);
  disableME42 = commonParams.getUntrackedParameter<bool>("disableME42", false);

  checkBadChambers_ = conf.getUntrackedParameter<bool>("checkBadChambers", true);

  // ORCA way of initializing boards.
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
                (cham <= 0 || stat > MAX_CHAMBERS))
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
            if (stat==1 && ring==1 && smartME1aME1b)
              tmb_[endc-1][stat-1][sect-1][subs-1][cham-1] = new CSCMotherboardME11(endc, stat, sect, subs, cham, conf);
            else
              tmb_[endc-1][stat-1][sect-1][subs-1][cham-1] = new CSCMotherboard(endc, stat, sect, subs, cham, conf);
          }
        }
      }
    }
  }

  // Get min and max BX to sort LCTs in MPC.
  m_minBX = conf.getParameter<int>("MinBX");
  m_maxBX = conf.getParameter<int>("MaxBX");

  // Init MPC
  m_muonportcard = new CSCMuonPortCard(conf);
}

//------------
// Destructor
//------------
CSCTriggerPrimitivesBuilder::~CSCTriggerPrimitivesBuilder()
{
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
					CSCCorrelatedLCTDigiCollection& oc_sorted_lct)
{
  // CSC geometry.
  CSCTriggerGeomManager* theGeom = CSCTriggerGeometry::get();

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
            
            int ring = CSCTriggerNumbering::ringFromTriggerLabels(stat, cham);
            
            if (disableME42 && stat==4 && ring==2) continue;

            CSCMotherboard* tmb = tmb_[endc-1][stat-1][sect-1][subs-1][cham-1];

            // Run processors only if chamber exists in geometry.
            if (tmb == 0 || theGeom->chamber(endc, stat, sect, subs, cham) == 0) continue;

            int chid = CSCTriggerNumbering::chamberFromTriggerLabels(sect, subs, stat, cham);

            // 0th layer means whole chamber.
            CSCDetId detid(endc, stat, ring, chid, 0);

            // Skip chambers marked as bad (usually includes most of ME4/2 chambers;
            // also, there's no ME1/a-1/b separation, it's whole ME1/1)
            if (checkBadChambers_ && badChambers->isInBadChamber(detid)) continue;


            // running upgraded ME1/1 TMBs (non-upgraded)
            if (stat==1 && ring==1 && smartME1aME1b)
            {
              CSCMotherboardME11* tmb11 = static_cast<CSCMotherboardME11*>(tmb);
 
              //LogTrace("CSCTriggerPrimitivesBuilder")<<"CSCTriggerPrimitivesBuilder::build in E:"<<endc<<" S:"<<stat<<" R:"<<ring;
 
              tmb11->run(wiredc,compdc);
              std::vector<CSCCorrelatedLCTDigi> lctV = tmb11->readoutLCTs1b();
              std::vector<CSCCorrelatedLCTDigi> lctV1a = tmb11->readoutLCTs1a();
 
              std::vector<CSCALCTDigi> alctV1a, alctV = tmb11->alct->readoutALCTs();

              std::vector<CSCCLCTDigi> clctV = tmb11->clct->readoutCLCTs();
              std::vector<int> preTriggerBXs = tmb11->clct->preTriggerBXs();
              std::vector<CSCCLCTDigi> clctV1a = tmb11->clct1a->readoutCLCTs();
              std::vector<int> preTriggerBXs1a = tmb11->clct1a->preTriggerBXs();

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

              // Correlated LCTs.
              if (!lctV.empty()) {
                LogTrace("L1CSCTrigger")
                  << "Put " << lctV.size() << " ME1b LCT digi"
                  << ((lctV.size() > 1) ? "s " : " ") << "in collection\n";
                oc_lct.put(std::make_pair(lctV.begin(),lctV.end()), detid);
              }
 
              // Anode LCTs.
              if (!alctV.empty()) {
                LogTrace("L1CSCTrigger")
                  << "Put " << alctV.size() << " ME1b ALCT digi"
                  << ((alctV.size() > 1) ? "s " : " ") << "in collection\n";
                oc_alct.put(std::make_pair(alctV.begin(),alctV.end()), detid);
              }
 
              // Cathode LCTs.
              if (!clctV.empty()) {
                LogTrace("L1CSCTrigger")
                  << "Put " << clctV.size() << " ME1b CLCT digi"
                  << ((clctV.size() > 1) ? "s " : " ") << "in collection\n";
                oc_clct.put(std::make_pair(clctV.begin(),clctV.end()), detid);
              }

              // Cathode LCTs pretriggers
              if (!preTriggerBXs.empty()) {
                LogTrace("L1CSCTrigger")
                  << "Put " << preTriggerBXs.size() << " CLCT pretrigger"
                  << ((preTriggerBXs.size() > 1) ? "s " : " ") << "in collection\n";
                oc_pretrig.put(std::make_pair(preTriggerBXs.begin(),preTriggerBXs.end()), detid);
              }            

              // ME1/a

              if (disableME1a) continue;

              CSCDetId detid1a(endc, stat, 4, chid, 0);

              if (!(lctV1a.empty()&&alctV1a.empty()&&clctV1a.empty())){
                LogTrace("L1CSCTrigger") << "CSCTriggerPrimitivesBuilder results in " <<detid1a;
              }
 
              // Correlated LCTs.
              if (!lctV1a.empty()) {
                LogTrace("L1CSCTrigger")
                  << "Put " << lctV1a.size() << " ME1a LCT digi"
                  << ((lctV1a.size() > 1) ? "s " : " ") << "in collection\n";
                oc_lct.put(std::make_pair(lctV1a.begin(),lctV1a.end()), detid1a);
              }
 
              // Anode LCTs.
              if (!alctV1a.empty()) {
                LogTrace("L1CSCTrigger")
                  << "Put " << alctV1a.size() << " ME1a ALCT digi"
                  << ((alctV1a.size() > 1) ? "s " : " ") << "in collection\n";
                oc_alct.put(std::make_pair(alctV1a.begin(),alctV1a.end()), detid1a);
              }
 
              // Cathode LCTs.
              if (!clctV1a.empty()) {
                LogTrace("L1CSCTrigger")
                  << "Put " << clctV1a.size() << " ME1a CLCT digi"
                  << ((clctV1a.size() > 1) ? "s " : " ") << "in collection\n";
                oc_clct.put(std::make_pair(clctV1a.begin(),clctV1a.end()), detid1a);
              }
              
              // Cathode LCTs pretriggers
              if (!preTriggerBXs1a.empty()) {
                LogTrace("L1CSCTrigger")
                  << "Put " << preTriggerBXs.size() << " CLCT pretrigger"
                  << ((preTriggerBXs.size() > 1) ? "s " : " ") << "in collection\n";
                oc_pretrig.put(std::make_pair(preTriggerBXs.begin(),preTriggerBXs.end()), detid);
              }
            } // upgraded TMB

            // running non-upgraded TMB
            else
            {
              tmb->run(wiredc,compdc);

              std::vector<CSCCorrelatedLCTDigi> lctV = tmb->readoutLCTs();
              std::vector<CSCALCTDigi> alctV = tmb->alct->readoutALCTs();
              std::vector<CSCCLCTDigi> clctV = tmb->clct->readoutCLCTs();
              std::vector<int> preTriggerBXs = tmb->clct->preTriggerBXs();

              if (!(alctV.empty() && clctV.empty() && lctV.empty())) {
                LogTrace("L1CSCTrigger")
                  << "CSCTriggerPrimitivesBuilder got results in " <<detid;
              }

              /*
              // tmp kludge: tightening of ME1a LCTs
              if (stat==1 && ring==1) {
                std::vector<CSCCorrelatedLCTDigi> lctV11;
                for (unsigned t=0;t<lctV.size();t++){
                  if (lctV[t].getStrip() < 127) lctV11.push_back(lctV[t]);
                  else if (lctV[t].getQuality() >= 14) lctV11.push_back(lctV[t]);
                }
                lctV = lctV11;
              }
              */

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
                  << "Put " << preTriggerBXs.size() << " CLCT pretrigger"
                  << ((preTriggerBXs.size() > 1) ? "s " : " ") << "in collection\n";
                oc_pretrig.put(std::make_pair(preTriggerBXs.begin(),preTriggerBXs.end()), detid);
              }
            } // non-upgraded TMB
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
