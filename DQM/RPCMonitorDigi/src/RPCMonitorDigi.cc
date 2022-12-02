#include "DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h"
#include "DQM/RPCMonitorClient/interface/utils.h"
#include "DQM/RPCMonitorClient/interface/RPCNameHelper.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <set>
#include <fmt/format.h>

const std::array<std::string, 3> RPCMonitorDigi::regionNames_ = {{"Endcap-", "Barrel", "Endcap+"}};

RPCMonitorDigi::RPCMonitorDigi(const edm::ParameterSet& pset)
    : counter(0),
      muonRPCEvents_(nullptr),
      NumberOfRecHitMuon_(nullptr),
      NumberOfMuon_(nullptr),
      numberOfDisks_(0),
      numberOfInnerRings_(0) {
  useMuonDigis_ = pset.getUntrackedParameter<bool>("UseMuon", true);
  useRollInfo_ = pset.getUntrackedParameter<bool>("UseRollInfo", false);

  muPtCut_ = pset.getUntrackedParameter<double>("MuonPtCut", 3.0);
  muEtaCut_ = pset.getUntrackedParameter<double>("MuonEtaCut", 1.9);

  subsystemFolder_ = pset.getUntrackedParameter<std::string>("RPCFolder", "RPC");
  globalFolder_ = pset.getUntrackedParameter<std::string>("GlobalFolder", "SummaryHistograms");

  //Parametersets for tokens
  muonLabel_ = consumes<reco::CandidateView>(pset.getParameter<edm::InputTag>("MuonLabel"));
  rpcRecHitLabel_ = consumes<RPCRecHitCollection>(pset.getParameter<edm::InputTag>("RecHitLabel"));
  scalersRawToDigiLabel_ = consumes<DcsStatusCollection>(pset.getParameter<edm::InputTag>("ScalersRawToDigiLabel"));

  noiseFolder_ = pset.getUntrackedParameter<std::string>("NoiseFolder", "AllHits");
  muonFolder_ = pset.getUntrackedParameter<std::string>("MuonFolder", "Muon");

  rpcGeomToken_ = esConsumes<edm::Transition::BeginRun>();
}

void RPCMonitorDigi::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& r, edm::EventSetup const& iSetup) {
  edm::LogInfo("rpcmonitordigi") << "[RPCMonitorDigi]: Begin Run ";

  numberOfInnerRings_ = 4;  // set default value

  std::set<int> disk_set;
  const auto& rpcGeo = iSetup.getData(rpcGeomToken_);

  //loop on geometry to book all MEs
  edm::LogInfo("rpcmonitordigi") << "[RPCMonitorDigi]: Booking histograms per roll. ";
  for (auto ich : rpcGeo.dets()) {
    const RPCChamber* ch = dynamic_cast<const RPCChamber*>(ich);
    if (!ch)
      continue;
    const auto& roles = ch->rolls();

    if (useRollInfo_) {
      for (auto roll : roles) {
        const RPCDetId& rpcId = roll->id();

        //get station and inner ring
        if (rpcId.region() != 0) {
          disk_set.insert(rpcId.station());
          numberOfInnerRings_ = std::min(numberOfInnerRings_, rpcId.ring());
        }

        //booking all histograms
        const std::string nameID = RPCNameHelper::rollName(rpcId);
        if (useMuonDigis_)
          bookRollME(ibooker, rpcId, &rpcGeo, muonFolder_, meMuonCollection[nameID]);
        bookRollME(ibooker, rpcId, &rpcGeo, noiseFolder_, meNoiseCollection[nameID]);
      }
    } else {
      const RPCDetId& rpcId = roles[0]->id();  //any roll would do - here I just take the first one
      const std::string nameID = RPCNameHelper::chamberName(rpcId);
      if (useMuonDigis_)
        bookRollME(ibooker, rpcId, &rpcGeo, muonFolder_, meMuonCollection[nameID]);
      bookRollME(ibooker, rpcId, &rpcGeo, noiseFolder_, meNoiseCollection[nameID]);
      if (rpcId.region() != 0) {
        disk_set.insert(rpcId.station());
        numberOfInnerRings_ = std::min(numberOfInnerRings_, rpcId.ring());
      }
    }
  }  //end loop on geometry to book all MEs

  numberOfDisks_ = disk_set.size();

  //Book
  this->bookRegionME(ibooker, noiseFolder_, regionNoiseCollection);
  this->bookSectorRingME(ibooker, noiseFolder_, sectorRingNoiseCollection);
  this->bookWheelDiskME(ibooker, noiseFolder_, wheelDiskNoiseCollection);

  ibooker.setCurrentFolder(subsystemFolder_ + "/" + noiseFolder_);

  noiseRPCEvents_ = ibooker.book1D("RPCEvents", "RPCEvents", 1, 0.5, 1.5);

  if (useMuonDigis_) {
    this->bookRegionME(ibooker, muonFolder_, regionMuonCollection);
    this->bookSectorRingME(ibooker, muonFolder_, sectorRingMuonCollection);
    this->bookWheelDiskME(ibooker, muonFolder_, wheelDiskMuonCollection);

    ibooker.setCurrentFolder(subsystemFolder_ + "/" + muonFolder_);

    muonRPCEvents_ = ibooker.book1D("RPCEvents", "RPCEvents", 1, 0.5, 1.5);
    NumberOfMuon_ = ibooker.book1D("NumberOfMuons", "Number of Muons", 11, -0.5, 10.5);
    NumberOfRecHitMuon_ = ibooker.book1D("NumberOfRecHitMuons", "Number of RPC RecHits per Muon", 8, -0.5, 7.5);
  }
}

void RPCMonitorDigi::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  counter++;
  edm::LogInfo("rpcmonitordigi") << "[RPCMonitorDigi]: Beginning analyzing event " << counter;

  //Muons
  edm::Handle<reco::CandidateView> muonCands;
  event.getByToken(muonLabel_, muonCands);

  std::map<RPCDetId, std::vector<RPCRecHit> > rechitMuon;

  int numMuons = 0;
  int numRPCRecHit = 0;

  if (muonCands.isValid()) {
    int nStaMuons = muonCands->size();

    for (int i = 0; i < nStaMuons; i++) {
      const reco::Candidate& goodMuon = (*muonCands)[i];
      const reco::Muon* muCand = dynamic_cast<const reco::Muon*>(&goodMuon);

      if (!muCand->isGlobalMuon())
        continue;
      if (muCand->pt() < muPtCut_ || fabs(muCand->eta()) > muEtaCut_)
        continue;
      numMuons++;
      reco::Track muTrack = (*(muCand->outerTrack()));
      std::vector<TrackingRecHitRef> rpcTrackRecHits;
      //loop on mu rechits
      for (trackingRecHit_iterator it = muTrack.recHitsBegin(); it != muTrack.recHitsEnd(); it++) {
        if (!(*it)->isValid())
          continue;
        int muSubDetId = (*it)->geographicalId().subdetId();
        if (muSubDetId == MuonSubdetId::RPC) {
          numRPCRecHit++;
          TrackingRecHit* tkRecHit = (*it)->clone();
          RPCRecHit* rpcRecHit = dynamic_cast<RPCRecHit*>(tkRecHit);
          int detId = (int)rpcRecHit->rpcId();
          if (rechitMuon.find(detId) == rechitMuon.end() || rechitMuon[detId].empty()) {
            std::vector<RPCRecHit> myVect(1, *rpcRecHit);
            rechitMuon[detId] = myVect;
          } else {
            rechitMuon[detId].push_back(*rpcRecHit);
          }
        }
      }  // end loop on mu rechits
    }

    //Fill muon counter
    if (NumberOfMuon_) {
      NumberOfMuon_->Fill(numMuons);
    }

    //Fill rechit counter for muons
    if (NumberOfRecHitMuon_ && numMuons > 0) {
      NumberOfRecHitMuon_->Fill(numRPCRecHit);
    }

    //Fill counter of RPC events with rechits associated in with a muon
    if (muonRPCEvents_ != nullptr && numRPCRecHit > 0) {
      muonRPCEvents_->Fill(1);
    }

    //Perform client operation
    this->performSourceOperation(rechitMuon, muonFolder_);

  } else {
    edm::LogError("rpcmonitordigi") << "[RPCMonitorDigi]: Muons - Product not valid for event" << counter;
  }

  //RecHits
  edm::Handle<RPCRecHitCollection> rpcHits;
  event.getByToken(rpcRecHitLabel_, rpcHits);
  std::map<RPCDetId, std::vector<RPCRecHit> > rechitNoise;

  if (rpcHits.isValid()) {
    //    RPC rec hits NOT associated to a muon
    for (auto rpcRecHitIter = rpcHits->begin(); rpcRecHitIter != rpcHits->end(); rpcRecHitIter++) {
      const RPCRecHit& rpcRecHit = (*rpcRecHitIter);
      int detId = (int)rpcRecHit.rpcId();
      if (rechitNoise.find(detId) == rechitNoise.end() || rechitNoise[detId].empty()) {
        std::vector<RPCRecHit> myVect(1, rpcRecHit);
        rechitNoise[detId] = myVect;
      } else {
        rechitNoise[detId].push_back(rpcRecHit);
      }
    }
  } else {
    edm::LogError("rpcmonitordigi") << "[RPCMonitorDigi]: RPCRecHits - Product not valid for event" << counter;
  }

  //Fill counter for all RPC events
  if (noiseRPCEvents_ != nullptr && !rechitNoise.empty()) {
    noiseRPCEvents_->Fill(1);
  }
  //Perform client operation
  this->performSourceOperation(rechitNoise, noiseFolder_);
}

void RPCMonitorDigi::performSourceOperation(std::map<RPCDetId, std::vector<RPCRecHit> >& recHitMap,
                                            std::string recHittype) {
  edm::LogInfo("rpcmonitordigi") << "[RPCMonitorDigi]: Performing DQM source operations for ";

  if (recHitMap.empty())
    return;

  std::map<std::string, std::map<std::string, MonitorElement*> > meRollCollection;
  std::map<std::string, MonitorElement*> meWheelDisk;
  std::map<std::string, MonitorElement*> meRegion;
  std::map<std::string, MonitorElement*> meSectorRing;

  if (recHittype == muonFolder_) {
    meRollCollection = meMuonCollection;
    meWheelDisk = wheelDiskMuonCollection;
    meRegion = regionMuonCollection;
    meSectorRing = sectorRingMuonCollection;
  } else if (recHittype == noiseFolder_) {
    meRollCollection = meNoiseCollection;
    meWheelDisk = wheelDiskNoiseCollection;
    meRegion = regionNoiseCollection;
    meSectorRing = sectorRingNoiseCollection;
  } else {
    edm::LogWarning("rpcmonitordigi") << "[RPCMonitorDigi]: RecHit type not valid.";
    return;
  }

  int totalNumberOfRecHits[3] = {0, 0, 0};

  //Loop on Rolls
  for (std::map<RPCDetId, std::vector<RPCRecHit> >::const_iterator detIdIter = recHitMap.begin();
       detIdIter != recHitMap.end();
       detIdIter++) {
    RPCDetId detId = (*detIdIter).first;
    RPCGeomServ geoServ(detId);

    //get roll number
    rpcdqm::utils rpcUtils;
    int nr = rpcUtils.detId2RollNr(detId);

    const std::string nameRoll = RPCNameHelper::name(detId, useRollInfo_);

    int region = (int)detId.region();
    int wheelOrDiskNumber;
    std::string wheelOrDiskType;
    int ring = 0;
    int sector = detId.sector();
    int totalRolls = 3;
    int roll = detId.roll();
    if (region == 0) {
      wheelOrDiskType = "Wheel";
      wheelOrDiskNumber = (int)detId.ring();
      int station = detId.station();

      if (station == 1) {
        if (detId.layer() == 1) {
          totalRolls = 2;  //RB1in
        } else {
          totalRolls = 2;  //RB1out
        }
        if (roll == 3)
          roll = 2;  // roll=3 is Forward
      } else if (station == 2) {
        if (detId.layer() == 1) {
          //RB2in
          if (abs(wheelOrDiskNumber) == 2 && roll == 3) {
            roll = 2;  //W -2, +2 RB2in has only 2 rolls
            totalRolls = 2;
          }
        } else {
          //RB2out
          if (abs(wheelOrDiskNumber) != 2 && roll == 3) {
            roll = 2;  //W -1, 0, +1 RB2out has only 2 rolls
            totalRolls = 2;
          }
        }
      } else if (station == 3) {
        totalRolls = 2;  //RB3
        if (roll == 3)
          roll = 2;
      } else {
        totalRolls = 2;  //RB4
        if (roll == 3)
          roll = 2;
      }

    } else {
      wheelOrDiskType = "Disk";
      wheelOrDiskNumber = region * (int)detId.station();
      ring = detId.ring();
    }

    std::vector<RPCRecHit> recHits = (*detIdIter).second;
    const int numberOfRecHits = recHits.size();
    totalNumberOfRecHits[region + 1] += numberOfRecHits;

    std::set<int> bxSet;
    int numDigi = 0;

    std::map<std::string, MonitorElement*> meMap = meRollCollection[nameRoll];

    //Loop on recHits
    std::string tmpName;
    for (std::vector<RPCRecHit>::const_iterator recHitIter = recHits.begin(); recHitIter != recHits.end();
         recHitIter++) {
      const RPCRecHit& recHit = (*recHitIter);

      int bx = recHit.BunchX();
      bxSet.insert(bx);
      int clusterSize = (int)recHit.clusterSize();
      numDigi += clusterSize;
      int firstStrip = recHit.firstClusterStrip();
      int lastStrip = clusterSize + firstStrip - 1;

      // ###################### Roll Level  #################################

      tmpName = "Occupancy_" + nameRoll;
      if (meMap[tmpName]) {
        for (int s = firstStrip; s <= lastStrip; s++) {
          if (useRollInfo_) {
            meMap[tmpName]->Fill(s);
          } else {
            const int nstrips = meMap[tmpName]->getNbinsX() / totalRolls;
            meMap[tmpName]->Fill(s + nstrips * (roll - 1));
          }
        }
      }

      tmpName = "BXDistribution_" + nameRoll;
      if (meMap[tmpName])
        meMap[tmpName]->Fill(bx);

      // ###################### Sector- Ring Level #################################

      tmpName = fmt::format("Occupancy_{}_{}_Sector_{}", wheelOrDiskType, wheelOrDiskNumber, sector);
      if (meSectorRing[tmpName]) {
        for (int s = firstStrip; s <= lastStrip; s++) {  //Loop on digis
          meSectorRing[tmpName]->Fill(s, nr);
        }
      }

      tmpName = fmt::format("ClusterSize_{}_{}_Sector_{}", wheelOrDiskType, wheelOrDiskNumber, sector);
      if (meSectorRing[tmpName]) {
        if (clusterSize >= meSectorRing[tmpName]->getNbinsX())
          meSectorRing[tmpName]->Fill(meSectorRing[tmpName]->getNbinsX(), nr);
        else
          meSectorRing[tmpName]->Fill(clusterSize, nr);
      }

      tmpName = fmt::format("Occupancy_{}_{}_Ring_{}", wheelOrDiskType, wheelOrDiskNumber, ring);
      if (geoServ.segment() > 0 && geoServ.segment() <= 18) {
        tmpName += "_CH01-CH18";
      } else if (geoServ.segment() >= 19) {
        tmpName += "_CH19-CH36";
      }

      if (meSectorRing[tmpName]) {
        for (int s = firstStrip; s <= lastStrip; s++) {  //Loop on digis
          meSectorRing[tmpName]->Fill(s + 32 * (detId.roll() - 1), geoServ.segment());
        }
      }

      tmpName = fmt::format("ClusterSize_{}_{}_Ring_{}", wheelOrDiskType, wheelOrDiskNumber, ring);
      if (geoServ.segment() > 0 && geoServ.segment() <= 9) {
        tmpName += "_CH01-CH09";
      } else if (geoServ.segment() >= 10 && geoServ.segment() <= 18) {
        tmpName += "_CH10-CH18";
      } else if (geoServ.segment() >= 19 && geoServ.segment() <= 27) {
        tmpName += "_CH19-CH27";
      } else if (geoServ.segment() >= 28 && geoServ.segment() <= 36) {
        tmpName += "_CH28-CH36";
      }

      if (meSectorRing[tmpName]) {
        if (clusterSize >= meSectorRing[tmpName]->getNbinsX())
          meSectorRing[tmpName]->Fill(meSectorRing[tmpName]->getNbinsX(),
                                      3 * (geoServ.segment() - 1) + (3 - detId.roll()) + 1);
        else
          meSectorRing[tmpName]->Fill(clusterSize, 3 * (geoServ.segment() - 1) + (3 - detId.roll()) + 1);
      }

      // ###################### Wheel/Disk Level #########################
      if (region == 0) {
        tmpName = fmt::format("1DOccupancy_Wheel_{}", wheelOrDiskNumber);
        if (meWheelDisk[tmpName])
          meWheelDisk[tmpName]->Fill(sector, clusterSize);

        tmpName = fmt::format("Occupancy_Roll_vs_Sector_{}_{}", wheelOrDiskType, wheelOrDiskNumber);
        if (meWheelDisk[tmpName])
          meWheelDisk[tmpName]->Fill(sector, nr, clusterSize);

      } else {
        tmpName = fmt::format("1DOccupancy_Ring_{}", ring);
        if ((meWheelDisk[tmpName])) {
          if (wheelOrDiskNumber > 0) {
            meWheelDisk[tmpName]->Fill(wheelOrDiskNumber + numberOfDisks_, clusterSize);
          } else {
            meWheelDisk[tmpName]->Fill(wheelOrDiskNumber + numberOfDisks_ + 1, clusterSize);
          }
        }

        tmpName = fmt::format("Occupancy_Ring_vs_Segment_{}_{}", wheelOrDiskType, wheelOrDiskNumber);
        if (meWheelDisk[tmpName])
          meWheelDisk[tmpName]->Fill(geoServ.segment(), (ring - 1) * 3 - detId.roll() + 1, clusterSize);
      }

      tmpName = fmt::format("BxDistribution_{}_{}", wheelOrDiskType, wheelOrDiskNumber);
      if (meWheelDisk[tmpName])
        meWheelDisk[tmpName]->Fill(bx);

    }  //end loop on recHits

    tmpName = "BXWithData_" + nameRoll;
    if (meMap[tmpName])
      meMap[tmpName]->Fill(bxSet.size());

    tmpName = "NumberOfClusters_" + nameRoll;
    if (meMap[tmpName])
      meMap[tmpName]->Fill(numberOfRecHits);

    tmpName = "Multiplicity_" + RPCMonitorDigi::regionNames_[region + 1];
    if (meRegion[tmpName])
      meRegion[tmpName]->Fill(numDigi);

    if (region == 0) {
      if (meRegion["Occupancy_for_Barrel"])
        meRegion["Occupancy_for_Barrel"]->Fill(sector, wheelOrDiskNumber, numDigi);
    } else {
      int xbin = wheelOrDiskNumber + numberOfDisks_;
      if (region == -1) {
        xbin = wheelOrDiskNumber + numberOfDisks_ + 1;
      }
      if (meRegion["Occupancy_for_Endcap"]) {
        meRegion["Occupancy_for_Endcap"]->Fill(xbin, ring, numDigi);
      }
    }

    tmpName = "Multiplicity_" + nameRoll;
    if (meMap[tmpName])
      meMap[tmpName]->Fill(numDigi);

  }  //end loop on rolls

  for (int i = 0; i < 3; i++) {
    const std::string tmpName = "NumberOfClusters_" + RPCMonitorDigi::regionNames_[i];
    if (meRegion[tmpName])
      meRegion[tmpName]->Fill(totalNumberOfRecHits[i]);
  }
}
