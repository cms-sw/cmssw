// Orso Iorio, INFN Napoli
//
//This module is a filter based on the number of RPC Hits
//

#include "DPGAnalysis/Skims/src/RPCRecHitFilter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
#include <DataFormats/RPCRecHit/interface/RPCRecHit.h>
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include <FWCore/Utilities/interface/Exception.h>
#include "DataFormats/Provenance/interface/Timestamp.h"

/////Trigger

#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "CondFormats/L1TObjects/interface/L1GtParameters.h"
#include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"

using namespace reco;

typedef std::vector<L1MuRegionalCand> L1MuRegionalCandCollection;

#include <sys/time.h>
#include <algorithm>
#include <memory>
#include <cmath>
#include <cmath>
#include "TFile.h"
#include "TMath.h"
#include "TTree.h"

#include "TDirectory.h"
#include "TFile.h"
#include "TTree.h"
#include <cstdlib>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <memory>
#include <cmath>

using namespace edm;
using namespace reco;
using namespace std;

RPCRecHitFilter::RPCRecHitFilter(const edm::ParameterSet& iConfig)
    : rpcGeomToken_(esConsumes()), trackingGeoToken_(esConsumes()) {
  LogTrace("RPCEffTrackExtrapolation") << "inside constructor" << std::endl;

  RPCDataLabel_ = iConfig.getUntrackedParameter<std::string>("rpcRecHitLabel");
  rpcRecHitToken_ = consumes<RPCRecHitCollection>(edm::InputTag(RPCDataLabel_));

  centralBX_ = iConfig.getUntrackedParameter<int>("CentralBunchCrossing", 0);
  BXWindow_ = iConfig.getUntrackedParameter<int>("BunchCrossingWindow", 9999);

  minHits_ = iConfig.getUntrackedParameter<int>("minimumNumberOfHits", 0);
  hitsInStations_ = iConfig.getUntrackedParameter<int>("HitsInStation", 0);

  Debug_ = iConfig.getUntrackedParameter<bool>("Debug", false);
  Verbose_ = iConfig.getUntrackedParameter<bool>("Verbose", false);

  Barrel_ = iConfig.getUntrackedParameter<bool>("UseBarrel", true);
  EndcapPositive_ = iConfig.getUntrackedParameter<bool>("UseEndcapPositive", true);
  EndcapNegative_ = iConfig.getUntrackedParameter<bool>("UseEndcapNegative", true);

  cosmicsVeto_ = iConfig.getUntrackedParameter<bool>("CosmicsVeto", false);
}

bool RPCRecHitFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<RPCGeometry> rpcGeo = iSetup.getHandle(rpcGeomToken_);
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry = iSetup.getHandle(trackingGeoToken_);

  edm::Handle<RPCRecHitCollection> rpcHits = iEvent.getHandle(rpcRecHitToken_);

  std::map<int, int> numberOfRecHitsBarrel;
  std::map<int, int> numberOfDigisBarrel;
  std::map<int, int> numberOfRecHitsEndcap;
  std::map<int, int> numberOfDigisEndcap;

  std::map<pair<int, int>, std::vector<RPCDetId> > numberOfRecHitsSameWheelSameSector;
  std::map<pair<int, int>, std::vector<RPCDetId> > numberOfDigisSameWheelSameSector;
  std::map<pair<int, int>, std::vector<RPCDetId> > numberOfHitsSameDiskSectorPositive;
  std::map<pair<int, int>, std::vector<RPCDetId> > numberOfHitsSameDiskSectorNegative;

  const std::vector<const RPCRoll*> rls = rpcGeo->rolls();

  bool condition = true;

  int nBarrel = 0;
  int nEndcap = 0;

  /////BEGIN LOOP ON THE ROLLS
  for (int i = 0; i < (int)rls.size(); ++i) {
    RPCDetId did = rls[i]->id();

    /// LOOP OVER THE RECHITS
    RPCRecHitCollection::range rpcRecHitRange = rpcHits->get(did);
    RPCRecHitCollection::const_iterator RecHitsIt;

    for (RecHitsIt = rpcRecHitRange.first; RecHitsIt != rpcRecHitRange.second; ++RecHitsIt) {
      //std::cout<< " roll is "<< did << " bx recHit is " << RecHitsIt->BunchX()<< " event number " <<eventNumber  <<std::endl;

      if (did.region() == 0) {
        if (cosmicsVeto_) {
          for (int u = -1; u <= 1; ++u) {
            for (int t = -1; t <= 1; ++t) {
              numberOfRecHitsSameWheelSameSector[pair<int, int>(did.ring() + u, did.sector() + t)].push_back(did);
            }
          }
        }

        else {
          if (did.station() == 1) {
            for (int u = -1; u <= 1; ++u) {
              for (int t = -1; t <= 1; ++t) {
                numberOfRecHitsSameWheelSameSector[pair<int, int>(did.ring() + u, did.sector() + t)].push_back(did);
              }
            }
          }
        }

        ++numberOfRecHitsBarrel[did.ring()];
        ++nBarrel;

      }

      else {
        if (did.region() == -1) {
          for (int t = -1; t <= 1; ++t) {
            numberOfHitsSameDiskSectorNegative[pair<int, int>(did.ring(), did.sector() + t)].push_back(did);
          }
        }

        if (did.region() == 1) {
          for (int t = -1; t <= 1; ++t) {
            numberOfHitsSameDiskSectorPositive[pair<int, int>(did.ring(), did.sector() + t)].push_back(did);
          }
        }
        ++numberOfRecHitsEndcap[did.station()];

        ++nEndcap;
      }

      break;
    }
  }
  /////END OF THE LOOP ON THE ROLLS

  ////CONDITION IS THAT THERE ARE TWO HITS IN RB1in and RB1out OF TWO ADJACENT SECTORS AND WHEELS
  ////OR THAT IN TWO DISKS SAME OF THE ENDCAP THERE ARE TWO HITS IN ADJACENT SECTORS IN THE SAME RING
  bool cond1 = false;
  bool cond2 = false;
  bool cond3 = false;

  std::map<int, bool> vectorBarrelCands;
  std::map<int, bool> vectorEndcapCandsPositive;
  std::map<int, bool> vectorEndcapCandsNegative;

  // barrel
  for (std::map<pair<int, int>, std::vector<RPCDetId> >::const_iterator iter =
           numberOfRecHitsSameWheelSameSector.begin();
       iter != numberOfRecHitsSameWheelSameSector.end();
       ++iter) {
    vectorBarrelCands[1] = false;
    vectorBarrelCands[2] = false;

    if (iter->second.size() > 1) {
      for (size_t i = 0; i < iter->second.size(); ++i) {
        if (iter->second[i].layer() == 1 && iter->second[i].station() == 1)
          vectorBarrelCands[0] = true;
        if (iter->second[i].layer() == 2 && iter->second[i].station() == 1)
          vectorBarrelCands[1] = true;
        if (cosmicsVeto_)
          if (iter->second[i].station() > 2) {
            vectorBarrelCands[1] = false;
            vectorBarrelCands[2] = false;
            break;
          }
      }
    }

    if ((vectorBarrelCands[0] && vectorBarrelCands[1])) {
      cond1 = true;
      break;
    }
  }

  // endcap positive
  for (std::map<pair<int, int>, std::vector<RPCDetId> >::const_iterator iter =
           numberOfHitsSameDiskSectorPositive.begin();
       iter != numberOfHitsSameDiskSectorPositive.end();
       ++iter) {
    vectorEndcapCandsPositive[1] = false;
    vectorEndcapCandsPositive[2] = false;
    vectorEndcapCandsPositive[3] = false;

    if (iter->second.size() > 1) {
      for (size_t i = 0; i < iter->second.size(); ++i) {
        if (iter->second[i].station() == 1)
          vectorEndcapCandsPositive[1] = true;
        if (iter->second[i].station() == 2)
          vectorEndcapCandsPositive[2] = true;
        if (iter->second[i].station() == 3)
          vectorEndcapCandsPositive[3] = true;
      }
    }

    if (((vectorEndcapCandsPositive[1] && vectorEndcapCandsPositive[2]) ||
         (vectorEndcapCandsPositive[1] && vectorEndcapCandsPositive[3]) ||
         (vectorEndcapCandsPositive[2] && vectorEndcapCandsPositive[3]))) {
      cond2 = true;
      break;
    }
  }

  // endcap negative
  for (std::map<pair<int, int>, std::vector<RPCDetId> >::const_iterator iter =
           numberOfHitsSameDiskSectorNegative.begin();
       iter != numberOfHitsSameDiskSectorNegative.end();
       ++iter) {
    vectorEndcapCandsNegative[1] = false;
    vectorEndcapCandsNegative[2] = false;
    vectorEndcapCandsNegative[3] = false;

    if (iter->second.size() > 1) {
      for (size_t i = 0; i < iter->second.size(); ++i) {
        if (iter->second[i].station() == 1)
          vectorEndcapCandsNegative[1] = true;
        if (iter->second[i].station() == 2)
          vectorEndcapCandsNegative[2] = true;
        if (iter->second[i].station() == 3)
          vectorEndcapCandsNegative[3] = true;
      }
    }

    if (((vectorEndcapCandsNegative[1] && vectorEndcapCandsNegative[2]) ||
         (vectorEndcapCandsNegative[1] && vectorEndcapCandsNegative[3]) ||
         (vectorEndcapCandsNegative[2] && vectorEndcapCandsNegative[3]))) {
      cond3 = true;
      break;
    }
  }

  condition = condition && (nBarrel + nEndcap >= minHits_);

  cond1 = Barrel_ && cond1;
  cond2 = EndcapPositive_ && cond2;
  cond3 = EndcapNegative_ && cond3;

  bool condition2 = (cond1 || cond2 || cond3);
  if (Barrel_ || EndcapPositive_ || EndcapNegative_)
    condition = condition && condition2;

  return condition;
}

DEFINE_FWK_MODULE(RPCRecHitFilter);
