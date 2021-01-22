#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoLocalMuon/RPCRecHit/interface/TracktoRPC.h"
#include "RecoLocalMuon/RPCRecHit/src/DTStationIndex.h"
#include "RecoLocalMuon/RPCRecHit/src/DTObjectMap.h"
#include "RecoLocalMuon/RPCRecHit/src/CSCStationIndex.h"
#include "RecoLocalMuon/RPCRecHit/src/CSCObjectMap.h"

#include <ctime>
#include <TMath.h>

bool TracktoRPC::ValidRPCSurface(RPCDetId rpcid, LocalPoint LocalP, const RPCGeometry *rpcGeo) {
  const GeomDet *whichdet3 = rpcGeo->idToDet(rpcid.rawId());
  const RPCRoll *aroll = dynamic_cast<const RPCRoll *>(whichdet3);
  float locx = LocalP.x(), locy = LocalP.y();  //, locz=LocalP.z();
  if (aroll->isBarrel()) {
    const Bounds &rollbound = rpcGeo->idToDet((rpcid))->surface().bounds();
    float boundlength = rollbound.length();
    float boundwidth = rollbound.width();

    if (fabs(locx) < boundwidth / 2 && fabs(locy) < boundlength / 2 && locy > -boundlength / 2)
      return true;
    else
      return false;

  } else if (aroll->isForward()) {
    const Bounds &rollbound = rpcGeo->idToDet((rpcid))->surface().bounds();
    float boundlength = rollbound.length();
    float boundwidth = rollbound.width();

    float nminx = TMath::Pi() * (18 * boundwidth / TMath::Pi() - boundlength) / 18;
    float ylimit = ((boundlength) / (boundwidth / 2 - nminx / 2)) * fabs(locx) + boundlength / 2 -
                   ((boundlength) / (boundwidth / 2 - nminx / 2)) * (boundwidth / 2);
    if (ylimit < -boundlength / 2)
      ylimit = -boundlength / 2;

    if (fabs(locx) < boundwidth / 2 && fabs(locy) < boundlength / 2 && locy > ylimit)
      return true;
    else
      return false;
  } else
    return false;
}

TracktoRPC::TracktoRPC(const edm::ParameterSet &iConfig, const edm::InputTag &tracklabel, edm::ConsumesCollector iC)
    : rpcGeoToken_(iC.esConsumes()),
      dtGeoToken_(iC.esConsumes()),
      dtMapToken_(iC.esConsumes()),
      cscGeoToken_(iC.esConsumes()),
      cscMapToken_(iC.esConsumes()),
      propagatorToken_(iC.esConsumes(edm::ESInputTag("", "SteppingHelixPropagatorAny"))) {
  if (tracklabel.label().find("cosmic") == 0)
    theTrackTransformer = std::make_unique<TrackTransformerForCosmicMuons>(iConfig);
  else if (tracklabel.label().find("globalCosmic") == 0)
    theTrackTransformer = std::make_unique<TrackTransformerForCosmicMuons>(iConfig);
  else
    theTrackTransformer = std::make_unique<TrackTransformer>(iConfig, iC);
}

std::unique_ptr<RPCRecHitCollection> TracktoRPC::thePoints(reco::TrackCollection const *alltracks,
                                                           edm::EventSetup const &iSetup,
                                                           bool debug) {
  auto _ThePoints = std::make_unique<RPCRecHitCollection>();
  // if(alltracks->empty()) return;

  theTrackTransformer->setServices(iSetup);

  const RPCGeometry *rpcGeo = &iSetup.getData(rpcGeoToken_);
  const DTGeometry *dtGeo = &iSetup.getData(dtGeoToken_);
  const DTObjectMap *dtMap = &iSetup.getData(dtMapToken_);
  const CSCGeometry *cscGeo = &iSetup.getData(cscGeoToken_);
  const CSCObjectMap *cscMap = &iSetup.getData(cscMapToken_);
  const Propagator *propagator = &iSetup.getData(propagatorToken_);

  edm::OwnVector<RPCRecHit> RPCPointVector;
  std::vector<uint32_t> rpcput;
  double MaxD = 999.;

  for (TrackCollection::const_iterator track = alltracks->begin(); track != alltracks->end(); track++) {
    Trajectories trajectories = theTrackTransformer->transform(*track);
    if (debug)
      std::cout << "Building Trajectory from Track. " << std::endl;

    std::vector<uint32_t> rpcrolls;
    std::vector<uint32_t> rpcrolls2;
    std::map<uint32_t, int> rpcNdtcsc;
    std::map<uint32_t, int> rpcrollCounter;

    float tInX = track->innerPosition().X(), tInY = track->innerPosition().Y(), tInZ = track->innerPosition().Z();
    float tOuX = track->outerPosition().X(), tOuY = track->outerPosition().Y(), tOuZ = track->outerPosition().Z();
    if (tInX > tOuX) {
      float temp = tOuX;
      tOuX = tInX;
      tInX = temp;
    }
    if (tInY > tOuY) {
      float temp = tOuY;
      tOuY = tInY;
      tInY = temp;
    }
    if (tInZ > tOuZ) {
      float temp = tOuZ;
      tOuZ = tInZ;
      tInZ = temp;
    }

    if (debug)
      std::cout << "in (x,y,z): (" << tInX << ", " << tInY << ", " << tInZ << ")" << std::endl;
    if (debug)
      std::cout << "out (x,y,z): (" << tOuX << ", " << tOuY << ", " << tOuZ << ")" << std::endl;

    if (debug)
      std::cout << "1. Search expeted RPC roll detid !!" << std::endl;
    for (trackingRecHit_iterator hit = track->recHitsBegin(); hit != track->recHitsEnd(); hit++) {
      if ((*hit)->isValid()) {
        DetId id = (*hit)->geographicalId();

        if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::DT) {
          const GeomDet *geomDet = dtGeo->idToDet((*hit)->geographicalId());
          const DTLayer *dtlayer = dynamic_cast<const DTLayer *>(geomDet);
          if (dtlayer)
            for (Trajectories::const_iterator trajectory = trajectories.begin(); trajectory != trajectories.end();
                 ++trajectory) {
              const BoundPlane &DTSurface = dtlayer->surface();
              const GlobalPoint dcPoint = DTSurface.toGlobal(LocalPoint(0., 0., 0.));

              TrajectoryMeasurement tMt = trajectory->closestMeasurement(dcPoint);
              const TrajectoryStateOnSurface &upd2 = (tMt).updatedState();
              if (upd2.isValid()) {
                LocalPoint trajLP = upd2.localPosition();
                LocalPoint trackLP = (*hit)->localPosition();
                float dx = trajLP.x() - trackLP.x(), dy = trajLP.y() - trackLP.y();  //, dz=trajLP.z()-trackLP.z();
                if (dx > 10. && dy > 10.)
                  continue;

                DTChamberId dtid(geomDet->geographicalId().rawId());
                int dtW = dtid.wheel(), dtS = dtid.sector(), dtT = dtid.station();
                if (dtS == 13)
                  dtS = 4;
                if (dtS == 14)
                  dtS = 10;
                DTStationIndex theindex(0, dtW, dtS, dtT);
                const std::set<RPCDetId> &rollsForThisDT = dtMap->getRolls(theindex);
                for (std::set<RPCDetId>::iterator iteraRoll = rollsForThisDT.begin(); iteraRoll != rollsForThisDT.end();
                     iteraRoll++) {
                  const RPCRoll *rollasociated = rpcGeo->roll(*iteraRoll);

                  TrajectoryStateOnSurface ptss =
                      propagator->propagate(upd2, rpcGeo->idToDet(rollasociated->id())->surface());
                  if (ptss.isValid())
                    if (ValidRPCSurface(rollasociated->id().rawId(), ptss.localPosition(), rpcGeo)) {
                      rpcrollCounter[rollasociated->id().rawId()]++;
                      bool check = true;
                      std::vector<uint32_t>::iterator rpcroll;
                      for (rpcroll = rpcrolls.begin(); rpcroll < rpcrolls.end(); rpcroll++)
                        if (rollasociated->id().rawId() == *rpcroll)
                          check = false;
                      if (check == true) {
                        rpcrolls.push_back(rollasociated->id().rawId());
                        RPCGeomServ servId(rollasociated->id().rawId());
                        if (debug)
                          std::cout << "1\t Barrel RPC roll" << rollasociated->id().rawId() << " "
                                    << servId.name().c_str() << std::endl;
                      }
                    }
                }
              }
            }
        } else if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC) {
          const GeomDet *geomDet = cscGeo->idToDet((*hit)->geographicalId());
          const CSCLayer *csclayer = dynamic_cast<const CSCLayer *>(geomDet);

          CSCDetId cscid(geomDet->geographicalId().rawId());
          if (csclayer)
            for (Trajectories::const_iterator trajectory = trajectories.begin(); trajectory != trajectories.end();
                 ++trajectory) {
              const BoundPlane &CSCSurface = csclayer->surface();
              const GlobalPoint dcPoint = CSCSurface.toGlobal(LocalPoint(0., 0., 0.));

              TrajectoryMeasurement tMt = trajectory->closestMeasurement(dcPoint);
              const TrajectoryStateOnSurface &upd2 = (tMt).updatedState();

              if (upd2.isValid() && cscid.station() != 4 && cscid.ring() != 1) {
                LocalPoint trajLP = upd2.localPosition();
                LocalPoint trackLP = (*hit)->localPosition();
                float dx = trajLP.x() - trackLP.x(), dy = trajLP.y() - trackLP.y();  //, dz=trajLP.z()-trackLP.z();
                if (dx > 10. && dy > 10.)
                  continue;

                int En = cscid.endcap(), St = cscid.station(), Ri = cscid.ring();
                int rpcSegment = cscid.chamber();
                if (En == 2)
                  En = -1;
                if (Ri == 4)
                  Ri = 1;

                CSCStationIndex theindex(En, St, Ri, rpcSegment);
                const std::set<RPCDetId> &rollsForThisCSC = cscMap->getRolls(theindex);
                for (std::set<RPCDetId>::iterator iteraRoll = rollsForThisCSC.begin();
                     iteraRoll != rollsForThisCSC.end();
                     iteraRoll++) {
                  const RPCRoll *rollasociated = rpcGeo->roll(*iteraRoll);

                  TrajectoryStateOnSurface ptss =
                      propagator->propagate(upd2, rpcGeo->idToDet(rollasociated->id())->surface());
                  if (ptss.isValid())
                    if (ValidRPCSurface(rollasociated->id().rawId(), ptss.localPosition(), rpcGeo)) {
                      rpcrollCounter[rollasociated->id().rawId()]++;
                      bool check = true;
                      std::vector<uint32_t>::iterator rpcroll;
                      for (rpcroll = rpcrolls.begin(); rpcroll < rpcrolls.end(); rpcroll++)
                        if (rollasociated->id().rawId() == *rpcroll)
                          check = false;
                      if (check == true) {
                        rpcrolls.push_back(rollasociated->id().rawId());
                        RPCGeomServ servId(rollasociated->id().rawId());
                        if (debug)
                          std::cout << "1\t Forward RPC roll" << rollasociated->id().rawId() << " "
                                    << servId.name().c_str() << std::endl;
                      }
                    }
                }
              }
            }
        } else {
          if (debug)
            std::cout << "1\t The hit is not DT/CSC's.   " << std::endl;
        }
      }
    }
    if (debug)
      std::cout << "First step OK!!\n2. Search nearest DT/CSC sufrace!!" << std::endl;
    std::vector<uint32_t>::iterator rpcroll;
    for (rpcroll = rpcrolls.begin(); rpcroll < rpcrolls.end(); rpcroll++) {
      RPCDetId rpcid(*rpcroll);
      const GlobalPoint &rGP = rpcGeo->idToDet(*rpcroll)->surface().toGlobal(LocalPoint(0, 0, 0));
      RPCGeomServ servId(rpcid);
      int rEn = rpcid.region(), rSe = rpcid.sector(), rWr = rpcid.ring(), rSt = rpcid.station(), rCh = servId.segment();

      if (rpcrollCounter[*rpcroll] < 3)
        continue;

      uint32_t dtcscid = 0;
      double distance = MaxD;

      // if(rSt ==2 && rEn==0) MaxD=100;
      // else if(rSt ==3 && rEn==0) MaxD=100;
      // else if(rSt ==4 && rEn==0) MaxD =150;
      for (trackingRecHit_iterator hit = track->recHitsBegin(); hit != track->recHitsEnd(); hit++) {
        if ((*hit)->isValid()) {
          DetId id = (*hit)->geographicalId();
          if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::DT) {
            const GeomDet *geomDet = dtGeo->idToDet((*hit)->geographicalId());
            //const DTLayer *dtlayer = dynamic_cast<const DTLayer *>(geomDet);
            const GlobalPoint &dtGP = dtGeo->idToDet((*hit)->geographicalId())->surface().toGlobal(LocalPoint(0, 0, 0));
            double dx = rGP.x() - dtGP.x(), dy = rGP.y() - dtGP.y(), dz = rGP.z() - dtGP.z();
            double distanceN = sqrt(dx * dx + dy * dy + dz * dz);

            DTChamberId dtid(geomDet->geographicalId().rawId());
            int Se = dtid.sector(), Wh = dtid.wheel(), St = dtid.station();
            if (Se == 13)
              Se = 4;
            if (Se == 14)
              Se = 10;

            if (rEn == 0 && (rSe - Se) == 0 && (rWr - Wh) == 0 && (rSt - St) == 0 && distanceN < distance) {
              dtcscid = geomDet->geographicalId().rawId();
              distance = distanceN;
              if (debug)
                std::cout << "2\t DT " << dtcscid << " Wheel : " << Wh << " station : " << St << " sector : " << Se
                          << std::endl;
            }
          } else if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC) {
            const GeomDet *geomDet = cscGeo->idToDet((*hit)->geographicalId());
            //const CSCLayer *csclayer = dynamic_cast<const CSCLayer *>(geomDet);
            const GlobalPoint &cscGP =
                cscGeo->idToDet((*hit)->geographicalId())->surface().toGlobal(LocalPoint(0, 0, 0));
            double dx = rGP.x() - cscGP.x(), dy = rGP.y() - cscGP.y(), dz = rGP.z() - cscGP.z();
            double distanceN = sqrt(dx * dx + dy * dy + dz * dz);

            CSCDetId cscid(geomDet->geographicalId().rawId());
            int En = cscid.endcap(), Ri = cscid.ring(), St = cscid.station(), Ch = cscid.chamber();
            if (En == 2)
              En = -1;
            if (Ri == 4)
              Ri = 1;

            if ((rEn - En) == 0 && (rSt - St) == 0 && (Ch - rCh) == 0 && rWr != 1 && rSt != 4 && distanceN < distance) {
              dtcscid = geomDet->geographicalId().rawId();
              distance = distanceN;
              if (debug)
                std::cout << "2\t CSC " << dtcscid << " region : " << En << " station : " << St << " Ring : " << Ri
                          << " chamber : " << Ch << std::endl;
            }
          }
        }
      }
      if (dtcscid != 0 && distance < MaxD) {
        rpcrolls2.push_back(*rpcroll);
        rpcNdtcsc[*rpcroll] = dtcscid;
      }
    }
    if (debug)
      std::cout << "Second step OK!! \n3. Propagate to RPC from DT/CSC!!" << std::endl;
    //std::map<uint32_t, int> rpcput;
    std::vector<uint32_t>::iterator rpcroll2;
    for (rpcroll2 = rpcrolls2.begin(); rpcroll2 < rpcrolls2.end(); rpcroll2++) {
      bool check = true;
      std::vector<uint32_t>::iterator rpcput_;
      for (rpcput_ = rpcput.begin(); rpcput_ < rpcput.end(); rpcput_++)
        if (*rpcroll2 == *rpcput_)
          check = false;

      if (check == true) {
        uint32_t dtcscid = rpcNdtcsc[*rpcroll2];
        DetId id(dtcscid);
        if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::DT) {
          const GeomDet *geomDet = dtGeo->idToDet(dtcscid);
          const DTLayer *dtlayer = dynamic_cast<const DTLayer *>(geomDet);

          if (dtlayer)
            for (Trajectories::const_iterator trajectory = trajectories.begin(); trajectory != trajectories.end();
                 ++trajectory) {
              const BoundPlane &DTSurface = dtlayer->surface();
              const GlobalPoint dcPoint = DTSurface.toGlobal(LocalPoint(0., 0., 0.));

              TrajectoryMeasurement tMt = trajectory->closestMeasurement(dcPoint);
              const TrajectoryStateOnSurface &upd2 = (tMt).updatedState();
              if (upd2.isValid()) {
                TrajectoryStateOnSurface ptss = propagator->propagate(upd2, rpcGeo->idToDet(*rpcroll2)->surface());
                if (ptss.isValid())
                  if (ValidRPCSurface(*rpcroll2, ptss.localPosition(), rpcGeo)) {
                    float rpcGPX = ptss.globalPosition().x();
                    float rpcGPY = ptss.globalPosition().y();
                    float rpcGPZ = ptss.globalPosition().z();

                    if (tInX > rpcGPX || tOuX < rpcGPX)
                      continue;
                    if (tInY > rpcGPY || tOuY < rpcGPY)
                      continue;
                    if (tInZ > rpcGPZ || tOuZ < rpcGPZ)
                      continue;

                    const GeomDet *geomDet2 = rpcGeo->idToDet(*rpcroll2);
                    const RPCRoll *aroll = dynamic_cast<const RPCRoll *>(geomDet2);
                    const RectangularStripTopology *top_ =
                        dynamic_cast<const RectangularStripTopology *>(&(aroll->topology()));
                    LocalPoint xmin = top_->localPosition(0.);
                    LocalPoint xmax = top_->localPosition((float)aroll->nstrips());
                    float rsize = fabs(xmax.x() - xmin.x());
                    float stripl = top_->stripLength();
                    //float stripw = top_->pitch();
                    float eyr = 1;

                    float locx = ptss.localPosition().x(), locy = ptss.localPosition().y(),
                          locz = ptss.localPosition().z();
                    if (locx < rsize * eyr && locy < stripl * eyr && locz < 1.) {
                      RPCRecHit RPCPoint(*rpcroll2, 0, LocalPoint(locx, locy, locz));

                      RPCGeomServ servId(*rpcroll2);
                      if (debug)
                        std::cout << "3\t Barrel Expected RPC " << servId.name().c_str()
                                  << " \tLocalposition X: " << locx << ", Y: " << locy << " GlobalPosition(x,y,z) ("
                                  << rpcGPX << ", " << rpcGPY << ", " << rpcGPZ << ")" << std::endl;
                      RPCPointVector.clear();
                      RPCPointVector.push_back(RPCPoint);
                      _ThePoints->put(*rpcroll2, RPCPointVector.begin(), RPCPointVector.end());
                      rpcput.push_back(*rpcroll2);
                    }
                  }
              }
            }
        } else if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC) {
          const GeomDet *geomDet4 = cscGeo->idToDet(dtcscid);
          const CSCLayer *csclayer = dynamic_cast<const CSCLayer *>(geomDet4);

          if (csclayer)
            for (Trajectories::const_iterator trajectory = trajectories.begin(); trajectory != trajectories.end();
                 ++trajectory) {
              const BoundPlane &CSCSurface = csclayer->surface();
              const GlobalPoint dcPoint = CSCSurface.toGlobal(LocalPoint(0., 0., 0.));

              TrajectoryMeasurement tMt = trajectory->closestMeasurement(dcPoint);
              const TrajectoryStateOnSurface &upd2 = (tMt).updatedState();
              if (upd2.isValid()) {
                TrajectoryStateOnSurface ptss = propagator->propagate(upd2, rpcGeo->idToDet(*rpcroll2)->surface());
                if (ptss.isValid())
                  if (ValidRPCSurface(*rpcroll2, ptss.localPosition(), rpcGeo)) {
                    float rpcGPX = ptss.globalPosition().x();
                    float rpcGPY = ptss.globalPosition().y();
                    float rpcGPZ = ptss.globalPosition().z();

                    if (tInX > rpcGPX || tOuX < rpcGPX)
                      continue;
                    if (tInY > rpcGPY || tOuY < rpcGPY)
                      continue;
                    if (tInZ > rpcGPZ || tOuZ < rpcGPZ)
                      continue;

                    RPCDetId rpcid(*rpcroll2);
                    const GeomDet *geomDet3 = rpcGeo->idToDet(*rpcroll2);
                    const RPCRoll *aroll = dynamic_cast<const RPCRoll *>(geomDet3);
                    const TrapezoidalStripTopology *top_ =
                        dynamic_cast<const TrapezoidalStripTopology *>(&(aroll->topology()));
                    LocalPoint xmin = top_->localPosition(0.);
                    LocalPoint xmax = top_->localPosition((float)aroll->nstrips());
                    float rsize = fabs(xmax.x() - xmin.x());
                    float stripl = top_->stripLength();
                    //float stripw = top_->pitch();

                    float eyr = 1;
                    ////////////////////////////
                    float locx = ptss.localPosition().x(), locy = ptss.localPosition().y(),
                          locz = ptss.localPosition().z();
                    if (locx < rsize * eyr && locy < stripl * eyr && locz < 1.) {
                      RPCRecHit RPCPoint(*rpcroll2, 0, LocalPoint(locx, locy, locz));
                      RPCGeomServ servId(*rpcroll2);
                      if (debug)
                        std::cout << "3\t Forward Expected RPC " << servId.name().c_str()
                                  << " \tLocalposition X: " << locx << ", Y: " << locy << " GlobalPosition(x,y,z) ("
                                  << rpcGPX << ", " << rpcGPY << ", " << rpcGPZ << ")" << std::endl;
                      RPCPointVector.clear();
                      RPCPointVector.push_back(RPCPoint);
                      _ThePoints->put(*rpcroll2, RPCPointVector.begin(), RPCPointVector.end());
                      rpcput.push_back(*rpcroll2);
                    }
                  }
              }
            }
        }
      }
    }
    if (debug)
      std::cout << "last steps OK!! " << std::endl;
  }

  return _ThePoints;
}
