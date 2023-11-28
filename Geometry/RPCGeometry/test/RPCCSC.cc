// -*- C++ -*-
//
// Package:    RPCCSC
// Class:      RPCCSC
//
/**\class RPCCSC RPCCSC.cc TESTCSCRPC/RPCCSC/src/RPCCSC.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
 */
//
// Original Author:  Haiyun Teng
//         Created:  Wed Feb 25 18:09:15 CET 2009
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCChamber.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

//
// class decleration
//
class CSCStationIndex {
public:
  CSCStationIndex() : _region(0), _station(0), _ring(0), _chamber(0) {}
  CSCStationIndex(int region, int station, int ring, int chamber)
      : _region(region), _station(station), _ring(ring), _chamber(chamber) {}
  ~CSCStationIndex() {}
  int region() const { return _region; }
  int station() const { return _station; }
  int ring() const { return _ring; }
  int chamber() const { return _chamber; }
  bool operator<(const CSCStationIndex& cscind) const {
    if (cscind.region() != this->region())
      return cscind.region() < this->region();
    else if (cscind.station() != this->station())
      return cscind.station() < this->station();
    else if (cscind.ring() != this->ring())
      return cscind.ring() < this->ring();
    else if (cscind.chamber() != this->chamber())
      return cscind.chamber() < this->chamber();
    return false;
  }

private:
  int _region;
  int _station;
  int _ring;
  int _chamber;
};

class RPCCSC : public edm::one::EDAnalyzer<> {
public:
  explicit RPCCSC(const edm::ParameterSet&);
  ~RPCCSC() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> tokRPC_;
  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> tokCSC_;
  std::map<CSCStationIndex, std::set<RPCDetId> > rollstoreCSC;
};

RPCCSC::RPCCSC(const edm::ParameterSet& /*iConfig*/)
    : tokRPC_{esConsumes<RPCGeometry, MuonGeometryRecord>(edm::ESInputTag{})},
      tokCSC_{esConsumes<CSCGeometry, MuonGeometryRecord>(edm::ESInputTag{})} {}

RPCCSC::~RPCCSC() {}

// ------------ method called once each job just before starting event loop  ------------
void RPCCSC::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  using namespace std;
  const RPCGeometry* rpcGeometry = &iSetup.getData(tokRPC_);
  const CSCGeometry* cscGeometry = &iSetup.getData(tokCSC_);

  for (TrackingGeometry::DetContainer::const_iterator it = rpcGeometry->dets().begin(); it < rpcGeometry->dets().end();
       it++) {
    if (dynamic_cast<const RPCChamber*>(*it) != nullptr) {
      const RPCChamber* ch = dynamic_cast<const RPCChamber*>(*it);
      std::vector<const RPCRoll*> roles = (ch->rolls());
      for (std::vector<const RPCRoll*>::const_iterator r = roles.begin(); r != roles.end(); ++r) {
        RPCDetId rpcId = (*r)->id();
        int region = rpcId.region();
        //booking all histograms
        RPCGeomServ rpcsrv(rpcId);
        std::string nameRoll = rpcsrv.name();
        //edm::LogVerbatim("RPCGeometry") << "Booking for " << nameRoll;

        if (region != 0) {
          // 	  const TrapezoidalStripTopology* topE_=dynamic_cast<const TrapezoidalStripTopology*>(&((*r)->topology()));
          // 	  float stripl = topE_->stripLength();
          // 	  float stripw = topE_->pitch();
          int region = rpcId.region();
          int station = rpcId.station();
          int ring = rpcId.ring();
          int cscring = ring;
          int cscstation = station;
          RPCGeomServ rpcsrv(rpcId);
          int rpcsegment = rpcsrv.segment();  //This replace rpcsrv.segment();
          //edm::LogVerbatim("RPCGeometry") << "My segment=" << mySegment(rpcId) << " GeomServ=" << rpcsrv.segment();
          int cscchamber = rpcsegment;                        //FIX THIS ACCORDING TO RPCGeomServ::segment()Definition
          if ((station == 2 || station == 3) && ring == 3) {  //Adding Ring 3 of RPC to the CSC Ring 2
            cscring = 2;
          }

          CSCStationIndex ind(region, cscstation, cscring, cscchamber);
          std::set<RPCDetId> myrolls;
          if (rollstoreCSC.find(ind) != rollstoreCSC.end()) {
            myrolls = rollstoreCSC[ind];
          }
          myrolls.insert(rpcId);
          rollstoreCSC[ind] = myrolls;
        }
      }
    }
  }
  for (TrackingGeometry::DetContainer::const_iterator it = rpcGeometry->dets().begin(); it < rpcGeometry->dets().end();
       it++) {
    if (dynamic_cast<const RPCChamber*>(*it) != nullptr) {
      const RPCChamber* ch = dynamic_cast<const RPCChamber*>(*it);
      std::vector<const RPCRoll*> roles = (ch->rolls());
      for (std::vector<const RPCRoll*>::const_iterator r = roles.begin(); r != roles.end(); ++r) {
        RPCDetId rpcId = (*r)->id();
        int region = rpcId.region();
        if (region != 0 && (rpcId.ring() == 2 || rpcId.ring() == 3)) {
          int region = rpcId.region();
          int station = rpcId.station();
          int ring = rpcId.ring();
          int cscring = ring;

          if ((station == 2 || station == 3) && ring == 3)
            cscring = 2;  //CSC Ring 2 covers rpc ring 2 & 3

          int cscstation = station;
          RPCGeomServ rpcsrv(rpcId);
          int rpcsegment = rpcsrv.segment();

          int cscchamber = rpcsegment + 1;
          if (cscchamber == 37)
            cscchamber = 1;
          CSCStationIndex ind(region, cscstation, cscring, cscchamber);
          std::set<RPCDetId> myrolls;
          if (rollstoreCSC.find(ind) != rollstoreCSC.end())
            myrolls = rollstoreCSC[ind];
          myrolls.insert(rpcId);
          rollstoreCSC[ind] = myrolls;

          cscchamber = rpcsegment - 1;
          if (cscchamber == 0)
            cscchamber = 36;
          CSCStationIndex indDos(region, cscstation, cscring, cscchamber);
          std::set<RPCDetId> myrollsDos;
          if (rollstoreCSC.find(indDos) != rollstoreCSC.end())
            myrollsDos = rollstoreCSC[indDos];
          myrollsDos.insert(rpcId);
          rollstoreCSC[indDos] = myrollsDos;
        }
      }
    }
  }

  //adding more rpcs

  // Now check binding
  const CSCGeometry::ChamberContainer& cscChambers = cscGeometry->chambers();
  for (auto cscChamber : cscChambers) {
    CSCDetId CSCId = cscChamber->id();

    int cscEndCap = CSCId.endcap();
    int cscStation = CSCId.station();
    int cscRing = CSCId.ring();
    //     int cscChamber = CSCId.chamber();
    int rpcRegion = 1;
    if (cscEndCap == 2)
      rpcRegion = -1;  //Relacion entre las endcaps
    int rpcRing = cscRing;
    if (cscRing == 4)
      rpcRing = 1;
    int rpcStation = cscStation;
    int rpcSegment = CSCId.chamber();

    //edm::LogVerbatim("RPCGeometry") << "CSC \t \t Getting chamber from Geometry";
    const CSCChamber* TheChamber = cscGeometry->chamber(CSCId);
    //edm::LogVerbatim("RPCGeometry") << "CSC \t \t Getting ID from Chamber";

    std::set<RPCDetId> rollsForThisCSC = rollstoreCSC[CSCStationIndex(rpcRegion, rpcStation, rpcRing, rpcSegment)];
    if (CSCId.ring() != 1)
      edm::LogVerbatim("RPCGeometry") << "CSC for" << CSCId << " " << rollsForThisCSC.size() << " rolls.";

    for (auto iteraRoll : rollsForThisCSC) {
      const RPCRoll* rollasociated = rpcGeometry->roll(iteraRoll);
      RPCDetId rpcId = rollasociated->id();
      RPCGeomServ rpcsrv(rpcId);

      const BoundPlane& RPCSurface = rollasociated->surface();

      GlobalPoint CenterPointRollGlobal = RPCSurface.toGlobal(LocalPoint(0, 0, 0));
      GlobalPoint CenterPointCSCGlobal = TheChamber->toGlobal(LocalPoint(0, 0, 0));

      //LocalPoint CenterRollinCSCFrame = TheChamber->toLocal(CenterPointRollGlobal);

      float rpcphi = 0;
      float cscphi = 0;

      (CenterPointRollGlobal.barePhi() < 0) ? rpcphi = 2 * 3.141592 + CenterPointRollGlobal.barePhi()
                                            : rpcphi = CenterPointRollGlobal.barePhi();

      (CenterPointCSCGlobal.barePhi() < 0) ? cscphi = 2 * 3.1415926536 + CenterPointCSCGlobal.barePhi()
                                           : cscphi = CenterPointCSCGlobal.barePhi();

      float df = fabs(cscphi - rpcphi);
      float dr = fabs(CenterPointRollGlobal.perp() - CenterPointCSCGlobal.perp());
      float diffz = CenterPointRollGlobal.z() - CenterPointCSCGlobal.z();
      float dfg = df * 180. / 3.14159265;

      edm::LogVerbatim("RPCGeometry") << "CSC \t " << rpcsrv.segment() << rpcsrv.name() << " dr=" << dr
                                      << " dz=" << diffz << " dfg=" << dfg;

      bool print = false;

      if ((dr > 200. || fabs(diffz) > 55. || dfg > 1.) && print) {
        edm::LogVerbatim("RPCGeometry") << "\t \t problem CSC Station= " << CSCId.station() << " Ring= " << CSCId.ring()
                                        << " Chamber= " << CSCId.chamber() << " cscphi=" << cscphi * 180 / 3.14159265
                                        << "\t RPC Station= " << rpcId.station() << " ring= " << rpcId.ring()
                                        << " segment =-> " << rpcsrv.segment()
                                        << " rollphi=" << rpcphi * 180 / 3.14159265 << "\t dfg=" << dfg
                                        << " dz=" << diffz << " dr=" << dr;
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCCSC);
