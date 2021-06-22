// -*- C++ -*-
//
// Package:    RPCGEO2
// Class:      RPCGEO2
//
/**\class RPCGEO2 RPCGEO2.cc rpcgeo/RPCGEO2/src/RPCGEO2.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  pts/91
//         Created:  Wed Sep 26 17:08:29 CEST 2007
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include <DataFormats/RPCDigi/interface/RPCDigi.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>

//
// class decleration
//

class RPCGEO2 : public edm::one::EDAnalyzer<> {
public:
  explicit RPCGEO2(const edm::ParameterSet&);
  ~RPCGEO2() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> tokRPC_;
};

RPCGEO2::RPCGEO2(const edm::ParameterSet& /*iConfig*/)
    : tokRPC_{esConsumes<RPCGeometry, MuonGeometryRecord>(edm::ESInputTag{})} {}

RPCGEO2::~RPCGEO2() {}

// ------------ method called to for each event  ------------
void RPCGEO2::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::LogVerbatim("RPCGeometry") << " Getting the RPC Geometry";
  const RPCGeometry* rpcGeo = &iSetup.getData(tokRPC_);

  for (TrackingGeometry::DetContainer::const_iterator it = rpcGeo->dets().begin(); it < rpcGeo->dets().end(); it++) {
    if (dynamic_cast<const RPCChamber*>(*it) != nullptr) {
      const RPCChamber* ch = dynamic_cast<const RPCChamber*>(*it);
      std::vector<const RPCRoll*> roles = (ch->rolls());

      for (std::vector<const RPCRoll*>::const_iterator r = roles.begin(); r != roles.end(); ++r) {
        RPCDetId rpcId = (*r)->id();
        RPCGeomServ rpcsrv(rpcId);
        if (rpcId.region() == 0) {
          const BoundPlane& RPCSurface = (*r)->surface();
          GlobalPoint CenterPointRollGlobal = RPCSurface.toGlobal(LocalPoint(0, 0, 0));
          edm::LogVerbatim("RPCGeometry") << rpcsrv.name() << " " << CenterPointRollGlobal.x() << " "
                                          << CenterPointRollGlobal.y() << " " << CenterPointRollGlobal.z();
          GlobalPoint i = RPCSurface.toGlobal(LocalPoint(1, 0, 0));
          GlobalPoint j = RPCSurface.toGlobal(LocalPoint(0, 1, 0));
          edm::LogVerbatim("RPCGeometry") << " i " << i.x() << " " << i.y() << " " << i.z();
          edm::LogVerbatim("RPCGeometry") << " j " << j.x() << " " << j.y() << " " << j.z();

        } else {
          const BoundPlane& RPCSurface = (*r)->surface();
          GlobalPoint CenterPointRollGlobal = RPCSurface.toGlobal(LocalPoint(0, 0, 0));
          edm::LogVerbatim("RPCGeometry") << rpcsrv.name() << " " << CenterPointRollGlobal.x() << " "
                                          << CenterPointRollGlobal.y() << " " << CenterPointRollGlobal.z();
        }
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCGEO2);
