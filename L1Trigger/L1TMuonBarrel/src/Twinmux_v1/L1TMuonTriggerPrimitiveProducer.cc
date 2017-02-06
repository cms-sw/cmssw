//
// Class: L1TMuonTriggerPrimitiveProducer
//
// Info: This producer runs the subsystem collectors for each subsystem
//       specified in its configuration file.
//       It produces a concatenated common trigger primitive list
//       as well as the trigger primitives for each subsystem.
//
// Author: L. Gray (FNAL)
// Modified for BMTF-TwinMux: G. Flouris

#include <memory>
#include <map>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include "L1Trigger/L1TMuon/interface/deprecate/GeometryTranslator.h"
#include "L1Trigger/L1TMuonBarrel/src/Twinmux_v1/DTCollector.h"
#include "L1Trigger/L1TMuonBarrel/src/Twinmux_v1/RPCCollector.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"

using namespace L1TwinMux;
using namespace L1TMuon;

using namespace edm;
using namespace std;

inline void L1TMuonTPPproducer(
                                edm::Handle<L1MuDTChambPhContainer> phiDigis,
                                edm::Handle<L1MuDTChambThContainer> thetaDigis,
                                edm::Handle<RPCDigiCollection> rpcDigis,
                                TriggerPrimitiveCollection *master_out,
                                const edm::EventSetup& es) {

  std::unique_ptr<GeometryTranslator> geom;
  geom.reset(new GeometryTranslator());
  geom->checkAndUpdateGeometry(es);



    double eta,phi,bend;
    std::auto_ptr<TriggerPrimitiveCollection> subs_out(new TriggerPrimitiveCollection);
    std::auto_ptr<DTCollector> dtcolltr(new DTCollector);
    dtcolltr->extractPrimitives(phiDigis, thetaDigis, *subs_out);
    auto the_tp = subs_out->begin();
    auto tp_end   = subs_out->end();
    for ( ; the_tp != tp_end; ++the_tp ) {
      eta  = geom->calculateGlobalEta(*the_tp);
      phi  = geom->calculateGlobalPhi(*the_tp);
      bend = geom->calculateBendAngle(*the_tp);
      the_tp->setCMSGlobalEta(eta);
      the_tp->setCMSGlobalPhi(phi);
      the_tp->setThetaBend(bend);
    }
    std::auto_ptr<RPCCollector> rpccolltr(new RPCCollector);
    rpccolltr->extractPrimitives(rpcDigis,*subs_out);
    the_tp = subs_out->begin();
    tp_end   = subs_out->end();
    for ( ; the_tp != tp_end; ++the_tp ) {
      eta  = geom->calculateGlobalEta(*the_tp);
      phi  = geom->calculateGlobalPhi(*the_tp);
      bend = geom->calculateBendAngle(*the_tp);
      the_tp->setCMSGlobalEta(eta);
      the_tp->setCMSGlobalPhi(phi);
      the_tp->setThetaBend(bend);
    }

    master_out->insert(master_out->end(),
		       subs_out->begin(),
		       subs_out->end());

  return;
}

