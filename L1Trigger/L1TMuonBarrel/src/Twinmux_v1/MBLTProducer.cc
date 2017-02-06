//Authors:
// Carlo Battilana - Giuseppe Codispoti
// framework include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include <FWCore/Framework/interface/ConsumesCollector.h>

#include "L1Trigger/L1TMuon/interface/deprecate/MuonTriggerPrimitive.h"
#include "L1Trigger/L1TMuonBarrel/src/Twinmux_v1/MBLTCollection.h"
#include "L1Trigger/L1TMuonBarrel/src/Twinmux_v1/MBLTCollectionFwd.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
using namespace L1TwinMux;


inline std::shared_ptr<MBLTContainer> MBLTProducer( TriggerPrimitiveCollection* tps )
{

  double _maxDeltaPhi = 0.05;

  std::shared_ptr<MBLTContainer> out ( new MBLTContainer );
  MBLTContainer & tracksMap = *out;

  auto tp = tps->cbegin();
  auto tpbeg = tps->cbegin();
  auto tpend = tps->cend();
  for( ; tp != tpend; ++tp ) {

    DTChamberId key;

    TriggerPrimitive::subsystem_type type = tp->subsystem();
    switch ( type ) {

    case TriggerPrimitive::kDT :
      key = tp->detId<DTChamberId>();
      break;

    case TriggerPrimitive::kRPC : {
      if ( tp->detId<RPCDetId>().region() ) continue; // endcap
      int station = tp->detId<RPCDetId>().station();
      int sector  = tp->detId<RPCDetId>().sector();
      int wheel = tp->detId<RPCDetId>().ring();
      key = DTChamberId( wheel, station, sector );
      break;
    }

    default : continue;
    }

    if ( out->find( key ) == out->end() ) {
      tracksMap[key] = MBLTCollection( key );
    }

    TriggerPrimitiveRef tpref(tps, tp - tpbeg);
    tracksMap[key].addStub( tpref );
  }


  MBLTContainer::iterator st = out->begin();
  MBLTContainer::iterator stend = out->end();
  for ( ; st != stend; ++st ) st->second.associate( _maxDeltaPhi );

  return out;

}


