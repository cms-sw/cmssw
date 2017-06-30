#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "CaloLayer1Unpacker.h"
#include "CaloLayer1Collections.h"

namespace l1t {
namespace stage2 {
   bool
   CaloLayer1Unpacker::unpack(const Block& block, UnpackerCollections *coll)
   {
      LogDebug("L1T") << "Block size = " << block.header().getSize();
      LogDebug("L1T") << "Board ID = " << block.amc().getBoardID();

      auto res = static_cast<CaloLayer1Collections*>(coll);

      auto ctp7_phi = block.amc().getBoardID();
      uint32_t * ptr = &*block.payload().begin();

      int cEta = 19;
      int cPhi = 1;
      int iEta = 19;
      EcalTriggerPrimitiveSample sample(0x01); 
      int zSide = cEta / ((int) iEta);
      const EcalSubdetector ecalTriggerTower = (iEta > 17 ) ? EcalSubdetector::EcalEndcap : EcalSubdetector::EcalBarrel;
      EcalTrigTowerDetId id(zSide, ecalTriggerTower, iEta, cPhi);
      EcalTriggerPrimitiveDigi tpg(id);
      tpg.setSize(1);
      tpg.setSample(0, sample);
      res->getEcalDigis()->push_back(tpg);


      return true;
   }
}
}

DEFINE_L1T_UNPACKER(l1t::stage2::CaloLayer1Unpacker);
