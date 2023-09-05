#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "L1TObjectCollections.h"

#include "L1TStage2Layer2Constants.h"
#include "ZDCUnpacker.h"

namespace l1t {
  namespace stage2 {
    ZDCUnpacker::ZDCUnpacker() : EtSumCopy_(0) {}

    bool ZDCUnpacker::unpack(const Block& block, UnpackerCollections* coll) {
      using namespace l1t::stage2::layer2;

      LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

      int nBX = int(
          ceil(block.header().getSize() / demux::nOutputFramePerBX));  // Since there 6 frames per demux output event
      // expect the first four frames to be the first 4 EtSum objects reported per event (see CMS IN-2013/005)

      // Find the central, first and last BXs
      int firstBX = -(ceil((double)nBX / 2.) - 1);
      int lastBX;
      if (nBX % 2 == 0) {
        lastBX = ceil((double)nBX / 2.);
      } else {
        lastBX = ceil((double)nBX / 2.) - 1;
      }

      auto res_ = static_cast<L1TObjectCollections*>(coll)->getEtSums(EtSumCopy_);
      res_->setBXRange(firstBX, lastBX);

      LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

      // Loop over multiple BX and fill EtSums collection
      for (int bx = firstBX; bx <= lastBX; bx++) {
        // ZDC -
        int iFrame = (bx - firstBX) * demux::nOutputFramePerBX;

        uint32_t raw_data = block.payload().at(iFrame +1); // ZDC - info is found on frame 1 of each bx
        
        l1t::EtSum zdcm{l1t::EtSum::kZDCM};
        zdcm.setHwPt(raw_data & 0xFFFF);
        zdcm.setP4(l1t::CaloTools::p4Demux(&zdcm));

        LogDebug("L1T") << "ZDC -: pT " << zdcm.hwPt() << " bx " << bx;

        res_->push_back(bx, zdcm);
        
        // ZDC +
        raw_data = block.payload().at(iFrame +2); // ZDC + info is found on frame 2 of each bx

        l1t::EtSum zdcp{l1t::EtSum::kZDCP};
        zdcp.setHwPt(raw_data & 0xFFFF);
        zdcp.setP4(l1t::CaloTools::p4Demux(&zdcp));

        LogDebug("L1T") << "ZDC +: pT " << zdcp.hwPt() << " bx " << bx;

        res_->push_back(bx, zdcp);

      }

      return true;
    }
  }  // namespace stage2
}  // namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::ZDCUnpacker);
