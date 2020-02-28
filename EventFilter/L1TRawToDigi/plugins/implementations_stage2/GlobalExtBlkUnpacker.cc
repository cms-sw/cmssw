#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "GTCollections.h"
#include "GlobalExtBlkUnpacker.h"

namespace l1t {
  namespace stage2 {
    bool GlobalExtBlkUnpacker::unpack(const Block& block, UnpackerCollections* coll) {
      LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

      unsigned int wdPerBX = 6;  //should this be configured someplace else?
      int nBX = int(ceil(
          block.header().getSize() /
          6.));  // FOR GT Not sure what we have here...put at 6 because of 6 frames. Since there are 12 EGamma objects reported per event (see CMS IN-2013/005)

      // Find the central, first and last BXs
      int firstBX = -(ceil((double)nBX / 2.) - 1);
      int lastBX;
      if (nBX % 2 == 0) {
        lastBX = ceil((double)nBX / 2.);
      } else {
        lastBX = ceil((double)nBX / 2.) - 1;
      }

      auto res_ = static_cast<GTCollections*>(coll)->getExts();
      res_->setBXRange(firstBX, lastBX);

      LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

      // Loop over multiple BX and then number of EG cands filling collection
      int numBX = 0;  //positive int to count BX
      for (int bx = firstBX; bx <= lastBX; bx++) {
        // If this is the first block, instantiate GlobalExt so it is there to fill from mult. blocks
        if (block.header().getID() == 24) {
          LogDebug("L1T") << "Creating GT External Block for BX =" << bx;
          GlobalExtBlk tExt = GlobalExtBlk();
          res_->push_back(bx, tExt);
        }

        //fetch from collection
        GlobalExtBlk ext = res_->at(bx, 0);

        //Determine offset of algorithm bits based on block.ID
        // ID=24  offset = 0;  ID=26  offset=64;  ID=28  offset=128=2*64; ID=30 offset=3*64=192
        int extOffset = ((block.header().getID() - 24) / 2) * 64;

        for (unsigned int wd = 0; wd < wdPerBX; wd++) {
          uint32_t raw_data = block.payload()[wd + numBX * wdPerBX];
          LogDebug("L1T") << " payload word " << wd << " 0x" << hex << raw_data;

          if (wd < 2) {
            for (unsigned int bt = 0; bt < 32; bt++) {
              int val = ((raw_data >> bt) & 0x1);
              int extBit = bt + wd * 32 + extOffset;
              if (val == 1)
                ext.setExternalDecision(extBit, true);
            }
          }
        }

        // Put the object back into place (Must be better way???)
        res_->set(bx, 0, ext);

        //ext.print(std::cout);
        numBX++;
      }

      return true;
    }
  }  // namespace stage2
}  // namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::GlobalExtBlkUnpacker);
