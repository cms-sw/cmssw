#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "GTCollections.h"
#include "GlobalAlgBlkUnpacker.h"

namespace l1t {
  namespace stage2 {
    bool GlobalAlgBlkUnpacker::unpack(const Block& block, UnpackerCollections* coll) {
      LogDebug("L1T") << "AMCNo " << block.amc().getAMCNumber() << " Block ID  = " << block.header().getID()
                      << " size = " << block.header().getSize();

      // ================================================================================
      //Should this be configured someplace?
      unsigned int wdPerBX = 6;
      unsigned int initialBlkID = 33;  //first block of inital alg bits
      unsigned int intermBlkID = 39;   //first block of alg bits after intermediate step
      unsigned int finalBlkID = 45;    //first block of final alg bits
      // ================================================================================

      unsigned int uGTBoard = block.amc().getAMCNumber() - 1;

      int nBX =
          int(ceil(block.header().getSize() / 6.));  // FOR GT Not sure what we have here...put at 6 because of 6 frames

      // Find the central, first and last BXs
      int firstBX = -(ceil((double)nBX / 2.) - 1);
      int lastBX;
      if (nBX % 2 == 0) {
        lastBX = ceil((double)nBX / 2.);
      } else {
        lastBX = ceil((double)nBX / 2.) - 1;
      }

      auto res_ = static_cast<GTCollections*>(coll)->getAlgs();
      res_->setBXRange(firstBX, lastBX);

      LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX << endl;

      // Loop over multiple BX and then number of EG cands filling collection
      int numBX = 0;  //positive int to count BX
      for (int bx = firstBX; bx <= lastBX; bx++) {
        // If this is the first block on first board, instantiate GlobalAlg so it is there to fill from mult. blocks
        if (block.header().getID() == initialBlkID && uGTBoard == 0) {
          LogDebug("L1T") << "Creating GT Algorithm Block for BX =" << bx << std::endl;
          GlobalAlgBlk talg = GlobalAlgBlk();
          res_->push_back(bx, talg);
        }
        //If this is not the first block, but the vector is empty, something has gone wrong (corrupted data)
        else if (res_->isEmpty(bx))
          throw cms::Exception("InvalidGlobalAlgBlkBxCollection")
              << "The GlobalAlgBlk unpacker result vector is empty, but is not receiving the first expected header "
                 "ID! This may be due to corrupted, or poorly formatted events.\n"
              << "uGTBoard: " << uGTBoard << "\nBX: " << bx << "\nFirst expected block: " << initialBlkID
              << "\nReceived block: " << block.header().getID();

        //fetch
        GlobalAlgBlk alg = res_->at(bx, 0);

        //Determine offset of algorithm bits based on block.ID
        // ID=initialBlkID    offset = 0;  ID=initialBlkID+2    offset=192;  ID=initialBlkID+4    offset=384=2*192; (before prescale)
        // ID=intermBlkID  offset = 0;  ID=intermBlkID+2  offset=192;  ID=intermBlkID+4  offset=384=2*192; (after prescale)
        // ID=finalBlkID      offset = 0;  ID=finalBlkID+2      offset=192;  ID=finalBlkID+4      offset=384=2*192; (after mask (Final))
        int algOffset = (block.header().getID() - initialBlkID + 1) / 2;
        algOffset = (algOffset % 3) * 192;

        for (unsigned int wd = 0; wd < wdPerBX; wd++) {
          uint32_t raw_data = block.payload()[wd + numBX * wdPerBX];
          LogDebug("L1T") << "BX " << bx << " payload word " << wd << " 0x" << hex << raw_data << " offset=" << dec
                          << algOffset << std::endl;

          //parse these 32 bits into algorithm bits (perhaps needs a more efficient way of doing this?
          if ((block.header().getID() != initialBlkID + 4 && block.header().getID() != intermBlkID + 4 &&
               block.header().getID() != finalBlkID + 4) ||
              wd < 4) {
            for (unsigned int bt = 0; bt < 32; bt++) {
              int val = ((raw_data >> bt) & 0x1);
              unsigned int algBit = bt + wd * 32 + algOffset;

              if (val == 1 && algBit < alg.maxPhysicsTriggers) {
                LogDebug("L1T") << "Found valid alg bit (" << algBit << ") on bit (" << bt << ") word (" << wd
                                << ") algOffset (" << algOffset << ") block ID (" << block.header().getID() << ")"
                                << " Board# " << uGTBoard << std::endl;
                if (block.header().getID() < initialBlkID + 5) {
                  alg.setAlgoDecisionInitial(algBit, true);
                } else if (block.header().getID() < intermBlkID + 5) {
                  alg.setAlgoDecisionInterm(algBit, true);
                } else {
                  alg.setAlgoDecisionFinal(algBit, true);
                }
              } else if (val == 1) {
                LogDebug("L1T") << "Found invalid alg bit (" << algBit << ") out of range on bit (" << bt << ") word ("
                                << wd << ") algOffset (" << algOffset << ") block ID (" << block.header().getID() << ")"
                                << std::endl;
              }
            }

          } else if (block.header().getID() == initialBlkID + 4 && (wd == 4 || wd == 5)) {
            //This is the 32bit hash of menu name
            if (wd == 4)
              alg.setL1MenuUUID(raw_data);
            //This is the 32bit hash of menu firmware uuid
            if (wd == 5)
              alg.setL1FirmwareUUID(raw_data);

          } else if (block.header().getID() == finalBlkID + 4 && wd == 4) {
            //Get the local FINORs and Veto...Global FINOR calculated below
            if ((raw_data & 0x100) >> 8)
              alg.setFinalORVeto(true);
            if ((raw_data & 0x1) >> 0)
              alg.setFinalORPreVeto(true);
            LogDebug("L1T") << " Packing the FinalOR " << wd << " 0x" << hex << raw_data << endl;
          } else if (block.header().getID() == finalBlkID + 4 && wd == 5) {
            //This is the Prescale Column
            alg.setPreScColumn(raw_data & 0xFF);
            LogDebug("L1T") << " Packing the Prescale Column " << wd << " 0x" << hex << raw_data << endl;
          }
        }

        //Redetermine Final (multiboard)  FINOR
        //be explicit and must set to false if we find a board with veto set.
        alg.setFinalOR(alg.getFinalORPreVeto() && !alg.getFinalORVeto());

        // Put the object back into place (Must be better way)
        res_->set(bx, 0, alg);

        //alg.print(std::cout);

        //increment counter of which BX we are processing
        numBX++;
      }

      return true;
    }
  }  // namespace stage2
}  // namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::GlobalAlgBlkUnpacker);
