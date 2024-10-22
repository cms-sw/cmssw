#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "L1TObjectCollections.h"

#include "DataFormats/L1CaloTrigger/interface/CICADA.h"

#include "CaloSummaryUnpacker.h"
#include "GTSetup.h"

#include <cmath>

bool l1t::stage2::CaloSummaryUnpacker::unpack(const Block& block, UnpackerCollections* coll) {
  LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

  //Just a few things to help us handle the number of BXs
  //Strictly, we should generally get five BXs, starting at -2, and going to 2
  //With the central BX at 0. The frames count up from -2
  int nBX = int(ceil(block.header().getSize() / nFramesPerEvent));
  int firstBX = (nBX / 2) - nBX + 1;
  int lastBX = nBX / 2;
  int processedBXs = 0;  //This will just help us keep track of what words we are grabbing

  auto res_ = static_cast<L1TObjectCollections*>(coll)->getCICADAScore();
  res_->setBXRange(firstBX, lastBX);

  for (int bx = firstBX; bx <= lastBX; ++bx) {
    unsigned short baseLocation = processedBXs * nFramesPerEvent;
    const uint32_t* base = block.payload().data() + baseLocation;
    //The take the first 4 bits of the first 4 words, and arrange them in order
    uint32_t word = (cicadaBitsPattern & base[0]) >> 16 | (cicadaBitsPattern & base[1]) >> 20 |
                    (cicadaBitsPattern & base[2]) >> 24 | (cicadaBitsPattern & base[3]) >> 28;
    //The score needs to be shifted 8 bits over the decimal point
    float score = static_cast<float>(word) / 256.f;
    res_->push_back(bx, score);
    ++processedBXs;  //index BXs
  }

  return true;
}

DEFINE_L1T_UNPACKER(l1t::stage2::CaloSummaryUnpacker);
