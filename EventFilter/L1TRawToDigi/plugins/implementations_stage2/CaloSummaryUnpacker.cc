#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "L1TObjectCollections.h"

#include "CaloSummaryUnpacker.h"
#include "GTSetup.h"

#include <cmath>

float l1t::stage2::CaloSummaryUnpacker::processBitsToScore(const unsigned int bitsArray[]) {
  float constructedScore = 0.0;
  //All bits have been shifted to just left of the decimal point
  //We need to convert them to float, shift them back to their proper position
  //And then add them into the total
  //The proper power is 4(-(bitIndex+1) + numCICADAWords/2)
  // i.e. shift bitIndex to max out at half the number of CICADA words (indexed at 0) then count down
  //And we shift by 4 bits a time, hence the factor of 4
  for (unsigned short bitIndex = 0; bitIndex < numCICADAWords; ++bitIndex) {
    constructedScore += ((float)bitsArray[bitIndex]) * pow(2.0, 4 * (numCICADAWords / 2 - (bitIndex + 1)));
  }
  return constructedScore;
}

bool l1t::stage2::CaloSummaryUnpacker::unpack(const Block& block, UnpackerCollections* coll) {
  LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();
  //convert to float and then multiply by a factor based on the index?
  unsigned int cicadaBits[numCICADAWords] = {0, 0, 0, 0};

  auto res_ = static_cast<L1TObjectCollections*>(coll)->getCICADAScore();

  for (unsigned int wordNum = 0; wordNum < nFrames; ++wordNum) {
    uint32_t raw_data = block.payload().at(wordNum);
    if (wordNum < numCICADAWords) {  // check the first 4 frames for CICADA bits
      cicadaBits[wordNum] =
          (cicadaBitsPattern & raw_data) >>
          28;  //The 28 shifts the extracted bits over to the start of the 32 bit result data for easier working with
    }
  }
  *res_ = processBitsToScore(cicadaBits);

  return true;
}

DEFINE_L1T_UNPACKER(l1t::stage2::CaloSummaryUnpacker);
