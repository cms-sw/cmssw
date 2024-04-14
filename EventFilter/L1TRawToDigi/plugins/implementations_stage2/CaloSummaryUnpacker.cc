#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "L1TObjectCollections.h"

#include "DataFormats/L1CaloTrigger/interface/CICADA.h"

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
    constructedScore += ((float)bitsArray[bitIndex]) * pow(2.0, 4 * ((numCICADAWords / 2) - (bitIndex + 1)));
  }
  return constructedScore;
}

bool l1t::stage2::CaloSummaryUnpacker::unpack(const Block& block, UnpackerCollections* coll) {
  LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

  //Just a few things to help us handle the number of BXs
  //Strictly, we should generally get five BXs, starting at -2, and going to 2
  //With the central BX at 0. The frames count up from -2
  int nBX = int(ceil(block.header().getSize() / nFramesPerEvent));
  int firstBX = (nBX / 2) - nBX + 1;
  int lastBX = nBX / 2;
  int processedBXs = 0; //This will just help us keep track of what words we are grabbing
  

  auto res_ = static_cast<L1TObjectCollections*>(coll)->getCICADAScore();
  res_ -> setBXRange(firstBX, lastBX);

  for (int bx = firstBX; bx <= lastBX; ++bx){
    //convert to float and then multiply by a factor based on the index?
    unsigned int cicadaBits[numCICADAWords] = {0, 0, 0, 0};

    for (unsigned int wordNum = 0; wordNum < numCICADAWords; ++wordNum) {
      unsigned short wordLocation = processedBXs*nFramesPerEvent + wordNum; //Calculate the location of the needed CICADA word based on how many BXs we have already handled, and how many words of CICADA we have already grabbed.
      //Frame 0 of a bx are the most significant integer bits
      //Frame 1 of a bx are the least significant integer bits
      //Frame 2 of a bx are the most significant decimal bits
      //Frame 3 of a bx are the lest significant decimal bits
      //Frames 4&5 are unused (by CICADA), they are reserved.
      uint32_t raw_data = block.payload().at(wordLocation);
      cicadaBits[wordNum] =
	(cicadaBitsPattern & raw_data) >>
	28;  //The 28 shifts the extracted bits over to the start of the 32 bit result data for easier working with later
    }
    res_->push_back(bx, processBitsToScore(cicadaBits)); //Now we insert CICADA into the proper BX, after a quick utility constructs a number from the 4 sets of bits.
    ++processedBXs; //index BXs
  }

  return true;
}

DEFINE_L1T_UNPACKER(l1t::stage2::CaloSummaryUnpacker);
