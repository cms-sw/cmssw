#include "EventFilter/CSCRawToDigi/interface/CSCComparatorData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <cstdio>
#include <cstring>

#ifdef LOCAL_UNPACK
bool CSCComparatorData::debug = false;
#else
#include <atomic>
std::atomic<bool> CSCComparatorData::debug{false};
#endif

CSCComparatorData::CSCComparatorData(const CSCTMBHeader* tmbHeader)
    : ncfebs_(tmbHeader->NCFEBs()), ntbins_(tmbHeader->NTBins()) {
  if (tmbHeader != nullptr)
    theFirmwareVersion = tmbHeader->FirmwareVersion();
  else
    theFirmwareVersion = 2007;
  size_ = nlines();
  zero();
}

CSCComparatorData::CSCComparatorData(int ncfebs, int ntbins, int firmware_version)
    : ncfebs_(ncfebs), ntbins_(ntbins), theFirmwareVersion(firmware_version) {
  size_ = nlines();
  zero();
}

CSCComparatorData::CSCComparatorData(int ncfebs, int ntbins, const unsigned short* buf, int firmware_version)
    : ncfebs_(ncfebs), ntbins_(ntbins), theFirmwareVersion(firmware_version) {
  // add two more for odd ntbins, plus one for the e0c
  // Oct 2004 Rick: e0c line belongs to CSCTMBTrailer
  size_ = (nlines() % 2 == 1) ? nlines() + 2 : nlines();

  memcpy(theData, buf, size_ * 2);
}

void CSCComparatorData::zero() {
  for (int ifeb = 0; ifeb < ncfebs_; ++ifeb) {
    for (int tbin = 0; tbin < ntbins_; ++tbin) {
      for (int layer = CSCDetId::minLayerId(); layer <= CSCDetId::maxLayerId(); ++layer) {
        dataWord(ifeb, tbin, layer) = CSCComparatorDataWord(ifeb, tbin, 0);
      }
    }
  }
}

std::vector<CSCComparatorDigi> CSCComparatorData::comparatorDigis(uint32_t idlayer, unsigned cfeb) {
  static const bool doStripSwapping = true;
  bool me1a = (CSCDetId::station(idlayer) == 1) && (CSCDetId::ring(idlayer) == 4);
  bool zplus = (CSCDetId::endcap(idlayer) == 1);
  bool me1b = (CSCDetId::station(idlayer) == 1) && (CSCDetId::ring(idlayer) == 1);
  int layer = CSCDetId::layer(idlayer);

  assert(layer >= CSCDetId::minLayerId());
  assert(layer <= CSCDetId::maxLayerId());

  //looking for comp output on layer
  std::vector<CSCComparatorDigi> result;

  // this is pretty sparse data, so I wish we could check the
  // data word by word, not bit by bit, but I don't see how to
  // do the time sequencing that way.
  for (int distrip = 0; distrip < CSCConstants::NUM_DISTRIPS_PER_CFEB; ++distrip) {
    uint16_t tbinbitsS0HS0 = 0;
    uint16_t tbinbitsS0HS1 = 0;
    uint16_t tbinbitsS1HS0 = 0;
    uint16_t tbinbitsS1HS1 = 0;
    for (int tbin = 0; tbin < ntbins_ - 2; ++tbin) {
      if (bitValue(cfeb, tbin, layer, distrip)) {
        /// first do some checks
        CSCComparatorDataWord word = dataWord(cfeb, tbin, layer);
        assert(word.tbin_ == tbin);
        assert(word.cfeb_ == cfeb);
        // we have a hit.  The next two time samples
        // are the other two bits in the triad
        int bit2 = bitValue(cfeb, tbin + 1, layer, distrip);
        int bit3 = bitValue(cfeb, tbin + 2, layer, distrip);
        // should count from zero
        int chamberDistrip = distrip + cfeb * CSCConstants::NUM_DISTRIPS_PER_CFEB;
        int HalfStrip = 4 * chamberDistrip + bit2 * 2 + bit3;
        int output = 4 + bit2 * 2 + bit3;
        /*
         * Handles distrip logic; comparator output is for pairs of strips:
         * hit  bin  dec
         * x--- 100   4
         * -x-- 101   5
         * --x- 110   6
         * ---x 111   7
         *
         */

        if (debug)
          LogTrace("CSCComparatorData|CSCRawToDigi")
              << "fillComparatorOutputs: layer = " << layer << " timebin = " << tbin << " cfeb = " << cfeb
              << " distrip = " << chamberDistrip << " HalfStrip = " << HalfStrip << " Output " << output << std::endl;
        ///what is actually stored in comparator digis are 0/1 for left/right halfstrip for each strip

        ///constructing four bitted words for tbits on
        if (output == 4)
          tbinbitsS0HS0 = tbinbitsS0HS0 + (1 << tbin);
        if (output == 5)
          tbinbitsS0HS1 = tbinbitsS0HS1 + (1 << tbin);
        if (output == 6)
          tbinbitsS1HS0 = tbinbitsS1HS0 + (1 << tbin);
        if (output == 7)
          tbinbitsS1HS1 = tbinbitsS1HS1 + (1 << tbin);

        tbin += 2;
      }
    }  //end of loop over time bins
    //we do not have to check over the last couple of time bins if there are no hits since
    //comparators take 3 time bins

    // Store digis each of possible four halfstrips for given distrip:
    if (tbinbitsS0HS0 || tbinbitsS0HS1 || tbinbitsS1HS0 || tbinbitsS1HS1) {
      unsigned int cfeb_corr = cfeb;
      unsigned int distrip_corr = distrip;

      if (doStripSwapping) {
        // Fix ordering of strips and CFEBs in ME1/1.
        // SV, 27/05/08: keep CFEB=4 for ME1/a until CLCT trigger logic
        // stops combining it with the info from the other 4 CFEBs (ME1/b).
        //
        if (theFirmwareVersion >= 2013) {
          if (me1a && zplus) {
            distrip_corr = 7 - distrip;  // 0-7 -> 7-0
            cfeb_corr = 10 - cfeb;
          }
          if (me1b && !zplus) {
            distrip_corr = 7 - distrip;
            cfeb_corr = 3 - cfeb;
          }
        } else {
          // if ( me1a )           { cfeb_corr = 0; } // reset 4 to 0
          if (me1a && zplus) {
            distrip_corr = 7 - distrip;  // 0-7 -> 7-0
          }
          if (me1b && !zplus) {
            distrip_corr = 7 - distrip;
            cfeb_corr = 3 - cfeb;
          }
        }
      }

      int strip = 16 * cfeb_corr + 2 * distrip_corr + 1;

      if (debug)
        LogTrace("CSCComparatorData|CSCRawToDigi") << "fillComparatorOutputs: cfeb_corr = " << cfeb_corr
                                                   << " distrip_corr = " << distrip_corr << " strip = " << strip;

      if (doStripSwapping && ((me1a && zplus) || (me1b && !zplus))) {
        // Half-strips need to be flipped too.
        if (tbinbitsS1HS1)
          result.push_back(CSCComparatorDigi(strip, 0, tbinbitsS1HS1));
        if (tbinbitsS1HS0)
          result.push_back(CSCComparatorDigi(strip, 1, tbinbitsS1HS0));
        if (tbinbitsS0HS1)
          result.push_back(CSCComparatorDigi(strip + 1, 0, tbinbitsS0HS1));
        if (tbinbitsS0HS0)
          result.push_back(CSCComparatorDigi(strip + 1, 1, tbinbitsS0HS0));
      } else {
        if (tbinbitsS0HS0)
          result.push_back(CSCComparatorDigi(strip, 0, tbinbitsS0HS0));
        if (tbinbitsS0HS1)
          result.push_back(CSCComparatorDigi(strip, 1, tbinbitsS0HS1));
        if (tbinbitsS1HS0)
          result.push_back(CSCComparatorDigi(strip + 1, 0, tbinbitsS1HS0));
        if (tbinbitsS1HS1)
          result.push_back(CSCComparatorDigi(strip + 1, 1, tbinbitsS1HS1));
      }
      //uh oh ugly ugly ugly!
    }
  }  //end of loop over distrips
  return result;
}

std::vector<CSCComparatorDigi> CSCComparatorData::comparatorDigis(int layer) {
  assert(layer >= CSCDetId::minLayerId());
  assert(layer <= CSCDetId::maxLayerId());

  std::vector<CSCComparatorDigi> result;
  for (int cfeb = 0; cfeb < ncfebs_; ++cfeb) {
    std::vector<CSCComparatorDigi> oneCfebDigi = comparatorDigis(layer, cfeb);
    result.insert(result.end(), oneCfebDigi.begin(), oneCfebDigi.end());
  }

  return result;
}

void CSCComparatorData::add(const CSCComparatorDigi& digi, int layer) {
  int strip = digi.getStrip();
  int halfstrip = digi.getHalfStrip();
  int cfeb = digi.getCFEB();
  int distrip = digi.getDiStrip();

  // Check the distrip and half-strip number
  assert(cfeb < CSCConstants::MAX_CFEBS_RUN2);
  assert(distrip < CSCConstants::NUM_DISTRIPS_PER_CFEB);
  // note that half-strips are not staggered in the packer
  assert(halfstrip < CSCConstants::MAX_NUM_HALF_STRIPS_RUN2);

  std::vector<int> timeBinsOn = digi.getTimeBinsOn();
  for (std::vector<int>::const_iterator tbinItr = timeBinsOn.begin(); tbinItr != timeBinsOn.end(); ++tbinItr) {
    int tbin = *tbinItr;
    if (tbin >= 0 && tbin < ntbins_ - 2) {
      // First triad bit indicates the presence of the hit
      dataWord(cfeb, tbin, layer).set(distrip, true);
      // Second bit indicates which of the two strips contains the hit
      if (strip % 2 == 0)
        dataWord(cfeb, tbin + 1, layer).set(distrip, true);
      // Third bit indicates whether the hit is located on the left or on the
      // right side of the strip.
      if (digi.getComparator())
        dataWord(cfeb, tbin + 2, layer).set(distrip, true);
    }
  }
}

/***
 * Comparator packing version with ME11 strips swapping
 ***/
void CSCComparatorData::add(const CSCComparatorDigi& digi, const CSCDetId& cid) {
  static const bool doStripSwapping = true;
  bool me1a = (cid.station() == 1) && (cid.ring() == 4);
  bool zplus = (cid.endcap() == 1);
  bool me1b = (cid.station() == 1) && (cid.ring() == 1);

  unsigned layer = cid.layer();

  int strip = digi.getStrip();
  int halfstrip = digi.getHalfStrip();
  int cfeb = digi.getCFEB();
  int distrip = digi.getDiStrip();
  int bit2 = (strip - 1) % 2;
  int bit3 = digi.getComparator();

  // Check the distrip and half-strip number
  if (theFirmwareVersion >= 2013) {
    assert(cfeb < CSCConstants::MAX_CFEBS_RUN2);
    assert(distrip < CSCConstants::NUM_DISTRIPS_PER_CFEB);
    // note that half-strips are not staggered in the packer
    assert(halfstrip < CSCConstants::MAX_NUM_HALF_STRIPS_RUN2);
  } else {
    assert(cfeb < CSCConstants::MAX_CFEBS_RUN1);
    assert(distrip < CSCConstants::NUM_DISTRIPS_PER_CFEB);
    // note that half-strips are not staggered in the packer
    assert(halfstrip < CSCConstants::MAX_NUM_HALF_STRIPS_RUN1);
  }

  // Lets try to do ME11 strip flipping
  if (doStripSwapping) {
    if (theFirmwareVersion >= 2013) {
      if ((me1a || (me1b && (cfeb > CSCConstants::NUM_CFEBS_ME1A_UNGANGED))) && zplus) {
        distrip = 7 - distrip;  // 0-7 -> 7-0
        cfeb = 10 - cfeb;
        bit2 = ((31 - (halfstrip % CSCConstants::NUM_HALF_STRIPS_PER_CFEB)) % 4) / 2;
        bit3 = ((31 - (halfstrip % CSCConstants::NUM_HALF_STRIPS_PER_CFEB)) % 4) % 2;
      }
      if (me1b && !zplus && (cfeb < CSCConstants::NUM_CFEBS_ME1B)) {
        distrip = 7 - distrip;
        cfeb = 3 - cfeb;
        bit2 = ((31 - (halfstrip % CSCConstants::NUM_HALF_STRIPS_PER_CFEB)) % 4) / 2;
        bit3 = ((31 - (halfstrip % CSCConstants::NUM_HALF_STRIPS_PER_CFEB)) % 4) % 2;
      }
    } else {
      if ((me1a || (me1b && (cfeb > CSCConstants::NUM_CFEBS_ME1A_UNGANGED))) && zplus) {
        distrip = 7 - distrip;  // 0-7 -> 7-0
        bit2 = ((31 - (halfstrip % CSCConstants::NUM_HALF_STRIPS_PER_CFEB)) % 4) / 2;
        bit3 = ((31 - (halfstrip % CSCConstants::NUM_HALF_STRIPS_PER_CFEB)) % 4) % 2;
      }
      if (me1b && !zplus && (cfeb < CSCConstants::NUM_CFEBS_ME1B)) {
        distrip = 7 - distrip;
        cfeb = 3 - cfeb;
        bit2 = ((31 - (halfstrip % CSCConstants::NUM_HALF_STRIPS_PER_CFEB)) % 4) / 2;
        bit3 = ((31 - (halfstrip % CSCConstants::NUM_HALF_STRIPS_PER_CFEB)) % 4) % 2;
      }
    }
  }

  std::vector<int> timeBinsOn = digi.getTimeBinsOn();
  for (std::vector<int>::const_iterator tbinItr = timeBinsOn.begin(); tbinItr != timeBinsOn.end(); ++tbinItr) {
    int tbin = *tbinItr;
    if (tbin >= 0 && tbin < ntbins_ - 2) {
      // First triad bit indicates the presence of the hit
      dataWord(cfeb, tbin, layer).set(distrip, true);
      // Second bit indicates which of the two strips contains the hit
      // if (strip%2 == 0)
      if (bit2)
        dataWord(cfeb, tbin + 1, layer).set(distrip, true);
      // Third bit indicates whether the hit is located on the left or on the
      // right side of the strip.
      // if (digi.getComparator())
      if (bit3)
        dataWord(cfeb, tbin + 2, layer).set(distrip, true);
    }
  }
}

bool CSCComparatorData::check() const {
  bool result = true;
  for (int cfeb = 0; cfeb < ncfebs_; ++cfeb) {
    for (int tbin = 0; tbin < ntbins_; ++tbin) {
      for (int layer = CSCDetId::minLayerId(); layer <= CSCDetId::maxLayerId(); ++layer) {
        /// first do some checks
        const CSCComparatorDataWord& word = dataWord(cfeb, tbin, layer);
        bool wordIsGood = (word.tbin_ == tbin) && (word.cfeb_ == cfeb);
        result = result && wordIsGood;
        if (!wordIsGood && debug) {
          LogTrace("CSCComparatorData|CSCRawToDigi")
              << "Bad CLCT data  in layer " << layer << " expect CFEB " << cfeb << " tbin " << tbin;
          LogTrace("CSCComparatorData|CSCRawToDigi") << " See " << word.cfeb_ << " " << word.tbin_;
        }
      }
    }
  }
  if (!result)
    LogTrace("CSCComparatorData|CSCRawToDigi") << "++ Bad CLCT Data ++ ";
  return result;
}

void CSCComparatorData::dump() const {
  for (int i = 0; i < size_; i++) {
    printf("%04x %04x %04x %04x\n", theData[i + 3], theData[i + 2], theData[i + 1], theData[i]);
    i += 3;
  }
}

void CSCComparatorData::selfTest() {
  CSCComparatorData comparatorData(5, 16);
  // aim for output 4 in 5th time bin, = 0000000000010000
  CSCComparatorDigi comparatorDigi1(1, 0, 0x10);
  // aim for output 5 in 6th time bin, = 0000 0000 0010 0000
  CSCComparatorDigi comparatorDigi2(39, 1, 0x20);
  // aim for output 7 in 7th time bin, = 000 0000 0100 0000
  CSCComparatorDigi comparatorDigi3(80, 1, 0x40);

  comparatorData.add(comparatorDigi1, 1);
  comparatorData.add(comparatorDigi2, 4);
  comparatorData.add(comparatorDigi3, 6);

  CSCDetId layer1(1, 4, 1, 2, 1);
  CSCDetId layer4(1, 4, 1, 2, 4);
  CSCDetId layer6(1, 4, 1, 2, 6);

  std::vector<CSCComparatorDigi> digis1 = comparatorData.comparatorDigis(1);
  std::vector<CSCComparatorDigi> digis2 = comparatorData.comparatorDigis(4);
  std::vector<CSCComparatorDigi> digis3 = comparatorData.comparatorDigis(6);

  assert(digis1.size() == 1);
  assert(digis2.size() == 1);
  assert(digis3.size() == 1);

  assert(digis1[0].getStrip() == 1);
  assert(digis1[0].getComparator() == 0);
  assert(digis1[0].getTimeBin() == 4);

  assert(digis2[0].getStrip() == 39);
  assert(digis2[0].getComparator() == 1);
  assert(digis2[0].getTimeBin() == 5);

  assert(digis3[0].getStrip() == 80);
  assert(digis3[0].getComparator() == 1);
  assert(digis3[0].getTimeBin() == 6);
}
