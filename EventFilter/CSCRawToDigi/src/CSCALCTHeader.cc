#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "EventFilter/CSCRawToDigi/interface/bitset_append.h"
#include "EventFilter/CSCRawToDigi/interface/cscPackerCompare.h"
#include <iomanip>

#ifdef LOCAL_UNPACK

bool CSCALCTHeader::debug = false;
short unsigned int CSCALCTHeader::firmwareVersion = 2007;

#else

#include <atomic>

std::atomic<bool> CSCALCTHeader::debug{false};
std::atomic<short unsigned int> CSCALCTHeader::firmwareVersion{2007};

#endif

CSCALCTHeader::CSCALCTHeader(int chamberType)
    : header2006(chamberType), header2007(chamberType) {  //constructor for digi->raw packing based on header2006
  if (firmwareVersion == 2006) {
    memcpy(theOriginalBuffer, &header2006, header2006.sizeInWords() * 2);
  } else if (firmwareVersion == 2007) {
    memcpy(theOriginalBuffer, &header2007, header2007.sizeInWords() * 2);
    // assume no virtex or masks or registers
    sizeInWords2007_ = header2007.sizeInWords() + header2007.lctBins * CSCALCT::sizeInWords() * 2;
    theALCTs.resize(header2007.lctBins * 2);
  } else {
    edm::LogError("CSCALCTHeader|CSCRawToDigi")
        << "Cannot construct ALCT header: ALCT firmware version is bad/not defined!" << firmwareVersion;
  }
}

CSCALCTHeader::CSCALCTHeader(const unsigned short *buf) {
  ///collision and hot channel masks are variable sized
  ///the sizes vary depending on type of the ALCT board
  ///                                        number of words for various
  ///                                        alct board types:  1  2  3     5  6
  constexpr unsigned short int collisionMaskWordcount[7] = {8, 8, 12, 16, 16, 24, 28};
  constexpr unsigned short int hotChannelMaskWordcount[7] = {18, 18, 24, 36, 36, 48, 60};

  ///first determine the correct format
  if (buf[0] == 0xDB0A) {
    firmwareVersion = 2007;
  } else if ((buf[0] & 0xF800) == 0x6000) {
    firmwareVersion = 2006;
  } else {
    edm::LogError("CSCALCTHeader|CSCRawToDigi") << "failed to determine ALCT firmware version!!";
  }

  LogTrace("CSCALCTHeader|CSCRawToDigi") << "firmware version - " << firmwareVersion;

  ///Now fill data
#ifdef LOCAL_UNPACK
  switch (firmwareVersion) {
#else
  switch (firmwareVersion.load()) {
#endif
    case 2006:
      header2006.setFromBuffer(buf);  ///the header part
      buf += header2006.sizeInWords();
      alcts2006.setFromBuffer(buf);  ///the alct0 and alct1
      buf += alcts2006.sizeInWords();
      break;

    case 2007:
      header2007.setFromBuffer(buf);  ///the fixed sized header part
      buf += header2007.sizeInWords();
      sizeInWords2007_ = header2007.sizeInWords();
      ///now come the variable parts
      if (header2007.configPresent == 1) {
        virtexID.setFromBuffer(buf);
        buf += virtexID.sizeInWords();
        sizeInWords2007_ = virtexID.sizeInWords();
        configRegister.setFromBuffer(buf);
        buf += configRegister.sizeInWords();
        sizeInWords2007_ += configRegister.sizeInWords();

        collisionMasks.resize(collisionMaskWordcount[header2007.boardType]);
        for (unsigned int i = 0; i < collisionMaskWordcount[header2007.boardType]; ++i) {
          collisionMasks[i].setFromBuffer(buf);
          buf += collisionMasks[i].sizeInWords();
          sizeInWords2007_ += collisionMasks[i].sizeInWords();
        }

        hotChannelMasks.resize(hotChannelMaskWordcount[header2007.boardType]);
        for (unsigned int i = 0; i < hotChannelMaskWordcount[header2007.boardType]; ++i) {
          hotChannelMasks[i].setFromBuffer(buf);
          buf += hotChannelMasks[i].sizeInWords();
          sizeInWords2007_ += hotChannelMasks[i].sizeInWords();
        }
      }

      theALCTs.resize(header2007.lctBins * 2);  ///2007 has LCTbins * 2 alct words
      for (int i = 0; i < header2007.lctBins * 2; ++i) {
        theALCTs[i].setFromBuffer(buf);
        buf += theALCTs[i].sizeInWords();
        sizeInWords2007_ += theALCTs[i].sizeInWords();
      }

      ALCTDigis();
      break;

    default:
      edm::LogError("CSCALCTHeader|CSCRawToDigi") << "couldn't construct: ALCT firmware version is bad/not defined!";
      break;
  }

  ///also store raw data buffer too; it is later returned by data() method
  if ((firmwareVersion == 2006) || (firmwareVersion == 2007))
    memcpy(theOriginalBuffer, buf - sizeInWords(), sizeInWords() * 2);
}

CSCALCTHeader::CSCALCTHeader(const CSCALCTStatusDigi &digi) { CSCALCTHeader(digi.header()); }

void CSCALCTHeader::setEventInformation(const CSCDMBHeader &dmb) {
#ifdef LOCAL_UNPACK
  switch (firmwareVersion) {
#else
  switch (firmwareVersion.load()) {
#endif
    case 2006: {
      header2006.setEventInformation(dmb);
      break;
    }
    case 2007: {
      header2007.setEventInformation(dmb);
      break;
    }
    default:
      edm::LogError("CSCALCTHeader|CSCRawToDigi")
          << "setEventInformation: ALCT firmware version is bad/not defined!" << firmwareVersion;
      break;
  }
}

unsigned short CSCALCTHeader::nLCTChipRead() const {  ///header2006 method
  if (firmwareVersion == 2006) {
    return header2006.nLCTChipRead();
  } else {
    // nLCTChip obsolete in ALCT2007 format (email Andrey K. & Victor B., 20.10.2008)
    // and we don't think anyone makes uses of this call.
    //    edm::LogError("CSCALCTHeader|CSCRawToDigi")
    //      <<"How is nLCTChipRead() supposed to work for ALCTHeader2007?";
  }
  return 0;
}

std::vector<CSCALCTDigi> CSCALCTHeader::ALCTDigis() const {
  std::vector<CSCALCTDigi> result;

#ifdef LOCAL_UNPACK
  switch (firmwareVersion) {
#else
  switch (firmwareVersion.load()) {
#endif
    case 2006: {
      result = alcts2006.ALCTDigis();
      break;
    }
    case 2007: {
      result.reserve(theALCTs.size());
      for (unsigned int i = 0; i < theALCTs.size(); ++i) {  ///loop over all alct words
        CSCALCTDigi digi(theALCTs[i].valid,
                         theALCTs[i].quality,
                         theALCTs[i].accel,
                         theALCTs[i].pattern,
                         theALCTs[i].keyWire,
                         (int)i / 2,
                         i % 2 + 1);
        result.push_back(digi);
      }
      break;
    }
    default:
      edm::LogError("CSCALCTHeader|CSCRawToDigi")
          << "Empty Digis: ALCT firmware version is bad/not defined!" << firmwareVersion;
      break;
  }
  for (unsigned i = 0; i < result.size(); ++i) {
    result[i].setFullBX(BXNCount());
  }
  return result;
}

void CSCALCTHeader::add(const std::vector<CSCALCTDigi> &digis) {
  if (firmwareVersion == 2006) {
    alcts2006.add(digis);
  } else if (firmwareVersion == 2007) {
    if (theALCTs.empty()) {
      theALCTs.resize(header2007.lctBins * 2);
    }
    for (std::vector<CSCALCTDigi>::const_iterator digi = digis.begin(); digi != digis.end(); ++digi) {
      int bx = digi->getBX();
      if (bx < (int)header2007.lctBins) {
        // 2 ALCTs per bx
        int i = bx * 2;
        int q1 = theALCTs[i].quality;
        int q2 = theALCTs[i + 1].quality;
        // see if it's non=blank
        if (!theALCTs[i].valid) {
          theALCTs[i] = CSCALCT(*digi);
        }
        // new best LCT
        else if (digi->getQuality() > q1) {
          theALCTs[i + 1] = theALCTs[i];
          theALCTs[i] = CSCALCT(*digi);
        }
        // new second best
        else if (!theALCTs[i + 1].valid || (digi->getQuality() > q2)) {
          theALCTs[i + 1] = CSCALCT(*digi);
        }
      }
    }
  }
}

/// Add ALCT anodes HMT shower bits per ALCT BX
void CSCALCTHeader::addShower(const std::vector<CSCShowerDigi> &digis) {
  if (firmwareVersion == 2007) {
    if (theALCTs.empty()) {
      theALCTs.resize(header2007.lctBins * 2);
    }
    for (std::vector<CSCShowerDigi>::const_iterator digi = digis.begin(); digi != digis.end(); ++digi) {
      int bx = digi - digis.begin();
      if (bx < (int)header2007.lctBins) {
        const CSCShowerDigi &digi = digis[bx];
        int i = bx * 2;
        unsigned hmt_bits = digi.isValid() ? digi.bitsInTime() : 0;
        theALCTs[i].reserved = hmt_bits & 0x3;
        theALCTs[i + 1].reserved = hmt_bits & 0x3;
      }
    }
  }
}

boost::dynamic_bitset<> CSCALCTHeader::pack() {
  boost::dynamic_bitset<> result;
  if (firmwareVersion == 2006) {
    boost::dynamic_bitset<> header =
        bitset_utilities::ushortToBitset(header2006.sizeInWords() * 16, (unsigned short *)&header2006);
    boost::dynamic_bitset<> alcts =
        bitset_utilities::ushortToBitset(alcts2006.sizeInWords() * 16, (unsigned short *)&alcts2006);
    result = bitset_utilities::append(header, alcts);

    bitset_utilities::bitsetToChar(result, (unsigned char *)data());
  }

  else if (firmwareVersion == 2007) {
    result = bitset_utilities::ushortToBitset(header2007.sizeInWords() * 16, (unsigned short *)&header2007);

    for (unsigned i = 0; i < theALCTs.size(); ++i) {
      boost::dynamic_bitset<> alct =
          bitset_utilities::ushortToBitset(theALCTs[i].sizeInWords() * 16, (unsigned short *)&theALCTs[i]);
      result = bitset_utilities::append(result, alct);
    }

    bitset_utilities::bitsetToChar(result, (unsigned char *)data());
  }
  return result;
}

void CSCALCTHeader::selfTest(int firmware) {
  firmwareVersion = firmware;
  CSCALCTDigi alct0(true, 1, 1, 1, 10, 6, 1);
  CSCALCTDigi alct1(true, 1, 1, 0, 11, 6, 2);

  // tests packing and unpacking
  for (int station = 1; station <= 4; ++station) {
    CSCDetId detId(1, station, 1, 1, 0);

    std::vector<CSCALCTDigi> oldAlcts;
    oldAlcts.push_back(alct0);
    oldAlcts.push_back(alct1);
    CSCALCTHeader alctHeader(detId.iChamberType());

    alctHeader.add(oldAlcts);

    std::vector<CSCALCTDigi> alcts = alctHeader.ALCTDigis();
    // pick out the valid ones
    std::vector<CSCALCTDigi> validALCTs;
    for (std::vector<CSCALCTDigi>::const_iterator alctItr = alcts.begin(); alctItr != alcts.end(); ++alctItr) {
      if (alctItr->isValid()) {
        validALCTs.push_back(*alctItr);
      }
    }
    assert(validALCTs[0] == alct0);
    assert(validALCTs[1] == alct1);
    //cscClassPackerCompare(alctHeader);
  }
}

std::ostream &operator<<(std::ostream &os, const CSCALCTHeader &header) {
  os << "ALCT HEADER CSCID " << header.CSCID() << "  L1ACC " << header.L1Acc() << std::endl;
  os << " time samples " << header.NTBins() << std::endl;
  return os;
}
