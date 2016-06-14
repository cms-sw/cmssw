#ifndef UCTCTP7RawData_hh
#define UCTCTP7RawData_hh

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
using namespace edm;

class UCTCTP7RawData {
public:

  enum CaloType {EBEE=0, HBHE, HF};

  UCTCTP7RawData(const uint32_t *d) : myDataPtr(d) {
    if(myDataPtr != 0) {
      if(sof() != 0xA110CA7E) {
	LogError("UCTCTP7RawData") << "Failed to see 0xA110CA7E at start - but continuing" << std::endl;
      }
    }
  }
  
  virtual ~UCTCTP7RawData() {;}
  
  // Access functions for convenience
  
  const uint32_t *dataPtr() const {return myDataPtr;}

  uint32_t sof() {return myDataPtr[0];}

  uint32_t caloLinkBXID() {return (myDataPtr[1] & 0x00000FFF);}

  uint32_t nBXPerL1A() {return ((myDataPtr[1] & 0x000F0000) >> 16);}

  uint32_t getIndex(CaloType cType, bool negativeEta, uint32_t cEta, uint32_t iPhi) {
    uint32_t index = 0xDEADBEEF;
    if(cType == EBEE || cType == HBHE) {
      if(iPhi > 3) {
	LogError("UCTCTP7RawData") << "Incorrect iPhi; iPhi = " << iPhi << "; should be in [0,3]" << std::endl;
	return 0xDEADBEEF;
      }
      if(cEta < 1 || cEta > 28) {
	LogError("UCTCTP7RawData") << "Incorrect caloEta; cEta = " << cEta << "; should be in [1-28]" << std::endl;
	return 0xDEADBEEF;
      }
      // ECAL/HB+HE fragment size is 3 32-bit words
      // Each fragment covers 2 eta and 4 phi towers
      // All four phi towers are in one 32-bit word
      // Even and odd eta are in neighboring 32-bit words
      index = 2 + ( ((cEta - 1) / 2) * (3 + 3) + ((cEta - 1) % 2) );
      // But, towers are arranged in a peculiar order for firmware
      // convenience - the index needs to be computing with these
      // if statements.  This is brittle code that one should be
      // very careful with.
      if(negativeEta) {
	// Add offset for 6 ECAL and 6 HCAL fragments
	index += (6 * (3 + 3));
      }
      else {
	if(cEta > 12) {
	  // Add offset for 14 ECAL, 14 HB+HE and 2 HF fragments
	  // Note that first six are included in the definition of
	  // the variable - index  
	  // Note also that HF fragments are larger at 4 32-bit words
	  index += ((14 * (3 + 3) + (2 * 4)));
	}
      }
      // Data starts with ECAL towers so offset by 3 additional 32-bit words
      if(cType == HBHE) index += 3;
    }
    else if(cType == HF) {
      if(iPhi > 1) {
	LogError("UCTCTP7RawData") << "HF iPhi should be 0 or 1 (for a , b) - invalid iPhi  = " << iPhi << std::endl;
	return 0xDEADBEEF;
      }
      if(cEta < 30 || cEta > 41) {
	LogError("UCTCTP7RawData") << "HF cEta should be between 30 and 41 - invalid cEta = " << cEta << std::endl;
	return 0xDEADBEEF;
      }
      if(negativeEta) {
	if(iPhi == 0) {
	  // Offset by 6 positive eta and 14 negative eta EBEE/HBHE fragments (each 3 32-bit words)
	  // There are four HF cEta towers packed in each 32-bit word
	  // Add additional offset of 1 for (34-37) and 2 for (38-41)
	  index = 2 + 20 * (3 + 3) + ((cEta - 30) / 4);
	}
	else {
	  // Additional HF a fragment offset for HF b channel
	  index = 2 + 20 * (3 + 3) + 1 * 4 + ((cEta - 30) / 4);
	}
      }
      else {
	if(iPhi == 0) {
	  // Offset by all EBEE/HBHE and two HF fragments (4 32-bit words)
	  index = 2 + 2 * 14 * (3 + 3) + 2 * 4 + ((cEta - 30) / 4);
	}
	else {
	  // Additional HF a fragment offset for HF b channel
	  index = 2 + 2 * 14 * (3 + 3) + 3 * 4 + ((cEta - 30) / 4);
	}
      }
    }
    else {
      LogError("UCTCTP7RawData") << "Unknown CaloType " << cType << std::endl;
      return 0xDEADBEEF;
    }
    return index;
  }

  uint32_t getFeatureIndex(CaloType cType, bool negativeEta, uint32_t cEta, uint32_t iPhi) {
    // Get index into the data words for the tower
    uint32_t index = getIndex(cType, negativeEta, cEta, iPhi);
    if(cType == EBEE || cType == HBHE) {
      // Two 32-bit words contain ET, so we should offset the index to
      // to the feature and link status bits
      if(((cEta - 1) % 2) == 0) {
	// [index] is offset to ET of first four towers (0 - 3)
	// [index + 2] is where the feature and link status bits are
	index += 2;
      }
      else {
	// In this case [index] is offset to ET of second four towers (4 - 7)
	// [index + 1] is where the feature and link status bits are
	index += 1;
      }
    }
    else if(cType == HF) {
      // HF Fragment has different structure than EBEE and HBHE fragments
      // First three 32-bit words have ETs for 11 objects (yes, 11 not 12)
      // cEta = 40 / 41 are double in eta and flop bettween a and b HF fragments
      // Further the remaining upper byte of the third word actually has feature
      // bits.  This feature index will point to the 4th 32-bit word.  It is
      // expected that the top byte from 3rd 32-bit word will be patched in within
      // the feature bit access function.
      // Since there are three instead of if block as above for EBEE, HBHE
      // I wrote here a more compact implementation of index computation.
      index += (3 - ((cEta - 30) / 4));
    }
    else {
      return 0xDEADBEEF;
    }
    return index;
  }

  uint32_t getET(CaloType cType, bool negativeEta, uint32_t cEta, uint32_t iPhi) {
    uint32_t index = getIndex(cType, negativeEta, cEta, iPhi);
    const uint32_t data = myDataPtr[index];
    uint32_t et = 0xDEADBEEF;
    if(cType == HF) {
      // Pick out the correct 8-bits for the iEta chosen
      // Note that cEta = 41 is special, it only occurs for iPhi == 1 and shares cEta = 40 position
      if(cEta == 41) et = ((data >> 16) & 0xFF);
      else et = ((data >> ((cEta - 30) % 4) * 8) & 0xFF);
    }
    else {
      // Pick out the correct 8-bits for the iPhi chosen
      et = ((data >> (iPhi * 8)) & 0xFF);
    }
    return et;
  }

  uint32_t getFB(CaloType cType, bool negativeEta, uint32_t cEta, uint32_t iPhi) {
    uint32_t index = getFeatureIndex(cType, negativeEta, cEta, iPhi);
    const uint32_t data = myDataPtr[index];
    uint32_t fb = 0;
    if(cType == HF) {
      fb = getHFFeatureBits(negativeEta, cEta, iPhi);
    }
    else {
      // Pick out the correct bit for the tower chosen
      uint32_t tower = iPhi;
      if(((cEta - 1) % 2) == 1) {
	tower += 4;
      }
      fb = ((data & (0x1 << tower)) != 0) ? 1 : 0;
    }
    return fb;
  }

  uint32_t getHFFeatureBits(bool negativeEta, uint32_t cEta, uint32_t iPhi) {
    uint32_t index = getFeatureIndex(HF, negativeEta, cEta, iPhi);
    // Stitch together the top 8 bits from previous 32-bit word and bottom 14 bits from this word
    const uint32_t data = ((myDataPtr[index] & 0x3FFF) << 8) + (myDataPtr[index - 1] >> 24);
    uint32_t shift = (cEta - 30) * 2;
    if(cEta == 41) shift = 20; // 41 occurs on b-fiber but shares the position of 40
    return ((data >> shift) & 0x3);
  }

  uint32_t getLinkStatus(CaloType cType, bool negativeEta, uint32_t cEta, uint32_t iPhi) {
    uint32_t index = getFeatureIndex(cType, negativeEta, cEta, iPhi);
    const uint32_t data = myDataPtr[index];
    return (data >> 16);
  }

  uint32_t getSummaryIndex(bool negativeEta, uint32_t region) {
    uint32_t index = 2 + 2 * 14 * (3 + 3) + 4 * 4 + (region / 2);
    if(negativeEta) index += 4;
    return index;
  }

  uint32_t getRegionSummary(bool negativeEta, uint32_t region) {
    uint32_t index = getSummaryIndex(negativeEta, region);
    const uint32_t data = myDataPtr[index];
    return ((data >> (16 * (region % 2))) & 0xFFFF);
  }

  uint32_t getRegionET(bool negativeEta, uint32_t region) {
    return (getRegionSummary(negativeEta, region) & 0x3FF);
  }
  
  bool getRegionEGVeto(bool negativeEta, uint32_t region) {
    return (getRegionSummary(negativeEta, region) & 0x0400);
  }
  
  bool getRegionTauVeto(bool negativeEta, uint32_t region) {
    return (getRegionSummary(negativeEta, region) & 0x0800);
  }
  
  uint32_t getRegionHitLocation(bool negativeEta, uint32_t region) {
    return ((getRegionSummary(negativeEta, region) & 0xF000) >> 12);
  }

  bool isTowerMasked(CaloType cType, bool negativeEta, uint32_t cEta, uint32_t iPhi) {
    uint32_t linkStatus = getLinkStatus(cType, negativeEta, cEta, iPhi);
    uint32_t tower = iPhi;
    if(((cEta - 1) % 2) == 1) tower += 4;
    if(cType == HF) {
      tower = (cEta - 30);
      if(cEta == 41) tower = 10;
    }
    return ((linkStatus & (0x1 << tower)) != 0);
  }

  bool isLinkMisaligned(CaloType cType, bool negativeEta, uint32_t cEta, uint32_t iPhi) {
    uint32_t linkStatus = getLinkStatus(cType, negativeEta, cEta, iPhi);
    return ((linkStatus & 0x00001000) != 0);
  }

  bool isLinkInError(CaloType cType, bool negativeEta, uint32_t cEta, uint32_t iPhi) {
    uint32_t linkStatus = getLinkStatus(cType, negativeEta, cEta, iPhi);
    return ((linkStatus & 0x00002000) != 0);
  }

  bool isLinkDown(CaloType cType, bool negativeEta, uint32_t cEta, uint32_t iPhi) {
    uint32_t linkStatus = getLinkStatus(cType, negativeEta, cEta, iPhi);
    return ((linkStatus & 0x00004000) != 0);
  }

  bool isLinkMasked(CaloType cType, bool negativeEta, uint32_t cEta, uint32_t iPhi) {
    uint32_t linkStatus = getLinkStatus(cType, negativeEta, cEta, iPhi);
    return ((linkStatus & 0x00008000) != 0);
  }

  void print() {
    using namespace std;
    LogError("UCTCTP7RawData") << "CTP7 Payload Header:" << endl;
    LogError("UCTCTP7RawData") << "No BX per L1A = " << dec << nBXPerL1A() << endl;
    LogError("UCTCTP7RawData") << "Calo BX ID    = " << dec << caloLinkBXID() << endl;
    CaloType cType = EBEE;
    bool negativeEta = false;
    bool first = true;
    for(uint32_t i = 0; i < 2; i++) {
      if(i != 0) negativeEta = true;
      first = true;
      cType = EBEE;
      for(uint32_t cEta = 1; cEta <= 28; cEta++) {
	for(uint32_t iPhi = 0; iPhi < 4; iPhi++) {
	  if(getLinkStatus(cType, negativeEta, cEta, iPhi) != 0 ||
	     getET(cType, negativeEta, cEta, iPhi) != 0) {
	    if(first) LogError("UCTCTP7RawData") << "EcalET FG    LinkStatus" << endl;
	    first = false;
	    LogError("UCTCTP7RawData") << dec << setfill(' ') << setw(6) << getET(cType, negativeEta, cEta, iPhi) << "  "
		 << getFB(cType, negativeEta, cEta, iPhi) << "    "
		 << showbase << internal << setfill('0') << setw(10) << hex << getLinkStatus(cType, negativeEta, cEta, iPhi)
		 << " (" << dec << getIndex(cType, negativeEta, cEta, iPhi) << ", " << negativeEta << ", " << cEta << ", " << iPhi << ")"
		 << endl;
	  }
	}
      }
      first = true;
      cType = HBHE;
      for(uint32_t cEta = 1; cEta <= 28; cEta++) {
	for(uint32_t iPhi = 0; iPhi < 4; iPhi++) {
	  if(getLinkStatus(cType, negativeEta, cEta, iPhi) != 0 ||
	     getET(cType, negativeEta, cEta, iPhi) != 0) {
	    if(first) LogError("UCTCTP7RawData") << "HcalET Feature LinkStatus" << endl;
	    first = false;
	    LogError("UCTCTP7RawData") << dec << setfill(' ') << setw(6) << getET(cType, negativeEta, cEta, iPhi) << "  "
		 << getFB(cType, negativeEta, cEta, iPhi) << "   "
		 << showbase << internal << setfill('0') << setw(10) << hex << getLinkStatus(cType, negativeEta, cEta, iPhi)
		 << " (" << dec << getIndex(cType, negativeEta, cEta, iPhi) << ", " << negativeEta << ", " << cEta << ", " << iPhi << ")"
		 << endl;
	  }
	}
      }
      first = true;
      cType = HF;
      for(uint32_t cEta = 30; cEta <= 40; cEta++) {
	for(uint32_t iPhi = 0; iPhi < 2; iPhi++) {
	  if(iPhi == 1 && cEta == 40) cEta = 41;
	  if(getLinkStatus(cType, negativeEta, cEta, iPhi) != 0 ||
	     getET(cType, negativeEta, cEta, iPhi) != 0) {
	    if(first) LogError("UCTCTP7RawData") << "HF-ET    Feature LinkStatus" << endl;
	    first = false;
	    LogError("UCTCTP7RawData") << dec << setfill(' ') << setw(6) << getET(cType, negativeEta, cEta, iPhi) << "  "
		 << dec << setfill(' ') << setw(2) << getHFFeatureBits(negativeEta, cEta, iPhi) << "   "
		 << showbase << internal << setfill('0') << setw(10) << hex << getLinkStatus(cType, negativeEta, cEta, iPhi)
		 << " (" << dec << getIndex(cType, negativeEta, cEta, iPhi) << ", " << negativeEta << ", " << cEta << ", " << iPhi << ")"
		 << endl;
	  }
	}
      }
      first = true;
      for(uint32_t region = 0; region < 7; region++) {
	if(first) LogError("UCTCTP7RawData") << "Region      ET   EGVeto  TauVeto HitLocation" << endl;
	first = false;
	LogError("UCTCTP7RawData") << dec << setfill(' ') << setw(6) << region
	     << "  " << hex << showbase << internal << setfill('0') << setw(6) << getRegionET(negativeEta, region) << dec
	     << "        " << getRegionEGVeto(negativeEta, region)
	     << "        " << getRegionTauVeto(negativeEta, region)
	     << "        " << showbase << internal << setfill('0') << setw(3) << hex << getRegionHitLocation(negativeEta, region)
	     << endl;
      }
    }
  }
  
private:

  // No copy constructor and equality operator are needed
  
  UCTCTP7RawData(const UCTCTP7RawData&);
  const UCTCTP7RawData& operator=(const UCTCTP7RawData& i);
  
  // RawData data
  
  const uint32_t* myDataPtr;

};

#endif
