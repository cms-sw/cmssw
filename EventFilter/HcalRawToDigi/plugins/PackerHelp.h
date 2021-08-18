#ifndef EventFilter_HcalRawToDigi_PackerHelp_h
#define EventFilter_HcalRawToDigi_PackerHelp_h

#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"
#include "DataFormats/HcalDigi/interface/QIE11DataFrame.h"

#include <iostream>
#include <cstdio>
#include <vector>
#include <map>
#include <cmath>

namespace CDFHeaderSpec {
  static const int OFFSET_H = 0;
  static const int MASK_H = 0x8;
  static const int OFFSET_FOV = 4;
  static const int MASK_FOV = 0xF;
  static const int OFFSET_SOURCE_ID = 8;
  static const int MASK_SOURCE_ID = 0xFFF;
  static const int OFFSET_BX_ID = 20;
  static const int MASK_BX_ID = 0xFFF;
  static const uint64_t OFFSET_LV1_ID = 32;
  static const int MASK_LV1_ID = 0xFFFFFF;
  static const int OFFSET_EVT_TY = 56;
  static const int MASK_EVT_TY = 0xF;
  static const int OFFSET_FIXED_MSB = 60;
  static const int MASK_FIXED_MSB = 0xF;
  static const uint64_t FIXED_MSB = 0x5;
}  // namespace CDFHeaderSpec

namespace AMC13HeaderSpec {
  static const int OFFSET_FIXED_LSB = 0;
  static const int MASK_FIXED_LSB = 0xF;
  static const int FIXED_LSB = 0x0;
  static const int OFFSET_ORN = 4;
  static const int MASK_ORN = 0xFFFFFFFF;
  static const int OFFSET_RESERVED = 36;
  static const int MASK_RESERVED = 0xFFFF;
  static const int OFFSET_NAMC = 52;
  static const int MASK_NAMC = 0xF;
  static const int OFFSET_RES = 56;
  static const int MASK_RES = 0xF;
  static const int OFFSET_UFOV = 60;
  static const int MASK_UFOV = 0xF;
}  // namespace AMC13HeaderSpec

namespace AMCHeaderSpec {
  static const int OFFSET_CRATE_ID = 0;
  static const int MASK_CRATE_ID = 0xFF;
  static const int OFFSET_SLOT_ID = 8;
  static const int MASK_SLOT_ID = 0xF;
  static const int OFFSET_PRESAMPLES = 12;
  static const int MASK_PRESAMPLES = 0xF;
  static const int OFFSET_AMC_NO = 16;
  static const int MASK_AMC_NO = 0xF;
  static const int OFFSET_BLK_NO = 20;
  static const int MASK_BLK_NO = 0xFF;
  static const int OFFSET_FIXED = 28;
  static const int MASK_FIXED = 0xF;
  static const int OFFSET_AMCN_SIZE = 32;
  static const int MASK_AMCN_SIZE = 0xFFFFFF;
  static const int OFFSET_C = 56;
  static const int MASK_C = 0x1;
  static const int OFFSET_V = 57;
  static const int MASK_V = 0x1;
  static const int OFFSET_P = 58;
  static const int MASK_P = 0x1;
  static const int OFFSET_E = 59;
  static const int MASK_E = 0x1;
  static const int OFFSET_S = 60;
  static const int MASK_S = 0x1;
  static const int OFFSET_M = 61;
  static const int MASK_M = 0x1;
  static const int OFFSET_L = 62;
  static const int MASK_L = 0x1;
}  // namespace AMCHeaderSpec

namespace QIE8HeaderSpec {
  static const int OFFSET_FIBERCHAN = 0;
  static const int MASK_FIBERCHAN = 0x3;
  static const int OFFSET_FIBER = 2;
  static const int MASK_FIBER = 0x1F;
  static const int OFFSET_CAPID = 8;
  static const int MASK_CAPID = 0x3;
  static const int OFFSET_FIBERERR = 10;
  static const int MASK_FIBERERR = 0x3;
  static const int OFFSET_FLAVOR = 12;
  static const int MASK_FLAVOR = 0x7;
  static const int OFFSET_HEADER_BIT = 15;
  static const int MASK_HEADER_BIT = 0x1;
  static const int OFFSET_TECHNICAL_DATA_TYPE = 8;
  static const int MASK_TECHNICAL_DATA_TYPE = 0xF;
}  // namespace QIE8HeaderSpec

namespace QIE10HeaderSpec {
  static const int OFFSET_FIBERCHAN = 0;
  static const int MASK_FIBERCHAN = 0x7;
  static const int OFFSET_FIBER = 3;
  static const int MASK_FIBER = 0x1F;
  static const int OFFSET_MP = 8;  // mark-and-pass
  static const int MASK_MP = 0x1;
  static const int OFFSET_RESERV = 9;  // reserved
  static const int MASK_RESERV = 0x3;
  static const int OFFSET_LE = 11;  // link error
  static const int MASK_LE = 0x1;
  static const int OFFSET_FLAVOR = 12;
  static const int MASK_FLAVOR = 0x7;
  static const int OFFSET_HEADER_BIT = 15;
  static const int MASK_HEADER_BIT = 0x1;
}  // namespace QIE10HeaderSpec

// QIE11 specifications for various flavors

namespace QIE11HeaderSpec0 {
  static const int OFFSET_FIBERCHAN = 0;
  static const int MASK_FIBERCHAN = 0x7;
  static const int OFFSET_FIBER = 3;
  static const int MASK_FIBER = 0x1F;
  static const int OFFSET_CAPID = 8;
  static const int MASK_CAPID = 0x3;
  static const int OFFSET_FIBERERR = 10;
  static const int MASK_FIBERERR = 0x3;
  static const int OFFSET_FLAVOR = 12;
  static const int MASK_FLAVOR = 0x7;
  static const int OFFSET_HEADER_BIT = 15;
  static const int MASK_HEADER_BIT = 0x1;
}  // namespace QIE11HeaderSpec0

namespace QIE11HeaderSpec3 {
  static const int OFFSET_FIBERCHAN = 0;
  static const int MASK_FIBERCHAN = 0x7;
  static const int OFFSET_FIBER = 3;
  static const int MASK_FIBER = 0x1F;
  static const int OFFSET_MP = 8;
  static const int MASK_MP = 0x1;
  static const int OFFSET_LINKERROR = 11;
  static const int MASK_LINKERROR = 0x1;
  static const int OFFSET_FLAVOR = 12;
  static const int MASK_FLAVOR = 0x7;
  static const int OFFSET_HEADER_BIT = 15;
  static const int MASK_HEADER_BIT = 0x1;
}  // namespace QIE11HeaderSpec3

namespace TPHeaderSpec {
  static const int OFFSET_TOWER = 0;
  static const int MASK_TOWER = 0xF;
  static const int OFFSET_LINK = 4;
  static const int MASK_LINK = 0xF;
  static const int OFFSET_CHANID = 0;
  static const int MASK_CHANID = 0xFF;
  static const int OFFSET_RESV = 8;
  static const int MASK_RESV = 0x3;
  static const int OFFSET_TPERR = 10;
  static const int MASK_TPERR = 0x3;
  static const int OFFSET_FLAVOR = 12;
  static const int MASK_FLAVOR = 0x7;
  static const int OFFSET_HEADER_BIT = 15;
  static const int MASK_HEADER_BIT = 0x1;
  static const int OFFSET_SOI_BIT = 14;
  static const int MASK_SOI_BIT = 0x1;
}  // namespace TPHeaderSpec

namespace QIE8SampleSpec {
  static const int OFFSET_ADC = 0;
  static const int MASK_ADC = 0x7F;
  static const int OFFSET_CAPID = 8;
  static const int MASK_CAPID = 0x3;
  static const int OFFSET_DV = 10;
  static const int MASK_DV = 0x1;
  static const int OFFSET_ER = 11;
  static const int MASK_ER = 0x1;
}  // namespace QIE8SampleSpec

class HCalFED {
public:
  typedef std::vector<uint16_t> uhtrData;

  std::vector<unsigned char> fedData;
  std::vector<uint64_t> AMCHeaders;
  std::vector<uhtrData> uhtrs;
  int fedId;
  uint64_t AMC13Header, cdfHeader;
  uint64_t OrbitNum;
  uint64_t EventNum;
  uint64_t BxNum;
  uint64_t Crate;

  unsigned char cdfh[8];
  FEDHeader* rawFEDHeader;

  HCalFED(int fedId_, uint64_t EventNum_ = 9999, uint64_t OrbitNum_ = 999, uint64_t BxNum_ = 99) {
    fedId = fedId_;
    OrbitNum = OrbitNum_;
    EventNum = EventNum_;
    BxNum = BxNum_;
    Crate = fedId - FEDNumbering::MINHCALuTCAFEDID;

    setCDFHeader();
    setAMC13Header();
  };

  inline void split64bitTo8bit(std::vector<unsigned char>& outVec, const uint64_t& var64bit) {
    for (int is = 0; is < 8; is++) {  // 64/8 = 8
      outVec.push_back((var64bit >> (is * 8)) & 0xFF);
    }
  }

  void setCDFHeader() {
    cdfHeader = 0;
    cdfHeader |= (0x1 & CDFHeaderSpec::MASK_H) << CDFHeaderSpec::OFFSET_H;
    cdfHeader |= (0x0 & CDFHeaderSpec::MASK_FOV) << CDFHeaderSpec::OFFSET_FOV;
    cdfHeader |= ((fedId)&CDFHeaderSpec::MASK_SOURCE_ID)
                 << CDFHeaderSpec::OFFSET_SOURCE_ID;                                  // needs to be configurable
    cdfHeader |= (BxNum & CDFHeaderSpec::MASK_BX_ID) << CDFHeaderSpec::OFFSET_BX_ID;  // needs to be configurable
    cdfHeader |= (uint64_t(0x1) & CDFHeaderSpec::MASK_LV1_ID) << CDFHeaderSpec::OFFSET_LV1_ID;
    cdfHeader |= (uint64_t(0x1) & CDFHeaderSpec::MASK_EVT_TY) << CDFHeaderSpec::OFFSET_EVT_TY;
    cdfHeader |= (CDFHeaderSpec::FIXED_MSB & CDFHeaderSpec::MASK_FIXED_MSB) << CDFHeaderSpec::OFFSET_FIXED_MSB;
  }

  void setAMC13Header() {
    AMC13Header = 0;
    AMC13Header |= (AMC13HeaderSpec::FIXED_LSB & AMC13HeaderSpec::MASK_FIXED_LSB) << AMC13HeaderSpec::OFFSET_FIXED_LSB;
    AMC13Header |= ((OrbitNum + 1) & AMC13HeaderSpec::MASK_ORN) << AMC13HeaderSpec::OFFSET_ORN;
    AMC13Header |= (uint64_t(0x0) & AMC13HeaderSpec::MASK_RESERVED) << AMC13HeaderSpec::OFFSET_RESERVED;
    AMC13Header |= (uint64_t(0x0) & AMC13HeaderSpec::MASK_NAMC) << AMC13HeaderSpec::OFFSET_NAMC;
    AMC13Header |= (uint64_t(0x0) & AMC13HeaderSpec::MASK_RES) << AMC13HeaderSpec::OFFSET_RES;
    AMC13Header |= (uint64_t(0x1) & AMC13HeaderSpec::MASK_UFOV) << AMC13HeaderSpec::OFFSET_UFOV;
  }

  void setNAMC(uint64_t NAMC) { AMC13Header |= (NAMC & AMC13HeaderSpec::MASK_NAMC) << AMC13HeaderSpec::OFFSET_NAMC; }

  void addAMCHeader(uint64_t crate, uint64_t slot, uint64_t AMCsize, uint64_t presamples = 10, uint64_t blockNum = 0) {
    uint64_t header = 0;
    header |= (crate & AMCHeaderSpec::MASK_CRATE_ID) << AMCHeaderSpec::OFFSET_CRATE_ID;
    header |= (slot & AMCHeaderSpec::MASK_SLOT_ID) << AMCHeaderSpec::OFFSET_SLOT_ID;
    header |= (presamples & AMCHeaderSpec::MASK_PRESAMPLES) << AMCHeaderSpec::OFFSET_PRESAMPLES;  // boardId
    header |= (slot & AMCHeaderSpec::MASK_AMC_NO) << AMCHeaderSpec::OFFSET_AMC_NO;                // AMC no.
    header |= (blockNum & AMCHeaderSpec::MASK_BLK_NO) << AMCHeaderSpec::OFFSET_BLK_NO;            // Block No.
    header |= (AMCsize & AMCHeaderSpec::MASK_AMCN_SIZE) << AMCHeaderSpec::OFFSET_AMCN_SIZE;       // size
    header |= uint64_t(0x1) << AMCHeaderSpec::OFFSET_C;                                           // CRC is valid
    header |= uint64_t(0x1) << AMCHeaderSpec::OFFSET_V;                                           // EvN, BcN match
    header |= uint64_t(0x1) << AMCHeaderSpec::OFFSET_P;  // Present, header is only made if data is present
    header |= uint64_t(0x1) << AMCHeaderSpec::OFFSET_E;  // Enabled, header is only made if AMC is enabled
    header |= uint64_t(0x0) << AMCHeaderSpec::OFFSET_S;  // Segmented, always zero for unsegmented data
    header |= uint64_t(0x0) << AMCHeaderSpec::OFFSET_M;  // More data
    header |= uint64_t(0x0) << AMCHeaderSpec::OFFSET_L;  // Indicates length error

    AMCHeaders.push_back(header);
  }

  void addUHTR(uhtrData uhtr, uint64_t crate, uint64_t slot) {
    // push uhtr data into FED container
    uhtrs.push_back(uhtr);
    // create the corresponding AMC header
    addAMCHeader(crate, slot, uhtr.size() / 4);
  };

  // does not include HEADER and TRAILER
  void formatFEDdata(FEDRawData& rawData) {
    //std::vector<unsigned char> output;
    if (uhtrs.size() != AMCHeaders.size()) {
      return;
    }

    // put common data format header in fed container
    split64bitTo8bit(fedData, cdfHeader);

    // set the number of AMCs in the AMC13 header
    setNAMC(uhtrs.size());
    // put the AMC13 header into the fed container
    split64bitTo8bit(fedData, AMC13Header);

    // fill fedData with AMC headers
    for (unsigned int iAMC = 0; iAMC < AMCHeaders.size(); ++iAMC) {
      // adjust the AMCsize bits to match uhtr header
      //AMCHeaders[iAMC] |= uint64_t(uhtrs[iAMC][1]&0xF)<<51 ;
      //AMCHeaders[iAMC] |= uint64_t(uhtrs[iAMC][0]&0xFFFF)<<47 ;
      split64bitTo8bit(fedData, AMCHeaders[iAMC]);
    }

    // fill fedData with AMC data
    for (unsigned int iAMC = 0; iAMC < uhtrs.size(); ++iAMC) {
      unsigned int nWords16 = uhtrs[iAMC].size();
      for (unsigned int amcWord = 0; amcWord < nWords16; ++amcWord) {
        fedData.push_back((uhtrs[iAMC][amcWord] >> 0) & 0xFF);  // split 16-bit words into 8-bit
        fedData.push_back((uhtrs[iAMC][amcWord] >> 8) & 0xFF);
      }  // end loop over uhtr words
    }    // end loop over uhtrs

    // fedData should be a integer number of 64 bit words
    while (fedData.size() % 8 != 0)
      fedData.push_back(0);

    // convert to the correct format
    rawData.resize(fedData.size());
    unsigned char* words = reinterpret_cast<unsigned char*>(rawData.data());

    for (unsigned int i = 0; i < fedData.size(); ++i) {
      *words = fedData[i];
      words++;
    }
  };
};

class UHTRpacker {
public:
  typedef std::vector<uint16_t> uhtrData;
  typedef std::map<int, uhtrData> UHTRMap;  // the int here is intended to be the raw electronics ID

  UHTRMap uhtrs;

  // FIRST WORD
  static const int OFFSET_DATA_LENGTH = 0;
  static const int MASK_DATA_LENGTH = 0xFFFFF;
  static const int OFFSET_BCN = 20;
  static const int MASK_BCN = 0xFFF;
  static const int OFFSET_EVN = 32;
  static const int MASK_EVN = 0xFFFFFF;
  static const int OFFSET_FILED_BY_AMC13 = 56;
  static const int MASK_FILED_BY_AMC13 = 0xFF;
  // SECOND WORD
  static const int OFFSET_CRATE_ID = 0;
  static const int MASK_CRATE_ID = 0xFFFFFF;
  static const int OFFSET_SLOT_ID = 8;
  static const int MASK_SLOT_ID = 0xF;
  static const int OFFSET_PRESAMPLES = 12;
  static const int MASK_PRESAMPLES = 0xF;
  static const int OFFSET_ORN = 16;
  static const int MASK_ORN = 0xFFFF;
  static const int OFFSET_FW_FLAVOR = 32;
  static const int MASK_FW_FLAVOR = 0xFF;
  static const int OFFSET_EVENT_TYPE = 40;
  static const int MASK_EVENT_TYPE = 0xF;
  static const int OFFSET_PAYLOAD_FORMAT = 44;
  static const int MASK_PAYLOAD_FORMAT = 0xF;
  static const int OFFSET_FW_VERSION = 48;
  static const int MASK_FW_VERSION = 0xFFFF;

  UHTRpacker() {}

  bool exist(int uhtrIndex) { return uhtrs.count(uhtrIndex) != 0; };

  // flavor should be 5, or 7 (only for premixing in sim)
  uint16_t packQIE8header(const HcalQIESample& qieSample, const HcalElectronicsId& eid, int flavor) {
    uint16_t header = 0;

    int fiber = eid.fiberIndex() + 1;
    int fiberchan = eid.fiberChanId();

    header |= (fiberchan & QIE8HeaderSpec::MASK_FIBERCHAN) << QIE8HeaderSpec::OFFSET_FIBERCHAN;
    header |= ((fiber - 1) & QIE8HeaderSpec::MASK_FIBER) << QIE8HeaderSpec::OFFSET_FIBER;
    if (flavor == 7) {
      header |= (15 & QIE8HeaderSpec::MASK_TECHNICAL_DATA_TYPE) << QIE8HeaderSpec::OFFSET_TECHNICAL_DATA_TYPE;
    } else {
      int fiberErr = qieSample.er();
      int capid0 = qieSample.capid();
      header |= (capid0 & QIE8HeaderSpec::MASK_CAPID) << QIE8HeaderSpec::OFFSET_CAPID;
      header |= (fiberErr & QIE8HeaderSpec::MASK_FIBERERR) << QIE8HeaderSpec::OFFSET_FIBERERR;
    }
    header |= (flavor & QIE8HeaderSpec::MASK_FLAVOR) << QIE8HeaderSpec::OFFSET_FLAVOR;  //flavor
    header |= (0x1 & QIE8HeaderSpec::MASK_HEADER_BIT) << QIE8HeaderSpec::OFFSET_HEADER_BIT;

    return header;
  }

  uint16_t packQIE8sample(const HcalQIESample& qieSample) {
    uint16_t sample = 0;

    int adc = qieSample.adc();
    int capid = qieSample.capid();
    int dv = qieSample.dv();
    int er = qieSample.er();

    sample |= (adc & QIE8SampleSpec::MASK_ADC) << QIE8SampleSpec::OFFSET_ADC;
    sample |= (capid & QIE8SampleSpec::MASK_CAPID) << QIE8SampleSpec::OFFSET_CAPID;
    sample |= (dv & QIE8SampleSpec::MASK_DV) << QIE8SampleSpec::OFFSET_DV;
    sample |= (er & QIE8SampleSpec::MASK_ER) << QIE8SampleSpec::OFFSET_ER;

    return sample;
  }

  uint16_t packTPheader(const HcalTriggerPrimitiveSample& tpSample, int channelid) {
    uint16_t header = 0;

    header |= (channelid & TPHeaderSpec::MASK_CHANID) << TPHeaderSpec::OFFSET_CHANID;
    header |= (0x0 & TPHeaderSpec::MASK_RESV) << TPHeaderSpec::OFFSET_RESV;
    header |= (0 & TPHeaderSpec::MASK_TPERR) << TPHeaderSpec::OFFSET_TPERR;
    header |= (0x4 & TPHeaderSpec::MASK_FLAVOR) << TPHeaderSpec::OFFSET_FLAVOR;  //flavor
    header |= (0x1 & TPHeaderSpec::MASK_HEADER_BIT) << TPHeaderSpec::OFFSET_HEADER_BIT;

    return header;
  }

  uint16_t packQIE10header(const HcalElectronicsId& eid) {
    uint16_t header = 0;

    int fiber = eid.fiberIndex();
    int fiberchan = eid.fiberChanId();

    header |= (fiberchan & QIE10HeaderSpec::MASK_FIBERCHAN) << QIE10HeaderSpec::OFFSET_FIBERCHAN;
    header |= (fiber & QIE10HeaderSpec::MASK_FIBER) << QIE10HeaderSpec::OFFSET_FIBER;
    header |= (0x0 & QIE10HeaderSpec::MASK_MP) << QIE10HeaderSpec::OFFSET_MP;
    header |= (0x0 & QIE10HeaderSpec::MASK_RESERV) << QIE10HeaderSpec::OFFSET_RESERV;
    header |= (0x0 & QIE10HeaderSpec::MASK_LE) << QIE10HeaderSpec::OFFSET_LE;
    header |= (0x2 & QIE10HeaderSpec::MASK_FLAVOR) << QIE10HeaderSpec::OFFSET_FLAVOR;  //flavor
    header |= (0x1 & QIE10HeaderSpec::MASK_HEADER_BIT) << QIE10HeaderSpec::OFFSET_HEADER_BIT;

    return header;
  }

  uint16_t packQIE11header(const QIE11DataFrame& qiedf, const HcalElectronicsId& eid) {
    uint16_t header = 0;

    int fiber = eid.fiberIndex();
    int fiberchan = eid.fiberChanId();
    int flavor = qiedf[0].flavor();

    if (flavor == 3) {
      header |= (fiberchan & QIE11HeaderSpec3::MASK_FIBERCHAN) << QIE11HeaderSpec3::OFFSET_FIBERCHAN;
      header |= (fiber & QIE11HeaderSpec3::MASK_FIBER) << QIE11HeaderSpec3::OFFSET_FIBER;
      header |= (0x0 & QIE11HeaderSpec3::MASK_MP) << QIE11HeaderSpec3::OFFSET_MP;
      header |= (0x0 & QIE11HeaderSpec3::MASK_LINKERROR) << QIE11HeaderSpec3::OFFSET_LINKERROR;
      header |= (flavor & QIE11HeaderSpec3::MASK_FLAVOR) << QIE11HeaderSpec3::OFFSET_FLAVOR;  //flavor
      header |= (0x1 & QIE11HeaderSpec3::MASK_HEADER_BIT) << QIE11HeaderSpec3::OFFSET_HEADER_BIT;
    } else {
      int capid0 = qiedf[0].capid();
      header |= (fiberchan & QIE11HeaderSpec0::MASK_FIBERCHAN) << QIE11HeaderSpec0::OFFSET_FIBERCHAN;
      header |= (fiber & QIE11HeaderSpec0::MASK_FIBER) << QIE11HeaderSpec0::OFFSET_FIBER;
      header |= (capid0 & QIE11HeaderSpec0::MASK_CAPID) << QIE11HeaderSpec0::OFFSET_CAPID;
      header |= (0x0 & QIE11HeaderSpec0::MASK_FIBERERR) << QIE11HeaderSpec0::OFFSET_FIBERERR;
      header |= (flavor & QIE11HeaderSpec0::MASK_FLAVOR) << QIE11HeaderSpec0::OFFSET_FLAVOR;  //flavor
      header |= (0x1 & QIE11HeaderSpec0::MASK_HEADER_BIT) << QIE11HeaderSpec0::OFFSET_HEADER_BIT;
    }

    return header;
  }

  uhtrData* newUHTR(int uhtrIndex, int ps = 0, int orn = 0, int bcn = 0, uint64_t evt = 0) {
    // initialize vector of 16-bit words
    uhtrs[uhtrIndex] = uhtrData(8);
    // build header -- some information will be updated at the end

    uint64_t presamples = std::max(ps, 0);
    uint64_t uhtrCrate = uhtrIndex & 0xFF;
    uint64_t uhtrSlot = (uhtrIndex & 0xF00) >> 8;
    // From Jeremy:
    // Set the firmware to zero, the firmware flavor to 0, the payload format to 1, and the event type to 1.
    uint64_t fwFlavor = 0;
    uint64_t eventType = 1;
    uint64_t payloadFormat = 1;
    uint64_t fwVersion = 0;

    uint64_t uhtrHeader1 = 0;
    uhtrHeader1 |= (uint64_t(0x0) & MASK_DATA_LENGTH) << OFFSET_DATA_LENGTH;
    uhtrHeader1 |= (bcn & MASK_BCN) << OFFSET_BCN;
    uhtrHeader1 |= (evt & MASK_EVN) << OFFSET_EVN;
    uhtrHeader1 |= (uint64_t(0x0) & MASK_FILED_BY_AMC13) << OFFSET_FILED_BY_AMC13;

    uint64_t uhtrHeader2 = 0;
    uhtrHeader2 |= (uhtrCrate & MASK_CRATE_ID) << OFFSET_CRATE_ID;
    uhtrHeader2 |= (uhtrSlot & MASK_SLOT_ID) << OFFSET_SLOT_ID;
    uhtrHeader2 |= (presamples & MASK_PRESAMPLES) << OFFSET_PRESAMPLES;
    uhtrHeader2 |= (orn & MASK_ORN) << OFFSET_ORN;
    uhtrHeader2 |= (fwFlavor & MASK_FW_FLAVOR) << OFFSET_FW_FLAVOR;
    uhtrHeader2 |= (eventType & MASK_EVENT_TYPE) << OFFSET_EVENT_TYPE;
    uhtrHeader2 |= (payloadFormat & MASK_PAYLOAD_FORMAT) << OFFSET_PAYLOAD_FORMAT;
    uhtrHeader2 |= (fwVersion & MASK_FW_VERSION) << OFFSET_FW_VERSION;

    // push header into vector of 16-bit words
    for (unsigned int i = 0; i < 4; ++i) {
      uhtrs[uhtrIndex][i] = (uhtrHeader1 >> (i * 16)) & 0xFFFF;
      uhtrs[uhtrIndex][i + 4] = (uhtrHeader2 >> (i * 16)) & 0xFFFF;
    }

    return &(uhtrs[uhtrIndex]);
  };

  void finalizeHeadTail(uhtrData* uhtr, bool verbosity) {
    uint64_t uhtr_size = uhtr->size() - 8;

    // adjust the size bits
    uhtr->at(0) = uhtr_size & 0xFFFF;
    uhtr->at(1) |= (uhtr_size >> 16) & 0xF;

    unsigned int toAdd = 4 - uhtr->size() % 4;
    for (unsigned int ia = 0; ia < toAdd; ia++) {
      uhtr->push_back(0xD07F);
    }

    // add trailer
    uhtr->push_back(uhtr_size & 0xFFFF);
    uhtr->push_back((uhtr_size >> 16) & 0xF);
    // this is ignoring the event number... I am not sure what this should be

    // adding some blank stuff for the CRC bits
    uhtr->push_back(0);
    uhtr->push_back(0);
  };

  void addChannel(int uhtrIndex,
                  edm::SortedCollection<HFDataFrame>::const_iterator& qiedf,
                  const HcalElectronicsMap* readoutMap,
                  bool premix,
                  int verbosity = 0) {
    if (qiedf->size() == 0)
      return;
    DetId detid = qiedf->id();
    HcalElectronicsId eid(readoutMap->lookup(detid));
    uint16_t header = packQIE8header(qiedf->sample(0), eid, premix ? 7 : 5);
    uhtrs[uhtrIndex].push_back(header);
    // loop over words in dataframe
    if (premix) {
      for (int iTS = 0; iTS < qiedf->size(); ++iTS) {
        uhtrs[uhtrIndex].push_back(packQIE8sample(qiedf->sample(iTS)));
      }
    } else {
      for (int iTS = 0; iTS < qiedf->size(); iTS += 2) {
        uint16_t cont = 0;
        int adc0 = qiedf->sample(iTS).adc();
        int adc1 = qiedf->sample(iTS + 1).adc();
        cont |= adc0 & 0xFF;
        cont |= (adc1 & 0xFF) << 8;
        uhtrs[uhtrIndex].push_back(cont);
      }
    }  // end loop over dataframe words
  };

  void addChannel(int uhtrIndex,
                  edm::SortedCollection<HBHEDataFrame>::const_iterator qiedf,
                  const HcalElectronicsMap* readoutMap,
                  bool premix,
                  int verbosity = 0) {
    if (qiedf->size() == 0)
      return;
    DetId detid = qiedf->id();
    HcalElectronicsId eid(readoutMap->lookup(detid));
    uint16_t header = packQIE8header(qiedf->sample(0), eid, premix ? 7 : 5);
    uhtrs[uhtrIndex].push_back(header);
    // loop over words in dataframe
    if (premix) {
      for (int iTS = 0; iTS < qiedf->size(); ++iTS) {
        uhtrs[uhtrIndex].push_back(packQIE8sample(qiedf->sample(iTS)));
      }
    } else {
      for (int iTS = 0; iTS < qiedf->size(); iTS += 2) {
        uint16_t cont = 0;
        int adc0 = qiedf->sample(iTS).adc();
        int adc1 = qiedf->sample(iTS + 1).adc();
        cont |= adc0 & 0xFF;
        cont |= (adc1 & 0xFF) << 8;
        uhtrs[uhtrIndex].push_back(cont);
      }
    }  // end loop over dataframe words
  };

  void addChannel(int uhtrIndex,
                  edm::SortedCollection<HcalTriggerPrimitiveDigi>::const_iterator qiedf,
                  int channelid,
                  int verbosity = 0) {
    if (qiedf->size() == 0)
      return;
    uint16_t header = packTPheader(qiedf->sample(0), channelid);
    uhtrs[uhtrIndex].push_back(header);
    // loop over words in dataframe
    for (int iTS = 0; iTS < qiedf->size(); iTS++) {
      // push data into uhtr data container
      auto raw = qiedf->sample(iTS).raw();
      // Add SOI information
      if (iTS == qiedf->presamples())
        raw |= TPHeaderSpec::MASK_SOI_BIT << TPHeaderSpec::OFFSET_SOI_BIT;
      uhtrs[uhtrIndex].push_back(raw);
    }  // end loop over dataframe words
  };

  void addChannel(int uhtrIndex, QIE11DataFrame qiedf, const HcalElectronicsMap* readoutMap, int verbosity = 0) {
    DetId detid = qiedf.detid();
    HcalElectronicsId eid(readoutMap->lookup(detid));
    // loop over words in dataframe
    for (edm::DataFrame::iterator dfi = qiedf.begin(); dfi != qiedf.end(); ++dfi) {
      if (dfi >= qiedf.end() - QIE11DataFrame::FLAG_WORDS) {
        continue;
      }
      if (dfi == qiedf.begin() && QIE11DataFrame::HEADER_WORDS == 1) {
        uint16_t header = packQIE11header(qiedf, eid);
        uhtrs[uhtrIndex].push_back(header);
        continue;
      }
      // push data into uhtr data container
      uhtrs[uhtrIndex].push_back(dfi[0]);
    }  // end loop over dataframe words
  };

  void addChannel(int uhtrIndex, QIE10DataFrame qiedf, const HcalElectronicsMap* readoutMap, int verbosity = 0) {
    DetId detid = qiedf.detid();
    HcalElectronicsId eid(readoutMap->lookup(detid));
    // loop over words in dataframe
    for (edm::DataFrame::iterator dfi = qiedf.begin(); dfi != qiedf.end(); ++dfi) {
      if (dfi >= qiedf.end() - QIE10DataFrame::FLAG_WORDS) {
        continue;
      }
      if (dfi == qiedf.begin() && QIE10DataFrame::HEADER_WORDS == 1) {
        uint16_t header = packQIE10header(eid);
        uhtrs[uhtrIndex].push_back(header);
        continue;
      }
      // push data into uhtr data container
      uhtrs[uhtrIndex].push_back(dfi[0]);
    }  // end loop over dataframe words
  };
};

// converts HE QIE digies to HB data format

inline QIE11DataFrame convertHB(QIE11DataFrame qiehe,
                                std::vector<int> const& tdc1,
                                std::vector<int> const& tdc2,
                                const int tdcmax) {
  QIE11DataFrame qiehb = qiehe;
  HcalDetId did = HcalDetId(qiehb.detid());
  int adc, tdc;
  bool soi;
  int is = 0;
  int capid = qiehe[0].capid();
  //  flavor for HB digies is hardcoded here
  static const int hbflavor = 3;
  //  maximum HB depth
  static const int maxHBdepth = 4;

  const int entry = (abs(did.ieta()) - 1) * maxHBdepth + did.depth() - 1;
  const int first = tdc1.at(entry);
  const int second = tdc2.at(entry);

  //  iterator over samples
  for (edm::DataFrame::const_iterator it = qiehe.begin(); it != qiehe.end(); ++it) {
    if (it == qiehe.begin())
      continue;
    adc = qiehe[is].adc();
    tdc = qiehe[is].tdc();
    soi = qiehe[is].soi();

    if (tdc >= 0 && tdc <= first)
      tdc = 0;
    else if (tdc > first && tdc <= second)
      tdc = 1;
    else if (tdc > second && tdc <= tdcmax)
      tdc = 2;
    else
      tdc = 3;

    qiehb.setSample(is, adc, tdc, soi);
    is++;
  };

  // puting flavor is safe here because flavor is stored in the same bits for all flavors
  qiehb.setFlavor(hbflavor);
  qiehb.setCapid0(capid);

  return qiehb;
}

#endif
