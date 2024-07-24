#ifndef DataFormats_HGCalDigis_HGCalFlaggedECONDInfo_h
#define DataFormats_HGCalDigis_HGCalFlaggedECONDInfo_h

#include <vector>

class HGCalFlaggedECONDInfo
{
 public:

  /**
     @short flags in the capture block header pertaining an ECON-D
   */
  enum CBFlagTypes { PAYLOADTOOLARGE=1,
		     CRCERROR=2,
		     EVENTIDMISMATCH=3,
		     FSMTIMEOUT=4,
		     BCorORBITIDMISMATCH=5,
		     MAINBUFFEROVERFLOW=6,
		     INACTIVEINPUT=7};

  /**
     @short flags in the ECON-D header
   */
  enum FlagTypes { CBSTATUS=1,
                   HTBITS=2,
                   EBOBITS=4,
                   MATCHBIT=8,
                   TRUNCATED=16,
                   WRONGHEADERMARKER=32,
                   PAYLOADOVERFLOWS=64,
                   PAYLOADMISMATCHES=128,
                   UNEXPECTEDTRUNCATED=256};
  
  HGCalFlaggedECONDInfo() : HGCalFlaggedECONDInfo(0,0,0,0,0) {}
  HGCalFlaggedECONDInfo(uint32_t loc, uint32_t cbflagbits, uint32_t flagbits, uint32_t id, uint32_t pl)
    : iword(loc), cbflags(cbflagbits), flags(flagbits), eleid(id), payload(pl) {}
  HGCalFlaggedECONDInfo(const HGCalFlaggedECONDInfo &t)
    : HGCalFlaggedECONDInfo(t.iword,t.cbflags,t.flags,t.eleid,t.payload) {}

  void setPayload(uint32_t v) { payload=v; }
  void addFlag(FlagTypes f) { flags += f; } 
  void addToFlag(uint32_t v) { flags += v; } 
  bool cbFlag() { return flags & 0x1; };
  bool htFlag() { return (flags>>1) & 0x1; };
  bool eboFlag() { return (flags>>2) & 0x1; };
  bool matchFlag() { return (flags>>3) & 0x1; };
  bool truncatedFlag() { return (flags>>4) & 0x1; };
  bool wrongHeaderMarker() { return (flags>>5) & 0x1; };
  bool payloadOverflows() { return (flags>>6) & 0x1; };
  bool payloadMismatches() { return (flags>>7) & 0x1; };
  bool unexpectedTruncated() { return (flags>>8) & 0x1; };
    
  uint32_t iword,cbflags,flags,eleid,payload;
};

typedef std::vector<HGCalFlaggedECONDInfo> HGCalFlaggedECONDInfoCollection;

#endif
