#ifndef CSCTMBHeader_h
#define CSCTMBHeader_h

///A.Tumanov Sept 18, 07

#include <iosfwd>
#include <vector>
#include <memory>
#include "DataFormats/CSCDigi/interface/CSCTMBStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "EventFilter/CSCRawToDigi/interface/CSCVTMBHeaderFormat.h"
#include "FWCore/Utilities/interface/Exception.h"

#ifndef LOCAL_UNPACK
#include <atomic>
#endif
class CSCDMBHeader;
struct CSCTMBHeader2006;
struct CSCTMBHeader2007;
struct CSCTMBHeader2007_rev0x50c3;
struct CSCTMBHeader2013;

class CSCTMBHeader {
public:
  CSCTMBHeader(int firmwareVersion, int firmwareRevision);
  CSCTMBHeader(const CSCTMBStatusDigi& digi);
  CSCTMBHeader(const unsigned short* buf);

  /// fills fields like bxn and l1a
  void setEventInformation(const CSCDMBHeader& dmbHeader) { theHeaderFormat->setEventInformation(dmbHeader); }

  int FirmwareVersion() const { return theFirmwareVersion; }
  int FirmwareRevision() const { return theHeaderFormat->firmwareRevision(); }

  uint16_t BXNCount() const { return theHeaderFormat->BXNCount(); }
  uint16_t ALCTMatchTime() const { return theHeaderFormat->ALCTMatchTime(); }
  uint16_t CLCTOnly() const { return theHeaderFormat->CLCTOnly(); }
  uint16_t ALCTOnly() const { return theHeaderFormat->ALCTOnly(); }
  uint16_t TMBMatch() const { return theHeaderFormat->TMBMatch(); }

  uint16_t Bxn0Diff() const { return theHeaderFormat->Bxn0Diff(); }
  uint16_t Bxn1Diff() const { return theHeaderFormat->Bxn1Diff(); }

  uint16_t L1ANumber() const { return theHeaderFormat->L1ANumber(); }

  uint16_t sizeInBytes() const { return theHeaderFormat->sizeInWords() * 2; }

  /// will throw if the cast fails
  CSCTMBHeader2007 tmbHeader2007() const;
  CSCTMBHeader2007_rev0x50c3 tmbHeader2007_rev0x50c3() const;
  CSCTMBHeader2006 tmbHeader2006() const;
  CSCTMBHeader2013 tmbHeader2013() const;

  uint16_t NTBins() const { return theHeaderFormat->NTBins(); }
  uint16_t NCFEBs() const { return theHeaderFormat->NCFEBs(); }

  uint16_t syncError() const { return theHeaderFormat->syncError(); }
  uint16_t syncErrorCLCT() const { return theHeaderFormat->syncErrorCLCT(); }
  uint16_t syncErrorMPC0() const { return theHeaderFormat->syncErrorMPC0(); }
  uint16_t syncErrorMPC1() const { return theHeaderFormat->syncErrorMPC1(); }

  void setNCFEBs(uint16_t ncfebs) { theHeaderFormat->setNCFEBs(ncfebs); }

  ///returns CLCT digis
  std::vector<CSCCLCTDigi> CLCTDigis(uint32_t idlayer) { return theHeaderFormat->CLCTDigis(idlayer); }

  ///returns CorrelatedLCT digis
  std::vector<CSCCorrelatedLCTDigi> CorrelatedLCTDigis(uint32_t idlayer) const {
    return theHeaderFormat->CorrelatedLCTDigis(idlayer);
  }

  /// in 16-bit words.  Add olne because we include beginning(b0c) and
  /// end (e0c) flags
  unsigned short int sizeInWords() const { return theHeaderFormat->sizeInWords(); }

  unsigned short int NHeaderFrames() const { return theHeaderFormat->NHeaderFrames(); }

  unsigned short* data() { return theHeaderFormat->data(); }

  /** turns on/off debug flag for this class */
  static void setDebug(const bool value) { debug = value; }

  bool check() const { return theHeaderFormat->check(); }

  /// Needed before data packing
  //void setChamberId(const CSCDetId & detId) {theChamberId = detId;}

  /// for data packing
  void addCLCT0(const CSCCLCTDigi& digi) { theHeaderFormat->addCLCT0(digi); }
  void addCLCT1(const CSCCLCTDigi& digi) { theHeaderFormat->addCLCT1(digi); }
  void addALCT0(const CSCALCTDigi& digi) { theHeaderFormat->addALCT0(digi); }
  void addALCT1(const CSCALCTDigi& digi) { theHeaderFormat->addALCT1(digi); }
  void addCorrelatedLCT0(const CSCCorrelatedLCTDigi& digi) { theHeaderFormat->addCorrelatedLCT0(digi); }
  void addCorrelatedLCT1(const CSCCorrelatedLCTDigi& digi) { theHeaderFormat->addCorrelatedLCT1(digi); }

  /// these methods need more brains to figure which one goes first
  void add(const std::vector<CSCCLCTDigi>& digis);
  void add(const std::vector<CSCCorrelatedLCTDigi>& digis);

  /// tests that packing and unpacking give same results
  static void selfTest();

  friend std::ostream& operator<<(std::ostream& os, const CSCTMBHeader& hdr);

private:
  //void swapCLCTs(CSCCLCTDigi& digi1, CSCCLCTDigi& digi2);

#ifdef LOCAL_UNPACK
  static bool debug;
#else
  static std::atomic<bool> debug;
#endif

  std::shared_ptr<CSCVTMBHeaderFormat> theHeaderFormat;
  int theFirmwareVersion;
};

#endif
