#ifndef EventFilter_CSCRawToDigi_CSCTMBHeader_h
#define EventFilter_CSCRawToDigi_CSCTMBHeader_h

///A.Tumanov Sept 18, 07

#include <iosfwd>
#include <vector>
#include <memory>
#include "DataFormats/CSCDigi/interface/CSCTMBStatusDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCShowerDigi.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCluster.h"
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
struct CSCTMBHeader2020_TMB;
struct CSCTMBHeader2020_CCLUT;
struct CSCTMBHeader2020_GEM;
struct CSCTMBHeader2020_Run2;

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
  void setALCTMatchTime(uint16_t alctmatchtime) { theHeaderFormat->setALCTMatchTime(alctmatchtime); }
  uint16_t CLCTOnly() const { return theHeaderFormat->CLCTOnly(); }
  uint16_t ALCTOnly() const { return theHeaderFormat->ALCTOnly(); }
  uint16_t TMBMatch() const { return theHeaderFormat->TMBMatch(); }

  uint16_t Bxn0Diff() const { return theHeaderFormat->Bxn0Diff(); }
  uint16_t Bxn1Diff() const { return theHeaderFormat->Bxn1Diff(); }

  uint16_t L1ANumber() const { return theHeaderFormat->L1ANumber(); }

  uint16_t sizeInBytes() const { return theHeaderFormat->sizeInWords() * 2; }

  uint16_t L1AMatchTime() const { return theHeaderFormat->L1AMatchTime(); }
  /// will throw if the cast fails
  CSCTMBHeader2007 tmbHeader2007() const;
  CSCTMBHeader2007_rev0x50c3 tmbHeader2007_rev0x50c3() const;
  CSCTMBHeader2006 tmbHeader2006() const;
  CSCTMBHeader2013 tmbHeader2013() const;
  CSCTMBHeader2020_TMB tmbHeader2020_TMB() const;
  CSCTMBHeader2020_CCLUT tmbHeader2020_CCLUT() const;
  CSCTMBHeader2020_GEM tmbHeader2020_GEM() const;
  CSCTMBHeader2020_Run2 tmbHeader2020_Run2() const;

  uint16_t NTBins() const { return theHeaderFormat->NTBins(); }
  uint16_t NCFEBs() const { return theHeaderFormat->NCFEBs(); }

  uint16_t syncError() const { return theHeaderFormat->syncError(); }
  uint16_t syncErrorCLCT() const { return theHeaderFormat->syncErrorCLCT(); }
  uint16_t syncErrorMPC0() const { return theHeaderFormat->syncErrorMPC0(); }
  uint16_t syncErrorMPC1() const { return theHeaderFormat->syncErrorMPC1(); }

  void setNCFEBs(uint16_t ncfebs) { theHeaderFormat->setNCFEBs(ncfebs); }

  /// == Run 3 CSC-GEM Trigger Format
  uint16_t clct0_ComparatorCode() const { return theHeaderFormat->clct0_ComparatorCode(); }
  uint16_t clct1_ComparatorCode() const { return theHeaderFormat->clct1_ComparatorCode(); }
  uint16_t clct0_xky() const { return theHeaderFormat->clct0_xky(); }
  uint16_t clct1_xky() const { return theHeaderFormat->clct1_xky(); }
  uint16_t hmt_nhits() const { return theHeaderFormat->hmt_nhits(); }
  uint16_t hmt_ALCTMatchTime() const { return theHeaderFormat->hmt_ALCTMatchTime(); }
  uint16_t alctHMT() const { return theHeaderFormat->alctHMT(); }
  uint16_t clctHMT() const { return theHeaderFormat->clctHMT(); }
  uint16_t gem_enabled_fibers() const { return theHeaderFormat->gem_enabled_fibers(); }
  uint16_t gem_fifo_tbins() const { return theHeaderFormat->gem_fifo_tbins(); }
  uint16_t gem_fifo_pretrig() const { return theHeaderFormat->gem_fifo_pretrig(); }
  uint16_t gem_zero_suppress() const { return theHeaderFormat->gem_zero_suppress(); }
  uint16_t gem_sync_dataword() const { return theHeaderFormat->gem_sync_dataword(); }
  uint16_t gem_timing_dataword() const { return theHeaderFormat->gem_timing_dataword(); }
  uint16_t run3_CLCT_patternID() const { return theHeaderFormat->run3_CLCT_patternID(); }
  ///returns Run3 lct Shower Digi for HMT
  CSCShowerDigi showerDigi(uint32_t idlayer) const { return theHeaderFormat->showerDigi(idlayer); }
  ///returns Run3 anode Shower Digi for HMT
  CSCShowerDigi anodeShowerDigi(uint32_t idlayer) const { return theHeaderFormat->anodeShowerDigi(idlayer); }
  ///returns Run3 cathode Shower Digi for HMT
  CSCShowerDigi cathodeShowerDigi(uint32_t idlayer) const { return theHeaderFormat->cathodeShowerDigi(idlayer); }

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

  /// for data packing
  void addCLCT0(const CSCCLCTDigi& digi) { theHeaderFormat->addCLCT0(digi); }
  void addCLCT1(const CSCCLCTDigi& digi) { theHeaderFormat->addCLCT1(digi); }
  void addALCT0(const CSCALCTDigi& digi) { theHeaderFormat->addALCT0(digi); }
  void addALCT1(const CSCALCTDigi& digi) { theHeaderFormat->addALCT1(digi); }
  void addCorrelatedLCT0(const CSCCorrelatedLCTDigi& digi) { theHeaderFormat->addCorrelatedLCT0(digi); }
  void addCorrelatedLCT1(const CSCCorrelatedLCTDigi& digi) { theHeaderFormat->addCorrelatedLCT1(digi); }
  // Add Run3 lct Shower digi for HMT
  void addShower(const CSCShowerDigi& digi) { theHeaderFormat->addShower(digi); }
  // Add Run3 anode Shower digi for HMT
  void addAnodeShower(const CSCShowerDigi& digi) { theHeaderFormat->addAnodeShower(digi); }
  // Add Run3 cathode Shower digi for HMT
  void addCathodeShower(const CSCShowerDigi& digi) { theHeaderFormat->addCathodeShower(digi); }

  /// these methods need more brains to figure which one goes first
  void add(const std::vector<CSCCLCTDigi>& digis);
  void add(const std::vector<CSCCorrelatedLCTDigi>& digis);
  void add(const std::vector<CSCShowerDigi>& digis);

  /// tests that packing and unpacking give same results
  static void selfTest(int firmwwareVersion, int firmwareRevision);

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
