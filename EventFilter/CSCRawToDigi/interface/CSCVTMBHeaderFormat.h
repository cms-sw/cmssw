#ifndef EventFilter_CSCRawToDigi_CSCVTMBHeaderFormat_h
#define EventFilter_CSCRawToDigi_CSCVTMBHeaderFormat_h

#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCShowerDigi.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"
#include <vector>
#include <iosfwd>
#include <strings.h>
class CSCDMBHeader;

class CSCVTMBHeaderFormat {
public:
  virtual ~CSCVTMBHeaderFormat() {}

  virtual void setEventInformation(const CSCDMBHeader&) = 0;
  virtual uint16_t BXNCount() const = 0;
  virtual uint16_t ALCTMatchTime() const = 0;
  virtual void setALCTMatchTime(uint16_t alctmatchtime) = 0;
  virtual uint16_t CLCTOnly() const = 0;
  virtual uint16_t ALCTOnly() const = 0;
  virtual uint16_t TMBMatch() const = 0;
  virtual uint16_t Bxn0Diff() const = 0;
  virtual uint16_t Bxn1Diff() const = 0;
  virtual uint16_t L1ANumber() const = 0;
  virtual uint16_t syncError() const = 0;
  virtual uint16_t syncErrorCLCT() const = 0;
  virtual uint16_t syncErrorMPC0() const = 0;
  virtual uint16_t syncErrorMPC1() const = 0;
  virtual uint16_t L1AMatchTime() const = 0;

  /// == Run 3 CSC-GEM Trigger Format
  virtual uint16_t clct0_ComparatorCode() const = 0;
  virtual uint16_t clct1_ComparatorCode() const = 0;
  virtual uint16_t clct0_xky() const = 0;
  virtual uint16_t clct1_xky() const = 0;
  virtual uint16_t hmt_nhits() const = 0;
  virtual uint16_t hmt_ALCTMatchTime() const = 0;
  virtual uint16_t alctHMT() const = 0;
  virtual uint16_t clctHMT() const = 0;
  virtual uint16_t gem_enabled_fibers() const = 0;
  virtual uint16_t gem_fifo_tbins() const = 0;
  virtual uint16_t gem_fifo_pretrig() const = 0;
  virtual uint16_t gem_zero_suppress() const = 0;
  virtual uint16_t gem_sync_dataword() const = 0;
  virtual uint16_t gem_timing_dataword() const = 0;
  virtual uint16_t run3_CLCT_patternID() const = 0;

  uint16_t sizeInBytes() const { return sizeInWords() * 2; }
  virtual uint16_t NTBins() const = 0;
  virtual uint16_t NCFEBs() const = 0;
  virtual void setNCFEBs(uint16_t ncfebs) = 0;
  virtual uint16_t firmwareRevision() const = 0;
  ///returns CLCT digis
  virtual std::vector<CSCCLCTDigi> CLCTDigis(uint32_t idlayer) = 0;
  ///returns CorrelatedLCT digis
  virtual std::vector<CSCCorrelatedLCTDigi> CorrelatedLCTDigis(uint32_t idlayer) const = 0;
  ///returns Run3 lct HMT Shower digi
  virtual CSCShowerDigi showerDigi(uint32_t idlayer) const = 0;
  ///returns Run3 anode HMT Shower digi
  virtual CSCShowerDigi anodeShowerDigi(uint32_t idlayer) const = 0;
  ///returns Run3 cathode HMT Shower digi
  virtual CSCShowerDigi cathodeShowerDigi(uint32_t idlayer) const = 0;

  /// in 16-bit words.  Add olne because we include beginning(b0c) and
  /// end (e0c) flags
  virtual unsigned short int sizeInWords() const = 0;

  virtual unsigned short int NHeaderFrames() const = 0;
  virtual unsigned short* data() = 0;
  virtual bool check() const = 0;

  /// for data packing
  virtual void addCLCT0(const CSCCLCTDigi& digi) = 0;
  virtual void addCLCT1(const CSCCLCTDigi& digi) = 0;
  virtual void addALCT0(const CSCALCTDigi& digi) = 0;
  virtual void addALCT1(const CSCALCTDigi& digi) = 0;
  virtual void addCorrelatedLCT0(const CSCCorrelatedLCTDigi& digi) = 0;
  virtual void addCorrelatedLCT1(const CSCCorrelatedLCTDigi& digi) = 0;
  virtual void addShower(const CSCShowerDigi& digi) = 0;
  virtual void addAnodeShower(const CSCShowerDigi& digi) = 0;
  virtual void addCathodeShower(const CSCShowerDigi& digi) = 0;

  virtual void print(std::ostream& os) const = 0;

protected:
  void swapCLCTs(CSCCLCTDigi& digi1, CSCCLCTDigi& digi2);
};

#endif
