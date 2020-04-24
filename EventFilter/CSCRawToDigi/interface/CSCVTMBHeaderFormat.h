#ifndef CSCVTMBHeaderFormat_h
#define CSCVTMBHeaderFormat_h

#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include <vector>
#include <iosfwd>
#include <strings.h>
class CSCDMBHeader;


class CSCVTMBHeaderFormat {
public:
  virtual ~CSCVTMBHeaderFormat() {}
  void init() {
    bzero(this, sizeInWords()*2);
  }
  
  virtual void setEventInformation(const CSCDMBHeader &) = 0;
  virtual uint16_t BXNCount() const = 0;
  virtual uint16_t ALCTMatchTime() const = 0;
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
  uint16_t sizeInBytes() const {
    return sizeInWords()*2;
  }
  virtual uint16_t NTBins() const = 0;
  virtual uint16_t NCFEBs() const = 0;
  virtual void setNCFEBs(uint16_t ncfebs) = 0;
  virtual uint16_t firmwareRevision() const = 0;
  ///returns CLCT digis
  virtual std::vector<CSCCLCTDigi> CLCTDigis(uint32_t idlayer) = 0;
  ///returns CorrelatedLCT digis
  virtual std::vector<CSCCorrelatedLCTDigi> CorrelatedLCTDigis(uint32_t idlayer) const = 0;
 
  
  /// in 16-bit words.  Add olne because we include beginning(b0c) and
  /// end (e0c) flags
  virtual unsigned short int sizeInWords() const = 0;

  virtual unsigned short int NHeaderFrames() const = 0;
  virtual unsigned short * data() = 0;
  virtual bool check() const = 0;

  /// Needed before data packing
  //void setChamberId(const CSCDetId & detId) {theChamberId = detId;}

  /// for data packing
  virtual void addCLCT0(const CSCCLCTDigi & digi) = 0;
  virtual void addCLCT1(const CSCCLCTDigi & digi) = 0;
  virtual void addALCT0(const CSCALCTDigi & digi) = 0;
  virtual void addALCT1(const CSCALCTDigi & digi) = 0;
  virtual void addCorrelatedLCT0(const CSCCorrelatedLCTDigi & digi) = 0;
  virtual void addCorrelatedLCT1(const CSCCorrelatedLCTDigi & digi) = 0;


  virtual void print(std::ostream & os) const = 0;
protected:

  void swapCLCTs(CSCCLCTDigi& digi1, CSCCLCTDigi& digi2);
};

#endif

