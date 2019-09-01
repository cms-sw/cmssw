//-------------------------------------------------
//
/**  \class DTuROSControlData
 *
 *   DT uROS Control Data
 *
 *
 *
 *
 *   J. Troconiz  - UAM Madrid
 */
//
//--------------------------------------------------
#ifndef DTDigi_DTuROSControlData_h
#define DTDigi_DTuROSControlData_h

#include <vector>

static const int DOCESLOTS = 12;
static const int SEISXOK = 12;

class DTuROSROSData {
public:
  /// Constructor
  DTuROSROSData() : slot_(-1), header1_(0), header2_(0), trailer_(0), okword1_(0), okword2_(0) {
    for (int i = 0; i < SEISXOK; i++)
      okxword_[i] = 0;
  }

  /// Destructor
  ~DTuROSROSData(){};

  void setslot(int slot) { slot_ = slot; }

  void setheader1(long dword) { header1_ = dword; }

  void setheader2(long dword) { header2_ = dword; }

  void settrailer(long dword) { trailer_ = dword; }

  void setokword1(long okword) { okword1_ = okword; }

  void setokword2(long okword) { okword2_ = okword; }

  void setokxword(int i, long okxword) { okxword_[i] = okxword; }

  void setexword(long exword) { exword_.push_back(exword); }

  void seterror(int error) { error_.push_back(error); }

  int getslot() const { return slot_; }

  long getheader1() const { return header1_; }

  long getheader2() const { return header2_; }

  long gettrailer() const { return trailer_; }

  long getokword1() const { return okword1_; }

  long getokword2() const { return okword2_; }

  int getokflag(int i) const {
    if (i < 60)
      return ((okword1_ >> i) & 0x1);
    return ((okword2_ >> (i - 60)) & 0x1);
  }

  long getokxword(int i) const { return okxword_[i]; }

  int getokxflag(int i) const { return ((okxword_[i / 12] >> (5 * (i % 12))) & 0x1F); }

  std::vector<long> getexwords() const { return exword_; }

  long getexword(int i) const { return exword_.at(i); }

  std::vector<int> geterrors() const { return error_; }

  int geterror(int i) const { return error_.at(i); }

  int geterrorROBID(int i) const { return (error_.at(i) >> 21) & 0x7F; }

  int geterrorTDCID(int i) const { return (error_.at(i) >> 19) & 0x3; }

  int geterrorFlag(int i) const { return (error_.at(i)) & 0x7FFF; }

  int getboardId() const { return (getheader2()) & 0xFFFF; }

  int getuserWord() const { return (getheader2() >> 32) & 0xFFFFFFFF; }

private:
  int slot_;

  long header1_, header2_, trailer_;

  long okword1_, okword2_, okxword_[SEISXOK];

  std::vector<long> exword_;

  std::vector<int> error_;
};

class DTuROSFEDData {
public:
  /// Constructor
  DTuROSFEDData() : header1_(0), header2_(0), trailer_(0), fed_(-1), nslots_(0), evtLgth_(0) {
    for (int i = 0; i < DOCESLOTS; i++)
      rsize_[i] = 0;
  }

  /// Destructor
  ~DTuROSFEDData(){};

  void setfed(int fed) { fed_ = fed; }

  void setheader1(long dword) { header1_ = dword; }

  void setheader2(long dword) { header2_ = dword; }

  void settrailer(long dword) { trailer_ = dword; }

  void setnslots(int nslots) { nslots_ = nslots; }

  void setevtlgth(int evtLgth) { evtLgth_ = evtLgth; }

  void setslotsize(int slot, int size) { rsize_[slot - 1] = size; }

  void setuROS(int slot, DTuROSROSData rwords) { rdata_[slot - 1] = rwords; }

  int getfed() const { return fed_; }

  long getheader1() const { return header1_; }

  long getheader2() const { return header2_; }

  long gettrailer() const { return trailer_; }

  int getnslots() const { return nslots_; }

  int getevtlgth() const { return evtLgth_; }

  int getslotsize(int slot) const { return rsize_[slot - 1]; }

  int getBXId() const { return (getheader1() >> 20) & 0xFFF; }

  int getTTS() const { return (gettrailer() >> 4) & 0xF; }

  DTuROSROSData getuROS(int slot) const { return rdata_[slot - 1]; }

private:
  long header1_, header2_, trailer_;

  int fed_, nslots_, evtLgth_, rsize_[DOCESLOTS];

  DTuROSROSData rdata_[DOCESLOTS];
};

typedef std::vector<DTuROSFEDData> DTuROSFEDDataCollection;
#endif
