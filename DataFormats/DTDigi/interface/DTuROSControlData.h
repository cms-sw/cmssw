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

#define DOCESLOTS 12


#include <vector>


class DTuROSROSData {

public:

  /// Constructor
  DTuROSROSData() {slot_ = -1;}

  /// Destructor
  ~DTuROSROSData(){};

  void setslot(int slot) {slot_ = slot;}

  void setheader1(long dword) {header1_ = dword;}

  void setheader2(long dword) {header2_ = dword;}

  void settrailer(long dword) {trailer_ = dword;}

  void setokword1(int okword) {okword1_ = okword;}

  void setokword2(int okword) {okword2_ = okword;}

  void seterror(int error) {error_.push_back(error);}

  int getslot() const {return slot_;}

  long getheader1() const {return header1_;}

  long getheader2() const {return header2_;}

  long gettrailer() const {return trailer_;}

  int getokword1() const {return okword1_;}

  int getokword2() const {return okword2_;}

  int getokflag(int i) const {if (i < 60) return ((okword1_ >> i)&0x1); return ((okword2_ >> (i-60))&0x1);}

  std::vector<int> geterrors() const {return error_;}

  int geterror(int i) const {return error_.at(i);}

  int geterrorROBID(int i) const {return (error_.at(i) >> 21)&0x7F;}

  int geterrorTDCID(int i) const {return (error_.at(i) >> 19)&0x3;}

  int geterrorFlag(int i) const {return (error_.at(i))&0x7FFF;}

private:

  long header1_, header2_, trailer_;

  int slot_, okword1_, okword2_;

  std::vector<int> error_;

};


class DTuROSFEDData {

public:

  /// Constructor
  DTuROSFEDData() {for (int i=0; i<DOCESLOTS; i++) rsize_[i] = 0;} 

  /// Destructor
  ~DTuROSFEDData(){};

  void setfed(int fed) {fed_ = fed;}

  void setheader1(long dword) {header1_ = dword;}

  void setheader2(long dword) {header2_ = dword;}

  void settrailer(long dword) {trailer_ = dword;}

  void setnslots(int nslots) {nslots_ = nslots;}

  void setevtlgth(int evtLgth) {evtLgth_ = evtLgth;}

  void setslotsize(int slot, int size) {rsize_[slot-1] = size;}

  void setuROS(int slot, DTuROSROSData rwords) {rdata_[slot-1] = rwords;}

  int  getfed() const {return fed_;}

  long getheader1() const {return header1_;}

  long getheader2() const {return header2_;}

  long gettrailer() const {return trailer_;}

  int getnslots() const {return nslots_;}

  int getevtlgth() const {return evtLgth_;}

  int getslotsize(int slot) const {return rsize_[slot-1];}

  DTuROSROSData getuROS(int slot) const {return rdata_[slot-1];}

private:

  long header1_, header2_, trailer_;

  int fed_, nslots_, evtLgth_, rsize_[DOCESLOTS];

  DTuROSROSData rdata_[DOCESLOTS];

};


typedef std::vector<DTuROSFEDData> DTuROSFEDDataCollection;
#endif
