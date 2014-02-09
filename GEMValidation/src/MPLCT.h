#ifndef SimMuL1_MPLCT_h
#define SimMuL1_MPLCT_h

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "GEMCode/GEMValidation/src/LCT.h" 

class MPLCT
{
typedef std::vector<const MPLCT*> MPLCTCollection;

 public:
  /// constructor
  MPLCT();
  /// copy constructor
  MPLCT(const MPLCT&);
  /// destructor
  ~MPLCT();

  /// get the underlying trigger digi
  const CSCCorrelatedLCTDigi* getTriggerDigi() const {return triggerDigi_;}
  /// get the lct
  const LCT* getLCT() const {return lct_;}
  /// get the bunch crossing
  const int getBX() const {return triggerDigi_->getBX();}
  /// get the detector Id
  const unsigned getDetId() const {return detId_;}
  /// is the LCT a ghost?
  const bool isGhost() const {return isGhost_;}
  /// does the LCT match?
  const bool deltaOk() const {return deltaOk_;}
  /// is the LCT in the readout? 
  const bool inReadout() const {return inReadout_;}

  /// set the underlying trigger digi
  void setTriggerDigi(const CSCCorrelatedLCTDigi* d) {triggerDigi_ = d;}
  /// set the clct
  void setLCT(const LCT* lct) {lct_ = lct;}
  /// set the bunch crossing
  void setBX(const int bx) {bx_ = bx;}
  /// set the detector Id
  void setDetId(const unsigned detId) {detId_ = detId;}
  /// is the LCT a ghost?
  void SetGhost(const bool ghost) {isGhost_ = ghost;}
  /// does the LCT match?
  void SetDeltaOk(const bool ok) {deltaOk_ = ok;}
  /// is the LCT in the readout? 
  void SetReadout(const bool inReadout) {inReadout_ = inReadout;}

 private:

  const CSCCorrelatedLCTDigi* triggerDigi_;
  const LCT* lct_;
  int bx_;
  unsigned detId_;
  bool isGhost_;
  bool deltaOk_;
  bool inReadout_;
};

#endif
