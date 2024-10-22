#ifndef RAWECAL_ESDCCHEADERBLOCK_H
#define RAWECAL_ESDCCHEADERBLOCK_H
#include <vector>

class ESDCCHeaderBlock {
public:
  typedef int key_type;

  enum ESDCCRunType {
    PEDESTAL_RUN = 1,
    TESTPULSE_RUN = 2,
    COSMIC_RUN = 3,
    BEAM_RUN = 4,
    PHYSICS_RUN = 5,
    TEST_RUN = 6,
    GLOBAL_COSMIC_RUN = 7
  };

  enum ESSeqType { STD_SEQ = 0, DELTASCAN_SEQ = 1, DELAYSCAN_SEQ = 2, PULSESHAPE_SEQ = 3 };

  enum ESTriggerType { PHYSICS_TRIGGER = 1, CALIBRATION_TRIGGER = 2, TEST_TRIGGER = 3, TECHNICAL_EXTERNAL_TRIGGER = 4 };

  ESDCCHeaderBlock();
  ESDCCHeaderBlock(const int& dccId);

  const int& id() const { return dccId_; }
  void setId(const int& dccId) { dccId_ = dccId; };

  const int fedId() const { return fedId_; }
  void setFedId(const int& fedId) { fedId_ = fedId; };

  void setLV1(const int& LV1) { LV1_ = LV1; };
  void setBX(const int& BX) { BX_ = BX; }
  void setGain(const int& gain) { gain_ = gain; }
  void setPrecision(const int& precision) { precision_ = precision; }
  void setDAC(const int& dac) { dac_ = dac; }
  void setEventLength(const int& evtLen) { evtLen_ = evtLen; };
  void setDCCErrors(const int& dccErrs) { dccErrs_ = dccErrs; };
  void setRunNumber(const int& runNum) { runNum_ = runNum; };
  void setRunType(const int& runType) { runType_ = runType; };
  void setSeqType(const int& seqType) { seqType_ = seqType; };
  void setTriggerType(const int& trgType) { trgType_ = trgType; };
  void setCompressionFlag(const int& compFlag) { compFlag_ = compFlag; };
  void setOrbitNumber(const int& orbit) { orbit_ = orbit; };
  void setMajorVersion(const int& vmajor) { vmajor_ = vmajor; };
  void setMinorVersion(const int& vminor) { vminor_ = vminor; };
  void setOptoRX0(const int& optoRX0) { optoRX0_ = optoRX0; };
  void setOptoRX1(const int& optoRX1) { optoRX1_ = optoRX1; };
  void setOptoRX2(const int& optoRX2) { optoRX2_ = optoRX2; };
  void setOptoBC0(const int& optoBC0) { optoBC0_ = optoBC0; };
  void setOptoBC1(const int& optoBC1) { optoBC1_ = optoBC1; };
  void setOptoBC2(const int& optoBC2) { optoBC2_ = optoBC2; };
  void setFEChannelStatus(const std::vector<int>& FEch) { FEch_ = FEch; };
  // crepe thing
  void setPacketLength(const int& packetLen) { packetLen_ = packetLen; };
  void setBC(const int& bc) { bc_ = bc; };
  void setEV(const int& ev) { ev_ = ev; };
  // top level supervisor
  void setBMMeasurements(const int& BMMeasurements) { BMMeasurements_ = BMMeasurements; };
  void setBeginOfSpillSec(const int& beginOfSpillSec) { beginOfSpillSec_ = beginOfSpillSec; };
  void setBeginOfSpillMilliSec(const int& beginOfSpillMilliSec) { beginOfSpillMilliSec_ = beginOfSpillMilliSec; };
  void setEndOfSpillSec(const int& endOfSpillSec) { endOfSpillSec_ = endOfSpillSec; };
  void setEndOfSpillMilliSec(const int& endOfSpillMilliSec) { endOfSpillMilliSec_ = endOfSpillMilliSec; };
  void setBeginOfSpillLV1(const int& beginOfSpillLV1) { beginOfSpillLV1_ = beginOfSpillLV1; };
  void setEndOfSpillLV1(const int& endOfSpillLV1) { endOfSpillLV1_ = endOfSpillLV1; };
  // Cosmic Trigger Supervisor
  void setTimeStampSec(const int& timestamp_sec) { timestamp_sec_ = timestamp_sec; };
  void setTimeStampUSec(const int& timestamp_usec) { timestamp_usec_ = timestamp_usec; };
  void setSpillNumber(const int& spillNum) { spillNum_ = spillNum; };
  void setEventInSpill(const int& evtInSpill) { evtInSpill_ = evtInSpill; };
  void setCAMACError(const int& camacErr) { camacErr_ = camacErr; };
  void setVMEError(const int& vmeErr) { vmeErr_ = vmeErr; };
  void setADCChannelStatus(const std::vector<int>& ADCch_status) { ADCch_status_ = ADCch_status; };
  void setADCChannel(const std::vector<int>& ADCch) { ADCch_ = ADCch; };
  void setTDCChannelStatus(const std::vector<int>& TDCch_status) { TDCch_status_ = TDCch_status; };
  void setTDCChannel(const std::vector<int>& TDCch) { TDCch_ = TDCch; };

  int getLV1() const { return LV1_; }
  int getBX() const { return BX_; }
  int getGain() const { return gain_; }
  int getPrecision() const { return precision_; }
  int getDAC() const { return dac_; }
  int getEventLength() const { return evtLen_; }
  int getDCCErrors() const { return dccErrs_; }
  int getRunNumber() const { return runNum_; }
  int getRunType() const { return runType_; }
  int getSeqType() const { return seqType_; }
  int getTriggerType() const { return trgType_; }
  int getCompressionFlag() const { return compFlag_; }
  int getOrbitNumber() const { return orbit_; }
  int getMajorVersion() const { return vmajor_; }
  int getMinorVersion() const { return vminor_; }
  int getOptoRX0() const { return optoRX0_; }
  int getOptoRX1() const { return optoRX1_; }
  int getOptoRX2() const { return optoRX2_; }
  int getOptoBC0() const { return optoBC0_; }
  int getOptoBC1() const { return optoBC1_; }
  int getOptoBC2() const { return optoBC2_; }
  const std::vector<int>& getFEChannelStatus() const { return FEch_; }
  int getPacketLength() const { return packetLen_; }
  int getBC() const { return bc_; }
  int getEV() const { return ev_; }
  int getBMMeasurements() const { return BMMeasurements_; }
  int getBeginOfSpillSec() const { return beginOfSpillSec_; }
  int getBeginOfSpillMiliSec() const { return beginOfSpillMilliSec_; }
  int getEndOfSpillSec() const { return endOfSpillSec_; }
  int getEndOfSpillMiliSec() const { return endOfSpillMilliSec_; }
  int getBeginOfSpillLV1() const { return beginOfSpillLV1_; }
  int getEndOfSpillLV1() const { return endOfSpillLV1_; }
  int getTimeStampSec() const { return timestamp_sec_; }
  int getTimeStampUSec() const { return timestamp_usec_; }
  int getSpillNumber() const { return spillNum_; }
  int getEventInSpill() const { return evtInSpill_; }
  int getCAMACError() const { return camacErr_; }
  int getVMEError() const { return vmeErr_; }
  const std::vector<int>& getADCChannelStatus() const { return ADCch_status_; }
  const std::vector<int>& getADCChannel() const { return ADCch_; }
  const std::vector<int>& getTDCChannelStatus() const { return TDCch_status_; }
  const std::vector<int>& getTDCChannel() const { return TDCch_; }

private:
  int dccId_;
  int fedId_;
  int LV1_;
  int BX_;
  int gain_;
  int precision_;
  int dac_;
  int evtLen_;
  int dccErrs_;
  int runNum_;
  int runType_;
  int seqType_;
  int trgType_;
  int compFlag_;
  int orbit_;
  int vmajor_;
  int vminor_;
  int optoRX0_;
  int optoRX1_;
  int optoRX2_;
  int optoBC0_;
  int optoBC1_;
  int optoBC2_;
  std::vector<int> FEch_;
  int packetLen_;
  int bc_;
  int ev_;
  int BMMeasurements_;
  int beginOfSpillSec_;
  int beginOfSpillMilliSec_;
  int endOfSpillSec_;
  int endOfSpillMilliSec_;
  int beginOfSpillLV1_;
  int endOfSpillLV1_;
  int timestamp_sec_;
  int timestamp_usec_;
  int spillNum_;
  int evtInSpill_;
  int camacErr_;
  int vmeErr_;
  std::vector<int> ADCch_status_;
  std::vector<int> ADCch_;
  std::vector<int> TDCch_status_;
  std::vector<int> TDCch_;
};

#endif
