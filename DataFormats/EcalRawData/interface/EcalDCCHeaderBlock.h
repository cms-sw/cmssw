#ifndef RAWECAL_ECALDCCHEADERBLOCK_H
#define RAWECAL_ECALDCCHEADERBLOCK_H

#include <boost/cstdint.hpp>

/** \class EcalDCCHeaderBlock
 *  Container for ECAL specific DCC Header information
 *
 *
 *  $Id: $
 */

#include <vector>

#define MAX_TCC_SIZE 4
#define MAX_TT_SIZE 70

class EcalDCCHeaderBlock
{

 public:
  typedef int key_type; ///< For the sorted collection 

  EcalDCCHeaderBlock();
  EcalDCCHeaderBlock(const int& dccId);
  
  const int& id() const { return dccId_; }
  void setId(const int& dccId) { dccId_=dccId; };
  void setErrors(const int& dccErrors) { dccErrors_=dccErrors; };
  void setCycleSettings(const int& cycleSettings) { cycleSettings_=cycleSettings; };
  void setRunType(const short& runType) { runType_=runType; };
  void setSequence(const short& sequence) { sequence_=sequence; } ;
  void setRtHalf(const short& rtHalf) { rtHalf_=rtHalf; } ;
  void setMgpaGain(const short& mgpaGain) { mgpaGain_=mgpaGain; };
  void setMemGain(const short& memGain) { memGain_=memGain; };
  void setSelectiveReadout(const bool& selectiveReadout) { selectiveReadout_=selectiveReadout; };
  void setZeroSuppression(const bool& zeroSuppression) { zeroSuppression_=zeroSuppression; };
  void setTestZeroSuppression(const bool& testZeroSuppression) { testZeroSuppression_ = testZeroSuppression; };
  void setSrpStatus(const short& srpStatus) { srpStatus_=srpStatus; };
  void setTccStatus(const std::vector<short>& tccStatus) { tccStatus_=tccStatus; };
  void setTriggerTowerStatus(const std::vector<short>& triggerTowerStatus) { triggerTowerStatus_ = triggerTowerStatus; };

  //TODO add all the get methods
  
 private:

  int dccId_;  //to be used as the Key
  int dccErrors_;
  long orbitNumber_; // do we need it here?
  int cycleSettings_;
  short runType_;
  short sequence_;
  short rtHalf_;
  short mgpaGain_;
  short memGain_;

  bool selectiveReadout_;
  bool testZeroSuppression_;
  bool zeroSuppression_;

  short srpStatus_;
  std::vector<short> tccStatus_;
  std::vector<short> triggerTowerStatus_;
 
};

#endif
