#ifndef RAWECAL_ECALDCCHEADERBLOCK_H
#define RAWECAL_ECALDCCHEADERBLOCK_H

#include <boost/cstdint.hpp>

/** \class EcalDCCHeaderBlock
 *  Container for ECAL specific DCC Header information
 *
 *
 *  $Id: EcalDCCHeaderBlock.h,v 1.6 2006/05/05 09:04:56 meridian Exp $
 */

#include <vector>

#define MAX_TCC_SIZE 4
#define MAX_TT_SIZE 70



class EcalDCCHeaderBlock
{

 public:

  typedef int key_type; ///< For the sorted collection 

  enum EcalDCCRuntype{
    COSMIC = 0,
    BEAMH4 =1, 
    BEAMH2 = 2,  
    MTCC =3,
    LASER_STD = 4,
    LASER_POWER_SCAN = 5,
    LASER_DELAY_SCAN = 6,
    TESTPULSE_SCAN_MEM = 7,
    TESTPULSE_MGPA = 8,
    PEDESTAL_STD = 9,
    PEDESTAL_OFFSET_SCAN = 10,
    PEDESTAL_25NS_SCAN = 11,
    LED_STD= 12
  };
  
  enum EcalBasicTriggerType{
    PHYSICS_TRIGGER = 1,
    CALIBRATION_TRIGGER = 2,
    TEST_TRIGGER = 3,
    TECHNICAL_EXTERNAL_TRIGGER = 4
  };
  
  struct EcalDCCEventSettings{
    short LaserPower;
    short LaserFilter;
    short wavelength;
    short delay;
    short MEMVinj;
    short mgpa_content;
    short ped_offset;
  };

  EcalDCCHeaderBlock();
  EcalDCCHeaderBlock(const int& dccId);
  
  const int& id() const { return dccId_; }
  void setId(const int& dccId) { dccId_=dccId; };
  void setErrors(const int& dccErrors) { dccErrors_=dccErrors; };
  void setRunNumber(const int& run){runNumber_ = run;}
  void setLV1(const int& LV1){LV1event_ = LV1;}
  void setBX(const int& BX){BX_ = BX;}
  void setEventSettings(const  EcalDCCEventSettings& EventSettings) { EventSettings_=EventSettings; };
  void setRunType(const short& runType) { runType_=runType; };
  void setBasicTriggerType(const short& triggerType) { basic_trigger_type_=triggerType; };
  //void setSequence(const short& sequence) { sequence_=sequence; } ;
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
  
  int getDCCErrors() const{ return dccErrors_;}
  int getRunNumber() const {return runNumber_ ;}
  int getLV1() const {return LV1event_ ;}
  int getBX() const {return BX_ ;}
  EcalDCCEventSettings getEventSettings() const { return EventSettings_;}
  short getRunType() const {return runType_ ;}
  short getBasicTriggerType() const {return basic_trigger_type_ ;}

  short getRtHalf() const { return rtHalf_; } 
  short getMgpaGain() const { return mgpaGain_;}
  short getMemGain() const  { return memGain_;}
  short getSelectiveReadout() const { return selectiveReadout_;}
  bool getZeroSuppression() const { return zeroSuppression_;}
  bool getTestZeroSuppression() const {return testZeroSuppression_ ;}
  short getSrpStatus() const  { return srpStatus_;}
  std::vector<short> getTccStatus() const { return tccStatus_ ;}
  std::vector<short> getTriggerTowerStatus() const { return triggerTowerStatus_ ;}
 private:

  int dccId_;  //to be used as the Key
  int dccErrors_;
  int orbitNumber_; // do we need it here?
  short runType_;

  short basic_trigger_type_;

  int LV1event_;
  int runNumber_;
  int BX_;
  EcalDCCEventSettings  EventSettings_;
  
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
