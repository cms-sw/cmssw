#ifndef RAWECAL_ECALDCCHEADERBLOCK_H
#define RAWECAL_ECALDCCHEADERBLOCK_H

#include <boost/cstdint.hpp>

/** \class EcalDCCHeaderBlock
 *  Container for ECAL specific DCC Header information
 *
 *
 *  $Id: EcalDCCHeaderBlock.h,v 1.16 2008/02/07 10:21:10 franzoni Exp $
 */

#include <vector>

#define MAX_TCC_SIZE 4
#define MAX_TT_SIZE 70



class EcalDCCHeaderBlock
{

 public:

  typedef int key_type; ///< For the sorted collection 

  enum EcalDCCRuntype{

    // run types pre-global runs
    COSMIC                   =0,
    BEAMH4                   =1, 
    BEAMH2                   =2,  
    MTCC                     =3,
    LASER_STD                =4,
    LASER_POWER_SCAN         =5,
    LASER_DELAY_SCAN         =6,
    TESTPULSE_SCAN_MEM       =7,
    TESTPULSE_MGPA           =8,
    PEDESTAL_STD             =9,
    PEDESTAL_OFFSET_SCAN     =10,
    PEDESTAL_25NS_SCAN       =11,
    LED_STD                  =12,

    // physics triggers in global runs
    PHYSICS_GLOBAL           =13,
    COSMICS_GLOBAL           =14,
    HALO_GLOBAL              =15,

    // gap events in global runs
    LASER_GAP                =16,
    TESTPULSE_GAP            =17,
    PEDESTAL_GAP             =18,
    LED_GAP                  =19,

    // physics triggers in local runs
    PHYSICS_LOCAL            =20,
    COSMICS_LOCAL            =21,
    HALO_LOCAL               =22,
    CALIB_LOCAL              =23

  };
  
  enum EcalBasicTriggerType{
    PHYSICS_TRIGGER            =1,
    CALIBRATION_TRIGGER        =2,
    TEST_TRIGGER               =3,
    TECHNICAL_EXTERNAL_TRIGGER =4
  };
  
  enum TTC_DTT_TYPE{
    TTC_LASER     = 4,
    TTC_LED       = 5,
    TTC_TESTPULSE = 6,
    TTC_PEDESTAL  = 7
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

  // partitional and geometrical:
  // CMS: 1-9 EE-, 10-45 EB, 46-54 EE+
  // TB and COSM: 1  (10 in case of EcalRawToDigiDev)
  const int& id() const { return dccId_; }

  // as found in raw data, namely
  // CMS: 601-654         TB and COSM: 0-35
  const int fedId() const { return fedId_; }

  void setId(const int& dccId) { dccId_=dccId; };
  void setFedId(const int& fedId) { fedId_=fedId; };

  void setErrors(const int& dccErrors) { dccErrors_=dccErrors; };
  void setDccInTTCCommand(const int& dccInTTCCommand) { dccInTTCCommand_=dccInTTCCommand; };
  void setRunNumber(const int& run){runNumber_ = run;}
  void setLV1(const int& LV1){LV1event_ = LV1;}
  void setBX(const int& BX){BX_ = BX;}
  void setOrbit(const int& orbit){orbitNumber_ = orbit;}
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
  void setTriggerTowerFlags(const std::vector<short>& triggerTowerFlag) { triggerTowerFlag_ = triggerTowerFlag; };
  void setFEStatus(const std::vector<short>& feStatus) { feStatus_ = feStatus; };

  //TODO add all the get methods
  
  int getDCCErrors() const{ return dccErrors_;}
  int getDccInTCCCommand() const{ return dccInTTCCommand_; };
  int getRunNumber() const {return runNumber_ ;}
  int getLV1() const {return LV1event_ ;}
  int getBX() const {return BX_ ;}
  int getOrbit() const {return orbitNumber_;}
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
  std::vector<short> getTriggerTowerFlag() const { return triggerTowerFlag_ ;}
  std::vector<short> getFEStatus() const { return feStatus_ ;}
 private:

  int dccId_;  //to be used as the Key
  int fedId_;
  int dccErrors_;
  int dccInTTCCommand_;
  int orbitNumber_;
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
  std::vector<short> triggerTowerFlag_;
  std::vector<short> feStatus_;
 
};

#endif
