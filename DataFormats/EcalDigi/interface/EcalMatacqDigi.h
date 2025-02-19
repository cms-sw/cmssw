// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-
//$Id: EcalMatacqDigi.h,v 1.6 2011/08/30 18:42:57 wmtan Exp $

#ifndef ECALMATACQDIGI_H
#define ECALMATACQDIGI_H 1

#include <ostream>
#include <vector>
#include <algorithm>
#include "Rtypes.h"

#define ECAL_MATACQ_DIGI_VERS 2

/** \class EcalMatacqDigi
*/

class EcalMatacqDigi {
public:
  /** LSB of ADC count in Volt
   */
  static const double lsb_;

  /** Maximum number of time samples
   */
  static const int MAXSAMPLES = 2560;

public:
  typedef int key_type; //matacq channel id used as key for edm::SortedCollection

public:
  /** Default constructor.
   */
  EcalMatacqDigi(): chId_(-1), ts_(0.), tTrigS_(999.), version_(-1){
    init();
  }
  
  /** Constructor
   * @param samples adc time samples
   * @param chId Matacq channel ID
   * @param ts sampling time in seconds
   * @param version Matacq raw data private version
   * @param tTrigg time position of the trigger in seconds
   */
  EcalMatacqDigi(const std::vector<Short_t>& samples, const int& chId, const double& ts,
		 const short& version=-1, const double& tTrig=999.)
    : chId_(chId), data_(samples), ts_(ts), tTrigS_(tTrig),
      version_(version){
    init();
  };
  
  
  /** Gets amplitude in ADC count of time sample i. i between 0 and size()-1.
   * Note: Amplitude is pedestal subtracted at acquisition time.
   */
  const float adcCount(const int& i) const { return data_[i]; }

  /** Gets amplitude in Volt of time sample i. i between 0 and size()-1.
   * Note: Amplitude is pedestal subtracted at acquisition time.
   */
  const float amplitudeV(const int& i) const { return data_[i]*lsb_;}

  /** Gets Matacq electronics channel id
   */
  int chId() const{ return chId_;}

  /** For edm::SortedCollection.
   * @return as key the matacq channel id
   */
  int id() const { return chId_;}

//   /** Sets Matacq electronics channel id
//    */
//   void chId(int newChId){ chId_ = newChId;}
  
  /** Number of time samples
   */
  int size() const { return data_.size();}
  
  /** Swaps samples with the passed samples. For package internal use.
   * @param samples new time samples in unit used in raw data
   * (a priori ADC count).
   */
  void swap(std::vector<short>& samples){ std::swap(data_, samples);}

  void swap(EcalMatacqDigi& a);
  
//   /** Gets time of sample i. i between 0 and size()-1.
//    */
//   float t(int i) const { return ts()*i;}
  
  /** Gets sampling time in seconds
   */
  float ts() const { return ts_;}

//   /** Sets sampling time period
//    * @param ts sampling time period in seconds
//    */
//   void ts(double newTs){
//     ts_ = newTs;
//   }

  /** Gets time of trigger in seconds.
   * @return (t_trig-t_0), with t_trig the trigger time and t_0 the first.
   * Returns 999 if not available.
   * sample time.
   */
  float tTrig() const { return tTrigS_;}

//   /** Sets trigger time position
//    * @param tTrigS (t_trig-t_0) in seconds, with t_trig the time of MATACQ
//    * trigger and t_0 the time of the first sample of each MATACQ channel.
//    */
//   void tTrig(double tTrigS){
//     tTrigS_ = tTrigS;
//   }

  /** version of raw data format, the digis originate from.
   * @return raw data format version, -1 if not available.
   */
  short version() const {return version_;}

//   /** Sets the raw data format, the digi is issued from.
//    * @param version internal matacq raw data format version
//    */
//   void version(short version){
//     version_ = version;
//   }

#if (ECAL_MATACQ_DIGI_VERS>=2)
  /** Gets the bunch crossing id field contents.
   * @return BX id
   */
  int bxId() const { return bxId_; }

  /** Sets the bunch crossing id field contents.
   * @param value new value
   */
  void bxId(int value) { bxId_ = value; }

  /** Gets level one accept counter of the event
   * @return l1a
   */
  int l1a() const { return l1a_; }
  
  /** Sets level one accept counter of the event
   * @param value new value
   */
  void l1a(int value) { l1a_ = value; }
  
  /** Gets type of test trigger
   * @return triggerType
   */
  int triggerType() const { return triggerType_; }

  /** Sets type of test trigger
   * @param value new value
   */
  void triggerType(int value) { triggerType_ = value; }

  /** Gets the matacq data timestamp with fine granularity (89.1us) 
   * @return acquisition date of the data expressed in number of "elapsed"
   * second and microseconds since the EPOCH as defined in POSIX.1.
   * See time() standard c function and gettimeofday UNIX function.
   */
  timeval timeStamp() const { timeval value; value.tv_sec = tv_sec_; value.tv_usec = tv_usec_; return value; }

  /** Sets the matcq event timestmap
   * @param value new value
   */
  void timeStamp(timeval value) { tv_sec_ = value.tv_sec; tv_usec_ = value.tv_usec; }
  
  /** Gets the LHC orbit ID of the event
   * Available only for Matacq data format version >=3 and for P5 data.
   * @return the LHC orbit ID
   */
  UInt_t orbitId() const { return orbitId_; }

  /** Sets the LHC orbit ID of the event.
   * @param value new value
   */
  void orbitId(UInt_t value) { orbitId_ = value; }
  
  /** Gets the Trig Rec value (see Matacq documentation)
   * Available only for Matacq data format version >=3.
   * @return the Trig Rec value
   */
  int trigRec() const { return trigRec_; }
  
  /** Sets the Trig Rec value (see Matacq documentation)
   * @param value new value
   */
  void trigRec(int value) { trigRec_ = value; }

  /** Gets the Posttrig value (see Matacq documentation).
   * Available only for Matacq data format version >=3.
   */
  int postTrig() const { return postTrig_; }

  /** Sets the Posttrig value (see Matacq documentation).
   * @param value new value
   */
  void postTrig(int value) { postTrig_ = value; }

  /** Gets the vernier values (see Matacq documentation)
   * @return verniers
   */
  std::vector<int> vernier() const { return vernier_; }
  
  /** Sets verniers
   * @param value new value
   */
  void vernier(const std::vector<int>& value) { vernier_ = value; }

  
  /** Gets "Delay A" setting of laser delay box in ns.delayA
   * @return delayA
   */
  int delayA() const { return delayA_; }

  /** Sets "Delay A" setting of laser delay box in ns.delayA
   * @param value new value
   */
  void delayA(int value) { delayA_ = value; }

  /** Gets the WTE-to-Laser delay of EMTC in LHC clock unit.
   * @return emtcDelay
   */
  int emtcDelay() const { return emtcDelay_; }
  
  /** Sets the WTE-to-Laser delay of EMTC in LHC clock unit.
   * @param value new value
   */
  void emtcDelay(int value) { emtcDelay_ = value; }
  
  /** Gets the EMTC laser phase in 1/8th LHC clock unit.
   * @return emtcPhase
   */
  int emtcPhase() const { return emtcPhase_; }
  
  /** Sets the EMTC laser phase in 1/8th LHC clock unit.
   * @param value new value
   */
  void emtcPhase(int value) { emtcPhase_ = value; }

  /** Gets the laser logarithmic attenuator setting in -10dB unit.
   * Between 0 and 5*(-10dB), -1 if unknown.
   * @return attenuation_dB
   */
  int attenuation_dB() const { return attenuation_dB_; }

  /** Sets the laser Logarithmic attenuator setting in -10dB unit.
   * Between 0 and 5*(-10dB), -1 if unknown.
   * @param value new value
   */
  void attenuation_dB(int value) { attenuation_dB_ = value; }
  
  /** Gets the laser power setting in percents
   * (set with the linear attenuator),
   * @return laserPower
   */
  int laserPower() const { return laserPower_; }
  
  /** Sets  the laser power setting in percents
   * (set with the linear attenuator),
   * @param value new value
   */
  void laserPower(int value) { laserPower_ = value; }

  void init(){
#if (ECAL_MATACQ_DIGI_VERS>=2)
    bxId_ = -1;
    l1a_ = -1;
    triggerType_ = -1;
    orbitId_ = -1;
    trigRec_ = -1;
    postTrig_ = -1;
    vernier_ = std::vector<Int_t>(4,-1);
    delayA_ = -1;
    emtcDelay_ = -1;
    emtcPhase_ = -1;
    attenuation_dB_ = -1;
    laserPower_ = -1;
#endif
  }

#endif

private:
  /** Electronic channel id
   */
  int chId_;
  
  /** ADC count of time samples
   */
  std::vector<Short_t> data_;

  /** Frequency mode. 1->1GHz sampling, 2->2GHz sampling
   */
  int freq;
  
  /**Sampling period in seconds. In priniciple 1ns or 0.5ns
   */
  double ts_;

  /** Trigger time in seconds
   */
  double tTrigS_;

  /** version of raw data format, the digis originate from.
   */
  short version_;

#if (ECAL_MATACQ_DIGI_VERS>=2)
  /** Type of test trigger
   * @return triggerType
   */
  char triggerType_;

  /**  Logarithmic attenuator setting in -10dB unit. Between 0 and
   *  5*(-10dB), -1 if unknown.
   */
  char attenuation_dB_;

  /** Bunch crossing Id 
   */
  Short_t bxId_;

  /** Event id. Actually LV1 ID.
   */
  Int_t l1a_;
  
  /* LHC orbit ID
   */
  Int_t orbitId_;
  
  /** Trig Rec value (see Matacq documentation)
   */
  Short_t trigRec_;
  
  /** Posttrig value (see Matacq documentation)
   */
  Short_t postTrig_;

  /** Vernier values (see Matacq documentation)
   */
  std::vector<Int_t> vernier_;

  /** "Delay A" setting of laser delay box in ns.
   */
  Int_t delayA_;

  /**  WTE-to-Laser delay of EMTC in LHC clock unit.
   */
  Int_t emtcDelay_;

  /** EMTC laser phase in 1/8th LHC clock unit.
   */
  Int_t emtcPhase_;

  /** Laser power in percents (set with the linear attenuator).
   */
  Int_t laserPower_;

  /** Matacq acquisition time stamp
   */
  /** We don't use timeval directly, because its typedef is platform dependent.
   */
  Long64_t tv_sec_;
  Long64_t tv_usec_;

#endif
  
};


std::ostream& operator<<(std::ostream& s, const EcalMatacqDigi& digi);

#endif
