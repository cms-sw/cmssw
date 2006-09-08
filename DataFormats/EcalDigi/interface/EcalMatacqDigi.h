// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-
//$Id: EcalMatacqDigi.h,v 1.1 2006/07/21 12:35:04 meridian Exp $

#ifndef ECALMATACQDIGI_H
#define ECALMATACQDIGI_H 1

#include <ostream>
#include <vector>
#include <algorithm>
#include "Rtypes.h"

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
  /** Default constructor.
   */
  EcalMatacqDigi(): chId_(-1), ts_(0.), tTrigS_(999.), version_(-1){}; 
  
  /** This constructor is here to satisfy reflex (see persistancy and dictionnary).
   * It will be remove as soon as it is no more reqired by reflex.
   */
  EcalMatacqDigi(const std::vector<Short_t>& samples, int chId, double ts,
		 short version=-1, double tTrig=999.)
    : chId_(chId), data_(samples), ts_(ts), tTrigS_(tTrig),
      version_(version){};
  
  
  /** Gets amplitude in ADC count of time sample i. i between 0 and size()-1.
   * Note: Amplitude is pedestal subtracted at acquisition time.
   */
  const float adcCount(int i) const { return data_[i]; }

  /** Gets amplitude in Volt of time sample i. i between 0 and size()-1.
   * Note: Amplitude is pedestal subtracted at acquisition time.
   */
  const float amplitudeV(int i) const { return data_[i]*lsb_;}

  /** Gets Matacq electronics channel id
   */
  int chId() const{ return chId_;}

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
  void swap(std::vector<short> samples){ std::swap(data_, samples);}

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

};


std::ostream& operator<<(std::ostream& s, const EcalMatacqDigi& digi);

#endif
