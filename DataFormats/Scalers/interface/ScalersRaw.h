/*
 *  File: DataFormats/Scalers/interface/ScalersRaw.h   (W.Badgett)
 *
 *  Description of the raw data from the Scalers FED
 *
 */

#ifndef SCALERSRAW_H
#define SCALERSRAW_H

#include <ostream>
#include <vector>

/*! \file ScalersRaw.h
 * \Header file for Raw Data Level 1 Global Trigger Scalers and Lumi Scalers
 * 
 * \author: William Badgett
 *
 */


/// \class ScalersRaw.h
/// \brief Raw Data Level 1 Global Trigger Scalers and Lumi Scalers

class ScalersRaw
{
 public:
  enum
  {
    N_L1_TRIGGERS_v1 = 128,
    N_LUMI_OCC_v1 = 2,
    SCALERS_FED_ID = 735
  };
};

struct TriggerScalersRaw_v1
{
  struct timespec collectionTimeSummary;
  unsigned long long CAL_L1A_;
  unsigned long long DEADT_;
  unsigned long long DEADT_A;
  unsigned long long DEADT_CALIBR_A;
  unsigned long long DEADT_PRIV_A;
  unsigned long long DEADT_PSTATUS_A;
  unsigned long long DEADT_THROTTLE_A;
  unsigned long long EVNR;
  unsigned long long FINOR_;
  unsigned long long LOST_BC_;
  unsigned long long LOST_TRIG_;
  unsigned long long LOST_TRIG_A;
  unsigned long long NR_RESETS_;
  unsigned long long ORBITNR;
  unsigned long long PHYS_L1A;
  unsigned long long RNDM_L1A_;
  unsigned long long TECHTRIG_;
  unsigned long long TRIGNR_;

  struct timespec collectionTimeDetails;
  unsigned int RATE_ALGO[ScalersRaw::N_L1_TRIGGERS_v1];
};

struct LumiScalersRaw_v1
{
  double DeadtimeNormalization;
  double Normalization;
  double InstantLumi;
  double InstantLumiErr;
  double InstantLumiQlty;
  double InstantETLumi;
  double InstantETLumiErr;
  double InstantETLumiQlty;
  double InstantOccLumi[ScalersRaw::N_LUMI_OCC_v1];
  double InstantOccLumiErr[ScalersRaw::N_LUMI_OCC_v1];
  double InstantOccLumiQlty[ScalersRaw::N_LUMI_OCC_v1];
  double lumiNoise[ScalersRaw::N_LUMI_OCC_v1];
  unsigned int sectionNumber;
  unsigned int startOrbit;
  unsigned int numOrbits;
};

struct ScalersEventRecordRaw_v1
{
  int version;
  struct TriggerScalersRaw_v1 trig;
  struct LumiScalersRaw_v1    lumi;
};

#endif
