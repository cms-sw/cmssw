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

#pragma pack(push)
#pragma pack(4)

/// \class ScalersRaw.h
/// \brief Raw Data Level 1 Global Trigger Scalers and Lumi Scalers

class ScalersRaw
{
 public:
  enum
  {
    N_L1_TRIGGERS_v1      = 128,
    N_L1_TEST_TRIGGERS_v1 = 64,
    N_LUMI_OCC_v1         = 2,
    N_BX_v2               = 4,
    N_BX_v6               = 8,
    N_SPARE_v5            = 3,
    I_SPARE_PILEUP_v7     = 0,
    I_SPARE_PILEUPRMS_v7  = 1,
    I_SPARE_BUNCHLUMI_v8  = 2,
    SCALERS_FED_ID        = 735
  };
};

struct TriggerScalersRaw_v1
{
  unsigned int       collectionTimeSpecial_sec;
  unsigned int       collectionTimeSpecial_nsec;
  unsigned int       ORBIT_NUMBER;          /* ORBITNR          */
  unsigned int       LUMINOSITY_SEGMENT;
  unsigned short     BC_ERRORS;            

  unsigned int       collectionTimeSummary_sec;
  unsigned int       collectionTimeSummary_nsec;
  unsigned int       TRIGGER_NR;            /* TRIGNR_          */
  unsigned int       EVENT_NR;              /* EVNR             */
  unsigned int       FINOR_DISTRIBUTED;     /* PHYS_L1A      ?? */
  unsigned int       CAL_TRIGGER;           /* CAL_L1A_         */
  unsigned int       RANDOM_TRIGGER;        /* RNDM_L1A_        */
  unsigned int       TEST_TRIGGER;          /* TECHTRIG_        */
  unsigned int       FINOR_GENERATED;       /* FINOR_        ?? */
  unsigned int       FINOR_IN_INVALID_BC;   /* LOST_BC_      ?? */
  unsigned long long DEADTIME;              /* DEADT_           */
  unsigned long long LOST_FINOR;            /* LOST_TRIG_    ?? */
  unsigned long long DEADTIMEA;             /* DEADT_A          */
  unsigned long long LOST_FINORA;           /* LOST_TRIG_A   ?? */
  unsigned long long PRIV_DEADTIMEA;        /* DEADT_PRIV_A     */
  unsigned long long PTCSTATUS_DEADTIMEA;   /* DEADT_PSTATUS_A  */
  unsigned long long THROTTLE_DEADTIMEA;    /* DEADT_THROTTLE_A */
  unsigned long long CALIBRATION_DEADTIMEA; /* DEADT_CALIBR_A   */
  unsigned long long TIMESLOT_DEADTIMEA;    /*                  */
  unsigned int       NR_OF_RESETS;          /* NR_RESETS_       */

  unsigned int       collectionTimeDetails_sec;
  unsigned int       collectionTimeDetails_nsec;
  unsigned int       ALGO_RATE[ScalersRaw::N_L1_TRIGGERS_v1];
  unsigned int       TEST_RATE[ScalersRaw::N_L1_TEST_TRIGGERS_v1];
};

struct TriggerScalersRaw_v3
{
  unsigned int collectionTime_sec;
  unsigned int collectionTime_nsec;
  unsigned int lumiSegmentNr;
  unsigned int lumiSegmentOrbits;
  unsigned int orbitNr;
  unsigned int gtResets;
  unsigned int bunchCrossingErrors;
  unsigned long long gtTriggers;
  unsigned long long gtEvents;
  float gtTriggersRate;
  float gtEventsRate;
  int prescaleIndexAlgo;
  int prescaleIndexTech;

  unsigned int collectionTimeLumiSeg_sec;
  unsigned int collectionTimeLumiSeg_nsec;
  unsigned int lumiSegmentNrLumiSeg;
  unsigned long long triggersPhysicsGeneratedFDL;
  unsigned long long triggersPhysicsLost;
  unsigned long long triggersPhysicsLostBeamActive;
  unsigned long long triggersPhysicsLostBeamInactive;
  unsigned long long l1AsPhysics;
  unsigned long long l1AsRandom;
  unsigned long long l1AsTest;
  unsigned long long l1AsCalibration;
  unsigned long long deadtime;
  unsigned long long deadtimeBeamActive;
  unsigned long long deadtimeBeamActiveTriggerRules;
  unsigned long long deadtimeBeamActiveCalibration;
  unsigned long long deadtimeBeamActivePrivateOrbit;
  unsigned long long deadtimeBeamActivePartitionController;
  unsigned long long deadtimeBeamActiveTimeSlot;

  unsigned int gtAlgoCounts[ScalersRaw::N_L1_TRIGGERS_v1];
  unsigned int gtTechCounts[ScalersRaw::N_L1_TEST_TRIGGERS_v1];
};

struct LumiScalersRaw_v1
{
  unsigned int collectionTime_sec;
  unsigned int collectionTime_nsec;
  float DeadtimeNormalization;
  float Normalization;

  float LumiFill;
  float LumiRun;
  float LiveLumiFill;
  float LiveLumiRun;
  float InstantLumi;
  float InstantLumiErr;
  unsigned char InstantLumiQlty;

  float LumiETFill;
  float LumiETRun;
  float LiveLumiETFill;
  float LiveLumiETRun;
  float InstantETLumi;
  float InstantETLumiErr;
  unsigned char InstantETLumiQlty;

  float LumiOccFill[ScalersRaw::N_LUMI_OCC_v1];
  float LumiOccRun[ScalersRaw::N_LUMI_OCC_v1];
  float LiveLumiOccFill[ScalersRaw::N_LUMI_OCC_v1];
  float LiveLumiOccRun[ScalersRaw::N_LUMI_OCC_v1];
  float InstantOccLumi[ScalersRaw::N_LUMI_OCC_v1];
  float InstantOccLumiErr[ScalersRaw::N_LUMI_OCC_v1];
  unsigned char InstantOccLumiQlty[ScalersRaw::N_LUMI_OCC_v1];
  float lumiNoise[ScalersRaw::N_LUMI_OCC_v1];

  unsigned int sectionNumber;
  unsigned int startOrbit;
  unsigned int numOrbits;
};

struct BeamSpotOnlineRaw_v4
{
  unsigned int collectionTime_sec;
  unsigned int collectionTime_nsec;
  float x;
  float y;
  float z;
  float dxdz;
  float dydz;
  float err_x;
  float err_y;
  float err_z;
  float err_dxdz;
  float err_dydz;
  float width_x;
  float width_y;
  float sigma_z;
  float err_width_x;
  float err_width_y;
  float err_sigma_z;
};

struct DcsStatusRaw_v4
{
  unsigned int collectionTime_sec;
  unsigned int collectionTime_nsec;
  unsigned int ready;
  float magnetCurrent;
  float magnetTemperature;
};

struct ScalersEventRecordRaw_v1
{
  unsigned long long header;
  int version;
  struct TriggerScalersRaw_v1 trig;
  struct LumiScalersRaw_v1    lumi;
  unsigned int filler;
  unsigned long long trailer;
};

struct ScalersEventRecordRaw_v2
{
  unsigned long long header;
  int version;
  struct TriggerScalersRaw_v1 trig;
  struct LumiScalersRaw_v1    lumi;
  unsigned int filler;
  unsigned long long bx[ScalersRaw::N_BX_v2];
  unsigned long long trailer;
};

struct ScalersEventRecordRaw_v3
{
  unsigned long long header;
  int version;
  struct TriggerScalersRaw_v3 trig;
  struct LumiScalersRaw_v1    lumi;
  unsigned int filler;
  unsigned long long bx[ScalersRaw::N_BX_v2];
  unsigned long long trailer;
};

struct ScalersEventRecordRaw_v4
{
  unsigned long long header;
  int version;
  struct TriggerScalersRaw_v3 trig;
  struct LumiScalersRaw_v1    lumi;
  struct BeamSpotOnlineRaw_v4 beamSpotOnline;
  struct DcsStatusRaw_v4      dcsStatus;
  unsigned long long bx[ScalersRaw::N_BX_v2];
  unsigned long long trailer;
};


struct ScalersEventRecordRaw_v5
{
  unsigned long long          header;
  int                         version;
  struct TriggerScalersRaw_v3 trig;
  struct LumiScalersRaw_v1    lumi;
  struct BeamSpotOnlineRaw_v4 beamSpotOnline;
  struct DcsStatusRaw_v4      dcsStatus;
  unsigned int                lastOrbitCounter0;
  unsigned int                lastTestEnable;
  unsigned int                lastResync;
  unsigned int                lastStart;
  unsigned int                lastEventCounter0;
  unsigned int                lastHardReset;
  unsigned long long          spare[ScalersRaw::N_SPARE_v5];
  unsigned long long          bx[ScalersRaw::N_BX_v2];
  unsigned long long          trailer;
};

struct ScalersEventRecordRaw_v6
{
  unsigned long long          header;
  int                         version;
  struct TriggerScalersRaw_v3 trig;
  struct LumiScalersRaw_v1    lumi;
  struct BeamSpotOnlineRaw_v4 beamSpotOnline;
  struct DcsStatusRaw_v4      dcsStatus;
  unsigned int                lastOrbitCounter0;
  unsigned int                lastTestEnable;
  unsigned int                lastResync;
  unsigned int                lastStart;
  unsigned int                lastEventCounter0;
  unsigned int                lastHardReset;
  unsigned long long          spare[ScalersRaw::N_SPARE_v5];
  unsigned long long          bx[ScalersRaw::N_BX_v6];
  unsigned long long          trailer;
};

#pragma pack(pop)

#endif
