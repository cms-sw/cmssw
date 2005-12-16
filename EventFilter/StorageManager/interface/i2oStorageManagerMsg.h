#ifndef __i2oStorageManagerMsg_h__
#define __i2oStorageManagerMsg_h__

#include "i2o/i2o.h"
//#include "FWCore/EDProduct/interface/EventID.h"

/*
   Author: Harry Cheung, FNAL

   Description:
     Used for FU I2O frame output module and by the
     Storage Manager I2O I2O input
     See CMS EventFilter wiki page for further notes.

   Modification:
     version 1.1 2005/11/23
       Initial implementation
     version 1.2 2005/12/15
       Added run and event numbers instead of just event.
       Replaced hltID with 5 variables to identify the XDAQ process.

*/

// how do we register these numbers/codes so no-one else uses these?
#define I2O_SM_PREAMBLE 0x001a
#define I2O_SM_DATA     0x001b
#define I2O_SM_OTHER    0x001c
// do we really want to hardwire these sizes?
//
// RunNumber_t and EventNumber_t are unsigned long variables
// Just hardware their sizes as we need to know them
//
// source (HLT) id could be compressed into fewer bytes!
//
//max data I2O frame is (2**16 - 1) * 4 = 65535 * 4 = 262140
//max data array size is then 262140 - 28 - 136 = 261976 bytes
#define MAX_I2O_SM_DATASIZE 261976
// Not sure if Registry always fit in a single I2O frame??!
// registry data array size is 262140 - 28 - 116 = 261996 bytes
#define MAX_I2O_REGISTRY_DATASIZE 261996
// we want to define the maximum event data size?
// max size is 20 x 262088 = about 5MB (used in testI2OReceiver only)
#define MAX_I2O_SM_DATAFRAMES 20
// maximum characters for the source class name and url
#define MAX_I2O_SM_URLCHARS 50

/**
 * Storage Manager Preample Message
 */
typedef struct _I2O_SM_PREAMBLE_MESSAGE_FRAME {
   I2O_PRIVATE_MESSAGE_FRAME PvtMessageFrame;
   unsigned long             dataSize;
   char                      hltURL[MAX_I2O_SM_URLCHARS];
   char                      hltClassName[MAX_I2O_SM_URLCHARS];
   unsigned long             hltLocalId;
   unsigned long             hltInstance;
   unsigned long             hltTid;
   char                      data[MAX_I2O_REGISTRY_DATASIZE];
} I2O_SM_PREAMBLE_MESSAGE_FRAME, *PI2O_SM_PREAMBLE_MESSAGE_FRAME;

/**
 * Storage Manager Data Message
 */
typedef struct _I2O_SM_DATA_MESSAGE_FRAME {
   I2O_PRIVATE_MESSAGE_FRAME PvtMessageFrame;
   unsigned long             dataSize;
   char                      hltURL[MAX_I2O_SM_URLCHARS];
   char                      hltClassName[MAX_I2O_SM_URLCHARS];
   unsigned long             hltLocalId;
   unsigned long             hltInstance;
   unsigned long             hltTid;
//   edm::RunNumber_t          runID;
//   edm::EventNumber_t        eventID;
   unsigned long             runID;
   unsigned long             eventID;
   unsigned long             numFrames;
   unsigned long             frameCount;
   unsigned long             originalSize;
   char                      data[MAX_I2O_SM_DATASIZE];
} I2O_SM_DATA_MESSAGE_FRAME, *PI2O_SM_DATA_MESSAGE_FRAME;

/**
 * Storage Manager OTHER Messages
 */
typedef struct _I2O_SM_OTHER_MESSAGE_FRAME {
   I2O_PRIVATE_MESSAGE_FRAME PvtMessageFrame;
   unsigned long             dataSize;
   char                      hltURL[MAX_I2O_SM_URLCHARS];
   char                      hltClassName[MAX_I2O_SM_URLCHARS];
   unsigned long             hltLocalId;
   unsigned long             hltInstance;
   unsigned long             hltTid;
   unsigned long             otherData;
   char                      data[1000];
} I2O_SM_OTHER_MESSAGE_FRAME, *PI2O_SM_OTHER_MESSAGE_FRAME;

#endif
