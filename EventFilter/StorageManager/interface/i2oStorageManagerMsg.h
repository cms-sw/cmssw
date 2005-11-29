#ifndef __i2oStorageManagerMsg_h__
#define __i2oStorageManagerMsg_h__

#include "i2o/i2o.h"

/*
   Author: Harry Cheung, FNAL

   Description:
     Used for FU I2O frame output module and by the
     Storage Manager I2O I2O input
     See CMS EventFilter wiki page for further notes.

   Modification:
     version 1.1 2005/11/23
       Initial implementation

*/

#define I2O_SM_PREAMBLE 0x001a
#define I2O_SM_DATA     0x001b
#define I2O_SM_OTHER    0x001c
// do we really want to hardwire this?
//max data I2O frame is (2**16 - 1) * 4 = 65535 * 4 = 262140
//max data array size is then 262140 - 28 -24 = 262088 bytes
#define MAX_I2O_SM_DATASIZE 262088
// Not sure if Registry always fit in a single I2O frame??!
// registry data array size is 262140 - 28 - 8 = 262104 bytes
#define MAX_I2O_REGISTRY_DATASIZE 262104
// we want to define the maximum event data size?
// max size is 20 x 262088 = about 5MB (used in testI2OReceiver only)
#define MAX_I2O_SM_DATAFRAMES 20

/**
 * Storage Manager Preample Message
 */
typedef struct _I2O_SM_PREAMBLE_MESSAGE_FRAME {
   I2O_PRIVATE_MESSAGE_FRAME PvtMessageFrame;
   unsigned long             dataSize;
   unsigned long             hltID;
   char                      data[MAX_I2O_REGISTRY_DATASIZE];
} I2O_SM_PREAMBLE_MESSAGE_FRAME, *PI2O_SM_PREAMBLE_MESSAGE_FRAME;

/**
 * Storage Manager Data Message
 */
typedef struct _I2O_SM_DATA_MESSAGE_FRAME {
   I2O_PRIVATE_MESSAGE_FRAME PvtMessageFrame;
   unsigned long             dataSize;
   unsigned long             hltID;
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
   unsigned long             hltID;
   unsigned long             otherData;
   char                      data[1000];
} I2O_SM_OTHER_MESSAGE_FRAME, *PI2O_SM_OTHER_MESSAGE_FRAME;

#endif
