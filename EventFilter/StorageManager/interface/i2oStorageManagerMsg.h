#ifndef __i2oStorageManagerMsg_h__
#define __i2oStorageManagerMsg_h__

#include "i2o/i2o.h"
#include <string.h>

#include "IOPool/Streamer/interface/MsgTools.h"

/*
   Description:
     Used for FU I2O frame output module and by the
     Storage Manager I2O input
     See CMS EvF Storage Manager wiki page for further notes.

   $Id$
*/

// These are the I2O function codes (should be) reserved for SM use
#define I2O_SM_PREAMBLE 0x001a
#define I2O_SM_DATA     0x001b
#define I2O_SM_OTHER    0x001c
#define I2O_SM_DQM      0x001d
//
// RunNumber_t and EventNumber_t are unsigned long variables
// We will use uint32 instead for 64-bit compatibility
//
// source (FU/HLT) id could be compressed into fewer bytes!
//
// max I2O frame is (2**16 - 1) * 4 = 65535 * 4 = 262140
// but this size should also be a multiple of 64 bits (8 bytes)
#define I2O_ABSOLUTE_MAX_SIZE 262136
// Illustrate calculation of max data I2O frame for given KB:
// If 32KB (e.g. for MTCC-I) max = (2**13 - 1) * 4 = 32764 
// but to be a multiple of 8 bytes it needs to be = 32760
//
// Actual value can be defined to any multiple of 8 bytes
// and less than I2O_ABSOLUTE_MAX_SIZE
#define I2O_MAX_SIZE 64000
// max data I2O frame needs to be calculated as
// I2O_MAX_SIZE - headers = I2O_MAX_SIZE - 28 -136
#define MAX_I2O_SM_DATASIZE 63836
// registry data array size is I2O_MAX_SIZE - 28 - 128
#define MAX_I2O_REGISTRY_DATASIZE 63844
// other data array size is I2O_MAX_SIZE - 28 - 132
#define MAX_I2O_OTHER_DATASIZE 63840
// DQM data array size is I2O_MAX_SIZE - 28 - 136
#define MAX_I2O_DQM_DATASIZE 63836
// maximum characters for the source class name and url
#define MAX_I2O_SM_URLCHARS 50

/**
 * Storage Manager Multi-part Message Base Struct (class)
 *   all multi-part messages build on this one
 */
struct _I2O_SM_MULTIPART_MESSAGE_FRAME {
   I2O_PRIVATE_MESSAGE_FRAME PvtMessageFrame;
   uint32                    dataSize;
   char                      hltURL[MAX_I2O_SM_URLCHARS];
   char                      hltClassName[MAX_I2O_SM_URLCHARS];
   uint32                    hltLocalId;
   uint32                    hltInstance;
   uint32                    hltTid;
   uint32                    numFrames;
   uint32                    frameCount;
   uint32                    originalSize;
};

typedef struct _I2O_SM_MULTIPART_MESSAGE_FRAME
    I2O_SM_MULTIPART_MESSAGE_FRAME, *PI2O_SM_MULTIPART_MESSAGE_FRAME;

/**
 * Storage Manager Preample Message
 *   we want the run number if we knew it in this message
 */
typedef struct _I2O_SM_PREAMBLE_MESSAGE_FRAME : _I2O_SM_MULTIPART_MESSAGE_FRAME {
   char* dataPtr()           const { return (char*)this+sizeof(_I2O_SM_PREAMBLE_MESSAGE_FRAME); }
} I2O_SM_PREAMBLE_MESSAGE_FRAME, *PI2O_SM_PREAMBLE_MESSAGE_FRAME;

/**
 * Storage Manager Data Message
 */
typedef struct _I2O_SM_DATA_MESSAGE_FRAME : _I2O_SM_MULTIPART_MESSAGE_FRAME {
   uint32                    runID;
   uint32                    eventID;
   char* dataPtr()           const { return (char*)this+sizeof(_I2O_SM_DATA_MESSAGE_FRAME); }
} I2O_SM_DATA_MESSAGE_FRAME, *PI2O_SM_DATA_MESSAGE_FRAME;

/**
 * Storage Manager OTHER Messages
 */
typedef struct _I2O_SM_OTHER_MESSAGE_FRAME : _I2O_SM_MULTIPART_MESSAGE_FRAME {
   uint32                    otherData;
   char* dataPtr()           const { return (char*)this+sizeof(_I2O_SM_OTHER_MESSAGE_FRAME); }
} I2O_SM_OTHER_MESSAGE_FRAME, *PI2O_SM_OTHER_MESSAGE_FRAME;

/**
 * Storage Manager OTHER Messages
 */
typedef struct _I2O_SM_DQM_MESSAGE_FRAME : _I2O_SM_MULTIPART_MESSAGE_FRAME {
   uint32                    runID;
   uint32                    folderID;
   char* dataPtr()           const { return (char*)this+sizeof(_I2O_SM_DQM_MESSAGE_FRAME); }
} I2O_SM_DQM_MESSAGE_FRAME, *PI2O_SM_DQM_MESSAGE_FRAME;

#endif
