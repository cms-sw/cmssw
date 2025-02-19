#ifndef I2OEVFMSGS_H
#define I2OEVFMSGS_H


#include "i2o/i2o.h"
#include <string.h>


//#include "IOPool/Streamer/interface/MsgTools.h"

/*
   Description:
   ------------
   define communication protocoll between FUResourceBroker ('FU') and
   StorageManager ('SM').
   
   $Id: i2oEvfMsgs.h,v 1.5 2012/05/02 15:13:23 smorovic Exp $
*/

// I2O function codes: *_SM_* / *_FU_* according to who *receives* the message
#define I2O_SM_PREAMBLE     0x001a
#define I2O_SM_DATA         0x001b
#define I2O_SM_ERROR        0x001c
#define I2O_SM_OTHER        0x001d
#define I2O_SM_DQM          0x001e
#define I2O_FU_DATA_DISCARD 0x001f
#define I2O_FU_DQM_DISCARD  0x0020

//
// RunNumber_t and EventNumber_t are unsigned long variables
// We will use U32 instead for 64-bit compatibility
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
// set the default if no value is given in the output module
#define I2O_MAX_SIZE 64000

// max data I2O frame needs to be calculated as e.g.
// I2O_MAX_SIZE - headers = I2O_MAX_SIZE - 28 -136
// now done dynamically in e.g. FUStreamerI2OWriter.cc
//
// maximum characters for the source class name and url
#define MAX_I2O_SM_URLCHARS 50


/**
 * Storage Manager Multi-part Message Base Struct (class)
 *   all multi-part messages build on this one
 */
struct _I2O_SM_MULTIPART_MESSAGE_FRAME
{
  I2O_PRIVATE_MESSAGE_FRAME PvtMessageFrame;
  U32                       dataSize;
  char                      hltURL[MAX_I2O_SM_URLCHARS];
  char                      hltClassName[MAX_I2O_SM_URLCHARS];
  U32                       hltLocalId;
  U32                       hltInstance;
  U32                       hltTid;
  U32                       numFrames;
  U32                       frameCount;
  U32                       originalSize;
};

typedef struct _I2O_SM_MULTIPART_MESSAGE_FRAME
I2O_SM_MULTIPART_MESSAGE_FRAME, *PI2O_SM_MULTIPART_MESSAGE_FRAME;


/**
 * Storage Manager Preample Message
 *   we want the run number if we knew it in this message
 */
typedef struct _I2O_SM_PREAMBLE_MESSAGE_FRAME : _I2O_SM_MULTIPART_MESSAGE_FRAME
{
  U32 rbBufferID;
  U32 outModID;
  U32 fuProcID;
  U32 fuGUID;
  U32 nExpectedEPs;
  char* dataPtr() const
  {
    return (char*)this+sizeof(_I2O_SM_PREAMBLE_MESSAGE_FRAME);
  }
}
I2O_SM_PREAMBLE_MESSAGE_FRAME, *PI2O_SM_PREAMBLE_MESSAGE_FRAME;


/**
 * Storage Manager Data Message
 */
typedef struct _I2O_SM_DATA_MESSAGE_FRAME : _I2O_SM_MULTIPART_MESSAGE_FRAME
{
  U32   rbBufferID;
  U32   runID;
  U32   eventID;
  U32   outModID;
  U32   fuProcID;
  U32   fuGUID;
  char* dataPtr() const {
    return (char*)this+sizeof(_I2O_SM_DATA_MESSAGE_FRAME);
  }
}
I2O_SM_DATA_MESSAGE_FRAME, *PI2O_SM_DATA_MESSAGE_FRAME;


/**
 * Storage Manager Data Discard Message
 */
typedef struct _I2O_FU_DATA_DISCARD_MESSAGE_FRAME
{
  I2O_PRIVATE_MESSAGE_FRAME PvtMessageFrame;
  U32                       rbBufferID;
}
I2O_FU_DATA_DISCARD_MESSAGE_FRAME, *PI2O_FU_DATA_DISCARD_MESSAGE_FRAME;


/**
 * Storage Manager DQM Messages
 */
typedef struct _I2O_SM_DQM_MESSAGE_FRAME : _I2O_SM_MULTIPART_MESSAGE_FRAME
{
  U32   rbBufferID;
  U32   runID;
  U32   eventAtUpdateID;
  U32   folderID;
  U32   fuProcID;
  U32   fuGUID;
  char* dataPtr() const
  {
    return (char*)this+sizeof(_I2O_SM_DQM_MESSAGE_FRAME);
  }
}
I2O_SM_DQM_MESSAGE_FRAME, *PI2O_SM_DQM_MESSAGE_FRAME;


/**
 * FUResourceBroker DQM Discard Messages
 */
typedef struct _I2O_FU_DQM_DISCARD_MESSAGE_FRAME
{
  I2O_PRIVATE_MESSAGE_FRAME PvtMessageFrame;
  U32                       rbBufferID;
}
I2O_FU_DQM_DISCARD_MESSAGE_FRAME, *PI2O_FU_DQM_DISCARD_MESSAGE_FRAME;


/**
 * Storage Manager OTHER Messages
 */
typedef struct _I2O_SM_OTHER_MESSAGE_FRAME : _I2O_SM_MULTIPART_MESSAGE_FRAME
{
  U32   otherData;
  char* dataPtr() const
  {
    return (char*)this+sizeof(_I2O_SM_OTHER_MESSAGE_FRAME);
  }
}
I2O_SM_OTHER_MESSAGE_FRAME, *PI2O_SM_OTHER_MESSAGE_FRAME;


#endif
