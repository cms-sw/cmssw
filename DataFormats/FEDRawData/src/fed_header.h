/* NOTE: this file temporarily stolen from the XDAQ project
 * 
 * fed_header.h  
 *
 * $Header: /cvs_server/repositories/CMSSW/CMSSW/DataFormats/FEDRawData/src/fed_header.h,v 1.1 2005/09/30 08:13:36 namapane Exp $
 *
 * This file contains the struct definition of the FED-header.
 * The FED-header is inserted by the FED at the top of a data fragment
 */


#ifndef _FED_HEADER_H
#define _FED_HEADER_H

#ifdef __cplusplus
extern "C" {
#endif


/*************************************************************************
 *
 * data structures and associated typedefs
 *
 *************************************************************************/


/*
 * FED header - in front of each FED block
 */

typedef struct fedh_struct {
  unsigned int sourceid ;   
  unsigned int eventid;
} fedh_t ;


#define FED_HCTRLID_MASK  0xF0000000
#define FED_EVTY_MASK     0x0F000000
#define FED_LVL1_MASK     0x00FFFFFF

#define FED_BXID_MASK     0xFFF00000
#define FED_SOID_MASK     0x000FFF00
#define FED_VERSION_MASK  0x000000F0
#define FED_MORE_HEADERS  0x00000008


#define FED_HCTRLID       0x50000000


#define FED_HCTRLID_SHIFT 28
#define FED_EVTY_SHIFT    24
#define FED_LVL1_SHIFT    0

#define FED_BXID_SHIFT    20
#define FED_SOID_SHIFT    8
#define FED_VERSION_SHIFT 4

#ifdef __cplusplus
}
#endif

#endif  /* _FED_HEADER_H */


