/* NOTE: this file temporarily stolen from the XDAQ project
 *
 * fed_trailer.h  
 *
 * $Header: /cvs_server/repositories/CMSSW/CMSSW/DataFormats/FEDRawData/src/fed_trailer.h,v 1.1 2005/09/30 08:13:36 namapane Exp $
 *
 * This file contains the struct definition of the FED-trailer.
 * The FED-trailer is inserted by the FED at the bottom of a data fragment
 */


#ifndef _FED_TRAILER_H
#define _FED_TRAILER_H

#ifdef __cplusplus
extern "C" {
#endif


/*************************************************************************
 *
 * data structures and associated typedefs
 *
 *************************************************************************/


/*
 * FED trailer - at the end of each FED block
 */

typedef struct fedt_struct {
  unsigned int conscheck;
  unsigned int eventsize;
} fedt_t ;


#define FED_TCTRLID_MASK   0xF0000000
#define FED_EVSZ_MASK      0x00FFFFFF

#define FED_CRCS_MASK      0xFFFF0000
#define FED_STAT_MASK      0x00000F00
#define FED_TTSI_MASK      0x000000F0
#define FED_MORE_TRAILERS  0x00000008


#define FED_TCTRLID        0xA0000000


#define FED_EVSZ_SHIFT      0

#define FED_CRCS_SHIFT      16
#define FED_STAT_SHIFT      8
#define FED_TTSI_SHIFT      4

#ifdef __cplusplus
}
#endif

#endif  /* _FED_TRAILER_H */


