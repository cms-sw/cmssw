//
//  SiStripTemplateDefs.h (v2.1)
//
// V2.0 - Expand the x dimensions to accommodate strip clusters to size 10.8
// V2.1 - Change to strip specific names and remove un-necessary definitions
//
// Created by Morris Swartz on 10/11/10.
// 2010 __TheJohnsHopkinsUniversity__. 
//
//
 
// Define template buffer size parameters 

#ifndef SiStripTemplateDefs_h
#define SiStripTemplateDefs_h 1

// Switch to use boost multiarrays to store the template entries (instead of plain c arrays).  
// It adds real time re-sizing and bounds checking at a cost in time (10%?).

//#define SI_STRIP_TEMPLATE_USE_BOOST 1

#define TSXSIZE 17
#define TSHX 8 // = TSXSIZE/2
#define TSHXP1 TSHX+1
#define BSXSIZE TSXSIZE+4
#define BSHX 10 // = BSXSIZE/2
#define BSXM1 TSXSIZE+3
#define BSXM2 TSXSIZE+2
#define BSXM3 TSXSIZE+1

#endif
