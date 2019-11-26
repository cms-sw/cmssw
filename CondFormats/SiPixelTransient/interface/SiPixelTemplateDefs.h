//
//  SiPixelTemplateDefs.h (v2.00)
//
// Created by Morris Swartz on 12/01/09.
// 2009 __TheJohnsHopkinsUniversity__.
//
// V2.00 - Resize the 2D objects to improve angle acceptance
//
//

// Define template buffer size parameters

#ifndef SiPixelTemplateDefs_h
#define SiPixelTemplateDefs_h 1

// Switch to use boost multiarrays to store the template entries (instead of plain c arrays).
// It adds real time re-sizing and bounds checking at a cost in time (10%?).

//#define SI_PIXEL_TEMPLATE_USE_BOOST 1

#define TYSIZE 21
#define THY 10  // = TYSIZE/2
#define THYP1 THY + 1
#define TYTEN 210  // = 10*TYSIZE
#define BYSIZE TYSIZE + 4
#define BHY 12  // = BYSIZE/2
#define BYM1 TYSIZE + 3
#define BYM2 TYSIZE + 2
#define BYM3 TYSIZE + 1
#define TXSIZE 13
#define THX 6  // = TXSIZE/2
#define THXP1 THX + 1
#define BXSIZE TXSIZE + 4
#define BHX 8  // = BXSIZE/2
#define BXM1 TXSIZE + 3
#define BXM2 TXSIZE + 2
#define BXM3 TXSIZE + 1
#define T2YSIZE 21
#define T2XSIZE 7
#define T2HY 10          // = T2YSIZE/2
#define T2HYP1 T2HY + 1  // = T2YSIZE/2+1
#define T2HX 3           // = T2XSIZE/2
#define TEMP_ENTRY_SIZEX_A 60
#define TEMP_ENTRY_SIZEX_B 60
#define TEMP_ENTRY_SIZEY 60
#endif
