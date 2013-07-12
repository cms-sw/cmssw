#ifndef FWCore_Concurrency_hardware_pause_h
#define FWCore_Concurrency_hardware_pause_h
// -*- C++ -*-
//
// Package:     Concurrency
// Class  :     hardware_pause
// 
/**\class hardware_pause hardware_pause.h FWCore/Concurrency/interface/hardware_pause.h

 Description: assembler instruction to allow a short pause

 Usage:
    This hardware instruction tells the CPU to pause momentarily. This can be useful
 in the case where one is doing a 'spin lock' on a quantity that you expect to change
 within a few clock cycles.

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 13:55:57 CST 2013
// $Id$
//

//NOTE: Taken from libdispatch shims/atomics.h
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2)
#define hardware_pause()      asm("")
#endif
#if defined(__x86_64__) || defined(__i386__)
#undef hardware_pause
#define hardware_pause() asm("pause")
#endif


#endif
