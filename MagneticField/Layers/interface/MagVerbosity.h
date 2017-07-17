#ifndef MagDebug_H
#define MagDebug_H

/*
 *  Hack while we wait for logging in the framework
 *
 *  \author N. Amapane - INFN Torino
 */

//#DEFINE MF_DEBUG

// Old debug control switch, being phased out
struct verbose {
#ifdef MF_DEBUG
  static constexpr bool debugOut = true;
#else
  static constexpr bool debugOut = false;
#endif
};

#endif

