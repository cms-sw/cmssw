#ifndef DataFormats_Luminosity_LumiConstants_h
#define DataFormats_Luminosity_LumiConstants_h

// Various constants used by the lumi classes.

namespace LumiConstants
{
  static const unsigned int numOrbits = 262144; // number of orbits per LS (2^18)
  static const unsigned int numBX = 3564;       // number of BX per orbit
  static const float bxSpacingExact = 24.95e-9; // BX spacing (exact value)
  static const int bxSpacingInt = 25;           // BX spacing (in ns) -- "standard" value of 25
}

#endif
