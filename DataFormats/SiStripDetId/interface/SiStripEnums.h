#ifndef SISTRIPENUMS_H
#define SISTRIPENUMS_H

// P1 Tracker

namespace SiStripSubdetector {
  enum Subdetector { UNKNOWN = 0, TIB = 3, TID = 4, TOB = 5, TEC = 6 };
}

enum class SiStripModuleGeometry { UNKNOWNGEOMETRY, IB1, IB2, OB1, OB2, W1A, W2A, W3A, W1B, W2B, W3B, W4, W5, W6, W7 };

// P2 Tracker

namespace Phase2Tracker {
  enum Subdetector { UNKNOWN = 0, Endcap = 4, Barrel = 5 };
  enum BarrelModuleTilt { nonBarrel = 0, tiltedZminus = 1, tiltedZplus = 2, flat = 3 };
}  // namespace Phase2Tracker

#endif
