
#ifndef DataFormats_LaserAlignment_classes_h
#define DataFormats_LaserAlignment_classes_h

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/LaserAlignment/interface/SiStripLaserRecHit2D.h"
#include "DataFormats/LaserAlignment/interface/TkLasBeam.h"
#include "DataFormats/LaserAlignment/interface/TkFittedLasBeam.h"

namespace {
  struct dictionary { // lighter than the old recommendation of namespace...
    TkLasBeam beam1; // not needed since not templated?
    edm::Wrapper<TkLasBeam> beam2; // not needed since not an EDProduct?
    TkLasBeamCollection beamCollection1;
    edm::Wrapper<TkLasBeamCollection> beamCollection2;

    TkFittedLasBeam fitBeam1; // not needed since not templated?
    TkFittedLasBeamCollection fitBeamCollection1;
    edm::Wrapper<TkFittedLasBeamCollection> fitBeamCollection2;

  };
}

#endif
