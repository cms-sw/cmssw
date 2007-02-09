#ifndef DataFormats_LaserAlignment_classes_h
#define DataFormats_LaserAlignment_classes_h

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/LaserAlignment/interface/LASBeamProfileFit.h"
#include "DataFormats/LaserAlignment/interface/LASBeamProfileFitCollection.h"

namespace {
  namespace {
    edm::Wrapper<LASBeamProfileFit> beamprofilefit;
  }
}

#include "DataFormats/LaserAlignment/interface/LASBeamProfileFitCollection.h"

namespace {
  namespace {
    edm::Wrapper<LASBeamProfileFitCollection> collection;
  }
}

#endif
