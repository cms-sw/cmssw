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

#include "DataFormats/LaserAlignment/interface/LASAlignmentParameter.h"
#include "DataFormats/LaserAlignment/interface/LASAlignmentParameterCollection.h"
#include <valarray>

namespace {
	namespace {
		edm::Wrapper<LASAlignmentParameter> alignmentParameter;
    edm::Wrapper<std::valarray<double> adummy0;
	}
}

#include "DataFormats/LaserAlignment/interface/LASAlignmentParameterCollection.h"

namespace {
	namespace {
		edm::Wrapper<LASAlignmentParameterCollection> collection2;
	}
}

#endif
