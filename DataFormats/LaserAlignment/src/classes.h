
#ifndef DataFormats_LaserAlignment_classes_h
#define DataFormats_LaserAlignment_classes_h

// this is the new section, the rest is obsolete ///////////
#include "DataFormats/Common/interface/Wrapper.h"
#include "Test/TestHolder/interface/SiStripLaserRecHit2D.h"
#include "Test/TestHolder/interface/TkLasBeam.h"

namespace {
  namespace {
    TkLasBeam beam1;
    edm::Wrapper<TkLasBeam> beam2;
  }
}

namespace {
  namespace {
    TkLasBeamCollection beamCollection1;
    edm::Wrapper<TkLasBeamCollection> beamCollection2;
  }
}
// end of new section //////////////////////////////////////


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

namespace {
	namespace {
		edm::Wrapper<LASAlignmentParameter> alignmentParameter;
	}
}

#include "DataFormats/LaserAlignment/interface/LASAlignmentParameterCollection.h"

namespace {
	namespace {
		edm::Wrapper<LASAlignmentParameterCollection> collection2;
	}
}

#endif
