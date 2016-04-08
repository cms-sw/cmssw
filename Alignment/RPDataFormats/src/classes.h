#include "DataFormats/Common/interface/Wrapper.h"

#include "Alignment/RPDataFormats/interface/LocalTrackFit.h"
#include "Alignment/RPDataFormats/interface/RPAlignmentCorrection.h"
#include "Alignment/RPDataFormats/interface/RPAlignmentCorrections.h"

namespace {
  namespace {
	LocalTrackFit ltf;
	edm::Wrapper<LocalTrackFit> wltf;

	RPAlignmentCorrection ac;
	RPAlignmentCorrections acs;
  }
}
