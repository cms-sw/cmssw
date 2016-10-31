#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/CTPPSAlignment/interface/LocalTrackFit.h"
#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionData.h"
#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionsData.h"

namespace DataFormats_CTPPSAlignment {
  struct dictionary {
	LocalTrackFit ltf;
	edm::Wrapper<LocalTrackFit> wltf;

	RPAlignmentCorrectionData ac;
	RPAlignmentCorrectionsData acs;
  };
}
