#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionData.h"
#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionsData.h"
#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionsDataSequence.h"

namespace DataFormats_CTPPSAlignment {
  struct dictionary {
	RPAlignmentCorrectionData ac;
	RPAlignmentCorrectionsData acs;
	RPAlignmentCorrectionsDataSequence acss;
  };
}
