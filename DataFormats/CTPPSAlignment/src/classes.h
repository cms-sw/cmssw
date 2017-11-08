#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionData.h"
#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionsData.h"

namespace DataFormats_CTPPSAlignment {
  struct dictionary {
	RPAlignmentCorrectionData ac;
	RPAlignmentCorrectionsData acs;
  };
}
