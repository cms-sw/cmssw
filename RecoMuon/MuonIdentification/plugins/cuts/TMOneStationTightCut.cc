#include "MuonSelectorVIDWrapper.h"

typedef MuonSelectorVIDWrapper<muon::TMOneStationTight> TMOneStationTightCut;

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  TMOneStationTightCut,
		  "TMOneStationTightCut");
