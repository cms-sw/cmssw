#include "MuonSelectorVIDWrapper.h"

typedef MuonSelectorVIDWrapper<muon::GlobalMuonPromptTight> GlobalMuonPromptTightCut;

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GlobalMuonPromptTightCut,
		  "GlobalMuonPromptTightCut");
