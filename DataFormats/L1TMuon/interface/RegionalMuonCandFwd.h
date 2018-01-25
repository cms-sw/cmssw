#ifndef __l1t_regional_muon_candidatefwd_h__
#define __l1t_regional_muon_candidatefwd_h__

#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {
	enum tftype {
  	bmtf, omtf_neg, omtf_pos, emtf_neg, emtf_pos
	};
	class RegionalMuonCand;
	typedef BXVector<RegionalMuonCand> RegionalMuonCandBxCollection;
}

#endif /* define __l1t_regional_muon_candidatefwd_h__ */
