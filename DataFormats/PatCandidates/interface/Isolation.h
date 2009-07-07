#ifndef DataFormats_PatCandidates_interface_Isolation_h
#define DataFormats_PatCandidates_interface_Isolation_h

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"

namespace pat {
    typedef reco::IsoDeposit IsoDeposit;
    /// Enum defining isolation keys
    enum IsolationKeys { TrackerIso=0, ECalIso=1, HCalIso=2,
			 ParticleIso=3, ChargedHadronIso=4, NeutralHadronIso=5, PhotonIso=6,  
			 User1Iso=7, User2Iso=8, User3Iso=9, User4Iso=10, User5Iso=11,
			 UserBaseIso=7, // offset of the first user isolation
			 CaloIso=-1     // keys which are not real indices are mapped to negative numbers.
    };
}

#endif
