#ifndef DataFormats_PatCandidates_interface_Isolation_h
#define DataFormats_PatCandidates_interface_Isolation_h

#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"

namespace pat {
    typedef reco::MuIsoDeposit IsoDeposit;
    /// Enum defining isolation keys
    enum IsolationKeys { TrackerIso=0, ECalIso=1, HCalIso=2, 
        User1Iso=3, User2Iso=4, User3Iso=5, User4Iso=6, User5Iso=7,
        UserBaseIso=3, // offset of the first user isolation
        CaloIso=-1     // keys which are not real indices are mapped to negative numbers.
    };
}

#endif
