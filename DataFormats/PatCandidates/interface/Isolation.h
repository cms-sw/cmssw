#ifndef DataFormats_PatCandidates_interface_Isolation_h
#define DataFormats_PatCandidates_interface_Isolation_h

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"

namespace pat {
  typedef reco::IsoDeposit IsoDeposit;
  /// Enum defining isolation keys
  enum IsolationKeys {
    TrackIso = 0,
    EcalIso = 1,
    HcalIso = 2,
    PfAllParticleIso = 3,
    PfChargedHadronIso = 4,
    PfNeutralHadronIso = 5,
    PfGammaIso = 6,
    User1Iso = 7,
    User2Iso = 8,
    User3Iso = 9,
    User4Iso = 10,
    User5Iso = 11,
    UserBaseIso = 7,  // offset of the first user isolation
    CaloIso = -1,     // keys which are not real indices are mapped to negative numbers.
    PfPUChargedHadronIso = 12,
    PfChargedAllIso = 13
  };
}  // namespace pat

#endif
