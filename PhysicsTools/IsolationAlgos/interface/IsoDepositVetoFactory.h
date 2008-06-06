#ifndef PhysicsTools_IsolationAlgos_IsoDepositVetoFactory_h
#define PhysicsTools_IsolationAlgos_IsoDepositVetoFactory_h

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"

class IsoDepositVetoFactory {
public:
    /// Returns a pointer to an AbsVeto defined by the string.
    /// The calling code owns the pointer, and must delete it at the end.
    static reco::isodeposit::AbsVeto * make(const char *string) ;
};

#endif

