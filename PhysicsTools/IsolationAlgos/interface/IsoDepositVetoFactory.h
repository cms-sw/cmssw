#ifndef PhysicsTools_IsolationAlgos_IsoDepositVetoFactory_h
#define PhysicsTools_IsolationAlgos_IsoDepositVetoFactory_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "PhysicsTools/IsolationAlgos/interface/EventDependentAbsVeto.h"

class IsoDepositVetoFactory {
public:
    /// Returns a pointer to an AbsVeto defined by the string.
    /// The calling code owns the pointer, and must delete it at the end.
    /// An exception will be thrown if the resulting AbsVeto depends on the edm::Event
    static reco::isodeposit::AbsVeto * make(const char *string, edm::ConsumesCollector& iC) ;

    /// As above, but will allow also AbsVetos which depend from the edm::Event
    /// If the resulting veto is dependent on the edm::Event, the value of the second pointer will be set to non-zero
    /// Note that both pointers will point to the same object, so you have to delete it only once.
    static reco::isodeposit::AbsVeto * make(const char *string, reco::isodeposit::EventDependentAbsVeto *&evdep, edm::ConsumesCollector& iC) ;
};

#endif

