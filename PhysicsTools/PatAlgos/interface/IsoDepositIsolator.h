#ifndef PhysicsTools_PatAlgos_interface_IsoDepositIsolator_h
#define PhysicsTools_PatAlgos_interface_IsoDepositIsolator_h

#include "PhysicsTools/PatAlgos/interface/BaseIsolator.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "PhysicsTools/IsolationAlgos/interface/EventDependentAbsVeto.h"


namespace pat { namespace helper {
class IsoDepositIsolator : public BaseIsolator {
    public:
        typedef edm::ValueMap<reco::IsoDeposit> Isolation;

        IsoDepositIsolator() {}
        IsoDepositIsolator(const edm::ParameterSet &conf, edm::ConsumesCollector & iC, bool withCut) ;
        virtual ~IsoDepositIsolator() ;
        virtual void beginEvent(const edm::Event &event, const edm::EventSetup &eventSetup) ;
        virtual void endEvent() ;

        virtual std::string description() const ;
    protected:
        enum Mode { Sum, Sum2, SumRelative, Sum2Relative, Max, MaxRelative, Count };
        edm::Handle<Isolation> handle_;

        float deltaR_;
        Mode  mode_;
        reco::isodeposit::AbsVetos vetos_;
        reco::isodeposit::EventDependentAbsVetos evdepVetos_; // subset of the above, don't delete twice
        bool skipDefaultVeto_;
        edm::EDGetTokenT<Isolation> inputIsoDepositToken_;

        virtual float getValue(const edm::ProductID &id, size_t index) const ;
}; // class IsoDepositIsolator
} } // namespaces

#endif
