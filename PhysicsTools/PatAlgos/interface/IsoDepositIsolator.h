#ifndef PhysicsTools_PatAlgos_interface_IsoDepositIsolator_h
#define PhysicsTools_PatAlgos_interface_IsoDepositIsolator_h

#include "PhysicsTools/PatAlgos/interface/BaseIsolator.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"


namespace pat { namespace helper {
class IsoDepositIsolator : public BaseIsolator {
    public:
        typedef edm::ValueMap<reco::IsoDeposit> Isolation;
 
        IsoDepositIsolator() {}
        IsoDepositIsolator(const edm::ParameterSet &conf, bool withCut) ;
        virtual ~IsoDepositIsolator() ;
        virtual void beginEvent(const edm::Event &event) ;
        virtual void endEvent() ;

        virtual std::string description() const ;
    protected:
        enum Mode { Sum, SumRelative, Count };
        edm::Handle<Isolation> handle_;

        float deltaR_;
        Mode  mode_;
        reco::isodeposit::AbsVetos vetos_;
        bool skipDefaultVeto_;

        virtual float getValue(const edm::ProductID &id, size_t index) const ;
        
}; // class IsoDepositIsolator
} } // namespaces

#endif
