#ifndef PhysicsTools_IsolationAlgos_EventDependentAbsVeto_h
#define PhysicsTools_IsolationAlgos_EventDependentAbsVeto_h

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace reco {
 namespace isodeposit {
    class EventDependentAbsVeto : public AbsVeto {
        public:
            virtual ~EventDependentAbsVeto() {}
            virtual void setEvent(const edm::Event &iEvent, const edm::EventSetup &iSetup) = 0;
    };
    typedef std::vector<EventDependentAbsVeto *> EventDependentAbsVetos;
 }
}

#endif
