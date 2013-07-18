#ifndef PhysicsTools_IsolationAlgos_EventDependentAbsVetos_h
#define PhysicsTools_IsolationAlgos_EventDependentAbsVetos_h

#include "PhysicsTools/IsolationAlgos/interface/EventDependentAbsVeto.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace reco {
 namespace isodeposit {
    class OtherCandidatesDeltaRVeto : public EventDependentAbsVeto {
      public:
          //! Create a veto specifying the input collection of the candidates, and the deltaR
          OtherCandidatesDeltaRVeto(const edm::InputTag candidates, double deltaR) :
            src_(candidates), deltaR2_(deltaR*deltaR) { }
   
          // Virtual destructor (should always be there) 
          virtual ~OtherCandidatesDeltaRVeto() {} 

          //! Return "true" if a deposit at specific (eta,phi) with that value must be vetoed in the sum
          //! This is true if the deposit is within the configured deltaR from any item of the source collection
          virtual bool veto(double eta, double phi, float value) const ;

          //! Nothing to do for this
          virtual void centerOn(double eta, double phi) { }

          //! Picks up the directions of the given candidates
          virtual void setEvent(const edm::Event &iEvent, const edm::EventSetup &iSetup) ;

      private:
          edm::InputTag src_;
          float         deltaR2_;
          std::vector<Direction> items_;
    };

    class OtherCandVeto : public EventDependentAbsVeto {
      public:
          //! Create a veto specifying the input collection of the candidates, and the deltaR
          OtherCandVeto(const edm::InputTag candidates, AbsVeto *veto) :
            src_(candidates), veto_(veto) { }
   
          // Virtual destructor (should always be there) 
          virtual ~OtherCandVeto() {} 

          //! Return "true" if a deposit at specific (eta,phi) with that value must be vetoed in the sum
          //! This is true if the deposit is within the stored AbsVeto of any item of the source collection
          virtual bool veto(double eta, double phi, float value) const ;

          //! Nothing to do for this
          virtual void centerOn(double eta, double phi) { }

          //! Picks up the directions of the given candidates
          virtual void setEvent(const edm::Event &iEvent, const edm::EventSetup &iSetup) ;

      private:
          edm::InputTag src_;
          std::vector<Direction> items_;
          std::auto_ptr<AbsVeto> veto_;
    };
 }
}
#endif
