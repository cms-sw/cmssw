#ifndef PhysicsTools_IsolationAlgos_EventDependentAbsVetos_h
#define PhysicsTools_IsolationAlgos_EventDependentAbsVetos_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "PhysicsTools/IsolationAlgos/interface/EventDependentAbsVeto.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"

namespace reco {
 namespace isodeposit {
    class OtherCandidatesDeltaRVeto : public EventDependentAbsVeto {
      public:
          //! Create a veto specifying the input collection of the candidates, and the deltaR
          OtherCandidatesDeltaRVeto(const edm::InputTag& candidates, double deltaR, edm::ConsumesCollector& iC) :
            src_(iC.consumes<edm::View<reco::Candidate> >(candidates)), deltaR2_(deltaR*deltaR) { }

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
          edm::EDGetTokenT<edm::View<reco::Candidate> > src_;
          float         deltaR2_;
          std::vector<Direction> items_;
    };

    class OtherCandVeto : public EventDependentAbsVeto {
      public:
          //! Create a veto specifying the input collection of the candidates, and the deltaR
          OtherCandVeto(const edm::InputTag& candidates, AbsVeto *veto, edm::ConsumesCollector& iC) :
            src_(iC.consumes<edm::View<reco::Candidate> >(candidates)), veto_(veto) { }

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
          edm::EDGetTokenT<edm::View<reco::Candidate> > src_;
          std::vector<Direction> items_;
          std::auto_ptr<AbsVeto> veto_;
    };

    class OtherJetConstituentsDeltaRVeto : public EventDependentAbsVeto {
      public:
          //! Create a veto specifying the input collection of the jets, the candidates, and the deltaR
          OtherJetConstituentsDeltaRVeto(Direction dir, const edm::InputTag& jets, double dRjet, const edm::InputTag& pfCandAssocMap, double dRconstituent, edm::ConsumesCollector& iC)
	    : evt_(0),
	      vetoDir_(dir),
	      srcJets_(iC.consumes<reco::PFJetCollection>(jets)),
	      dR2jet_(dRjet*dRjet),
	      srcPFCandAssocMap_(iC.consumes<JetToPFCandidateAssociation>(pfCandAssocMap)),
	      dR2constituent_(dRconstituent*dRconstituent)
	  {
	    //std::cout << "<OtherJetConstituentsDeltaRVeto::OtherJetConstituentsDeltaRVeto>:" << std::endl;
	    //std::cout << " vetoDir: eta = " << vetoDir_.eta() << ", phi = " << vetoDir_.phi() << std::endl;
	    //std::cout << " srcJets = " << srcJets_.label() << ":" << srcJets_.instance() << std::endl;
	    //std::cout << " dRjet = " << sqrt(dR2jet_) << std::endl;
	    //std::cout << " srcPFCandAssocMap = " << srcPFCandAssocMap_.label() << ":" << srcPFCandAssocMap_.instance() << std::endl;
	    //std::cout << " dRconstituent = " << sqrt(dR2constituent_) << std::endl;
	  }

          // Virtual destructor (should always be there)
          virtual ~OtherJetConstituentsDeltaRVeto() {}

          //! Return "true" if a deposit at specific (eta,phi) with that value must be vetoed in the sum
          //! This is true if the deposit is within the stored AbsVeto of any item of the source collection
          virtual bool veto(double eta, double phi, float value) const;

          //! Set axis for matching jets
          virtual void centerOn(double eta, double phi);

          //! Picks up the directions of the given candidates
          virtual void setEvent(const edm::Event& evt, const edm::EventSetup& es);

      private:
          typedef edm::AssociationMap<edm::OneToMany<std::vector<reco::PFJet>, std::vector<reco::PFCandidate>, unsigned int> > JetToPFCandidateAssociation;

	  void initialize();

	  const edm::Event* evt_;

	  Direction vetoDir_;
          edm::EDGetTokenT<reco::PFJetCollection> srcJets_;
	  double dR2jet_;
	  edm::EDGetTokenT<JetToPFCandidateAssociation> srcPFCandAssocMap_;
	  double dR2constituent_;
          std::vector<Direction> items_;
    };
 }
}
#endif
