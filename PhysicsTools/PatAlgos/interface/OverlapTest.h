#ifndef PhysicsTools_PatAlgos_interface_OverlapTest_h
#define PhysicsTools_PatAlgos_interface_OverlapTest_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "PhysicsTools/PatUtils/interface/StringParserTools.h"
#include "PhysicsTools/PatUtils/interface/PATDiObjectProxy.h"

namespace pat { namespace helper {

// Base class for a test for overlaps
class OverlapTest {
    public:
        /// constructor: reads 'src' and 'requireNoOverlaps' parameters
        OverlapTest(const std::string &name, const edm::ParameterSet &iConfig, edm::ConsumesCollector & iC) :
            srcToken_(iC.consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("src"))),
            name_(name),
            requireNoOverlaps_(iConfig.getParameter<bool>("requireNoOverlaps")) {}
        /// destructor, does nothing
        virtual ~OverlapTest() {}
        /// initializer for each event. to be implemented in child classes.
        virtual void readInput(const edm::Event & iEvent, const edm::EventSetup &iSetup) = 0;
        /// check for overlaps for a given item. to be implemented in child classes
        /// return true if overlaps have been found, and fills the PtrVector
        virtual bool fillOverlapsForItem(const reco::Candidate &item, reco::CandidatePtrVector &overlapsToFill) const = 0;
        /// end of event method. does nothing
        virtual void done() {}
        // -- basic getters ---

        const std::string & name() const { return name_; }
        bool requireNoOverlaps() const { return requireNoOverlaps_; }
    protected:
        edm::EDGetTokenT<reco::CandidateView> srcToken_;
        std::string   name_;
        bool          requireNoOverlaps_;
};

class BasicOverlapTest : public OverlapTest {
    public:
        BasicOverlapTest(const std::string &name, const edm::ParameterSet &iConfig, edm::ConsumesCollector && iC) :
            OverlapTest(name, iConfig, iC),
            presel_(iConfig.getParameter<std::string>("preselection")),
            deltaR_(iConfig.getParameter<double>("deltaR")),
            checkRecoComponents_(iConfig.getParameter<bool>("checkRecoComponents")),
            pairCut_(iConfig.getParameter<std::string>("pairCut")) {}
        // implementation of mother methods
        /// Read input, apply preselection cut
        virtual void readInput(const edm::Event & iEvent, const edm::EventSetup &iSetup) ;
        /// Check for overlaps
        virtual bool fillOverlapsForItem(const reco::Candidate &item, reco::CandidatePtrVector &overlapsToFill) const ;
    protected:
        // ---- configurables ----
        /// A generic preselection cut that can work on any Candidate, but has access also to methods of PAT specific objects
        PATStringCutObjectSelector presel_;
        /// Delta R for the match
        double deltaR_;
        /// Check the overlapping by RECO components
        bool checkRecoComponents_;
        /// Cut on the pair of objects together
        StringCutObjectSelector<pat::DiObjectProxy> pairCut_;
        // ---- working variables ----
        /// The collection to check overlaps against
        edm::Handle<reco::CandidateView> candidates_;
        /// Flag saying if each element has passed the preselection or not
        std::vector<bool> isPreselected_;
};

class OverlapBySuperClusterSeed : public OverlapTest {
    public:
        // constructor: nothing except initialize the base class
        OverlapBySuperClusterSeed(const std::string &name, const edm::ParameterSet &iConfig, edm::ConsumesCollector && iC) : OverlapTest(name, iConfig, iC) {}
        // every event: nothing except read the input list
        virtual void readInput(const edm::Event & iEvent, const edm::EventSetup &iSetup) {
            iEvent.getByToken(srcToken_, others_);
        }
         /// Check for overlaps
        virtual bool fillOverlapsForItem(const reco::Candidate &item, reco::CandidatePtrVector &overlapsToFill) const ;
    protected:
//         edm::Handle<edm::View<reco::RecoCandidate> > others_;
        edm::Handle<reco::CandidateView> others_;
};



} } // namespaces

#endif
