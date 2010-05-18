// -*- C++ -*-
//
// Package:    MuonMCClassifier
// Class:      MuonMCClassifier
// 
/**\class MuonMCClassifier MuonMCClassifier.cc PhysicsTools/PatAlgos/src/MuonMCClassifier.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Nov 16 16:12 (lxplus231.cern.ch)
//         Created:  Sun Nov 16 16:14:09 CET 2008
// $Id: MuonMCClassifier.cc,v 1.1 2009/04/14 23:20:25 gpetrucc Exp $
//
//


// system include files
#include <memory>
#include <set>
#include <ext/hash_map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "SimMuon/MCTruth/interface/MuonAssociatorByHits.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include <SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h>

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

//
// class decleration
class MuonMCClassifier : public edm::EDProducer {
    public:
        explicit MuonMCClassifier(const edm::ParameterSet&);
        ~MuonMCClassifier();

    private:
        virtual void produce(edm::Event&, const edm::EventSetup&);
        /// The RECO objects
        edm::InputTag muons_;
 
        /// Track to use
        MuonAssociatorByHits::MuonTrackType trackType_;

        /// The TrackingParticle objects 
        edm::InputTag trackingParticles_;

        /// The Associations
        std::string associatorLabel_;

        TrackingParticle::GenParticleRef getGenParent(TrackingParticleRef tp) const ;

        /// Write a ValueMap<int> in the event
        template<typename T>
        void writeValueMap(edm::Event &iEvent,
                const edm::Handle<edm::View<reco::Muon> > & handle,
                const std::vector<T> & values,
                const std::string    & label) const ;


};

MuonMCClassifier::MuonMCClassifier(const edm::ParameterSet &iConfig) :
    muons_(iConfig.getParameter<edm::InputTag>("muons")),
    trackingParticles_(iConfig.getParameter<edm::InputTag>("trackingParticles")),
    associatorLabel_(iConfig.getParameter< std::string >("associatorLabel"))
{
    std::string trackType = iConfig.getParameter< std::string >("trackType");
    if (trackType == "inner") trackType_ = MuonAssociatorByHits::InnerTk;
    else if (trackType == "outer") trackType_ = MuonAssociatorByHits::OuterTk;
    else if (trackType == "global") trackType_ = MuonAssociatorByHits::GlobalTk;
    else if (trackType == "segments") trackType_ = MuonAssociatorByHits::Segments;
    else throw cms::Exception("Configuration") << "Track type '" << trackType << "' not supported.\n";

    produces<edm::ValueMap<int> >(); 
    produces<edm::ValueMap<int> >("hitsPdgId"); 
}

MuonMCClassifier::~MuonMCClassifier() 
{
}

void
MuonMCClassifier::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    edm::Handle<edm::View<reco::Muon> > muons; 
    iEvent.getByLabel(muons_, muons);

    edm::Handle<TrackingParticleCollection> trackingParticles;
    iEvent.getByLabel(trackingParticles_, trackingParticles);

    edm::ESHandle<TrackAssociatorBase> associatorBase;
    iSetup.get<TrackAssociatorRecord>().get(associatorLabel_, associatorBase);
    const MuonAssociatorByHits * assoByHits = dynamic_cast<const MuonAssociatorByHits *>(associatorBase.product());
    if (assoByHits == 0) throw cms::Exception("Configuration") << "The Track Associator with label '" << associatorLabel_ << "' is not a MuonAssociatorByHits.\n";

    MuonAssociatorByHits::MuonToSimCollection recSimColl;
    MuonAssociatorByHits::SimToMuonCollection simRecColl;
    assoByHits->associateMuons(recSimColl, simRecColl, muons, trackType_, trackingParticles, &iEvent, &iSetup);

    typedef MuonAssociatorByHits::MuonToSimCollection::const_iterator r2s_it;
    typedef MuonAssociatorByHits::SimToMuonCollection::const_iterator s2r_it;

    size_t nmu = muons->size();
    std::vector<int> classif(nmu, 0), hitsPdgId(nmu, 0);
    for(size_t i = 0; i < nmu; ++i) {
        edm::RefToBase<reco::Muon> mu = muons->refAt(i);
        r2s_it match = recSimColl.find(mu);
        if (match != recSimColl.end()) {
            TrackingParticleRef tp = match->second.front().first; // match->second is vector, front is first element, first is the ref (second would be the quality)
            s2r_it matchback = simRecColl.find(tp);
            if (matchback == simRecColl.end()) {
                edm::LogWarning("Unexpected") << "This I do NOT understand: why no match back?\n";
                continue;
            } 
            hitsPdgId[i] = tp->pdgId();
            TrackingParticle::GenParticleRef genp = getGenParent(tp);
            if (matchback->second.front().first != mu) {
                // you're a ghost
                classif[i] = -1;
            } else {
                if (genp.isNonnull() && abs(genp->pdg_id()) == 13) {
                    classif[i] = 1; // prompt muon
                } else if (abs(tp->pdgId()) == 13) {
                    classif[i] = 2; // decay muon
                } else {
                    classif[i] = 3; // identified punch-through
                }
            }
        }
    }

    writeValueMap(iEvent, muons, classif,   "");
    writeValueMap(iEvent, muons, hitsPdgId, "hitsPdgId");
}    

template<typename T>
void
MuonMCClassifier::writeValueMap(edm::Event &iEvent,
        const edm::Handle<edm::View<reco::Muon> > & handle,
        const std::vector<T> & values,
        const std::string    & label) const 
{
    using namespace edm; 
    using namespace std;
    auto_ptr<ValueMap<T> > valMap(new ValueMap<T>());
    typename edm::ValueMap<T>::Filler filler(*valMap);
    filler.insert(handle, values.begin(), values.end());
    filler.fill();
    iEvent.put(valMap, label);
}

TrackingParticle::GenParticleRef
MuonMCClassifier::getGenParent(TrackingParticleRef tp) const {
    if (!tp->genParticle().empty()) return tp->genParticle()[0];
    TrackingParticle::TrackingVertexRef prod = tp->parentVertex();
    if (prod.isNonnull()) {
        foreach(TrackingParticleRef par, prod->sourceTracks()) {
            TrackingParticle::GenParticleRef gp = getGenParent(par);
            if (gp.isNonnull()) return gp;
        }
    }
    return TrackingParticle::GenParticleRef();
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonMCClassifier);
