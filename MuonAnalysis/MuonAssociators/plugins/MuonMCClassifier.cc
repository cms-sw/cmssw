// -*- C++ -*-
//
// Package:    MuonMCClassifier
// Class:      MuonMCClassifier
// 
/**\class MuonMCClassifier MuonMCClassifier.cc MuonAnalysis/MuonAssociators/src/MuonMCClassifier.cc


 CLASSIFICATION: For each RECO Muon, match to SIM particle, and then:
  - If the SIM is not a Muon, label as Punchthrough (1)
  - If the SIM is a Muon, then look at it's provenance.
     A) the SIM muon is also a GEN muon, whose parent is NOT A HADRON AND NOT A TAU
        -> classify as "primary" (4).
     B) the SIM muon is also a GEN muon, whose parent is HEAVY FLAVOURED HADRON OR A TAU
        -> classify as "heavy flavour" (3)
     C) classify as "light flavour/decay" (2)

  In any case, if the TP is not preferentially matched back to the same RECO muon,
  label as Ghost (flip the classification)


 FLAVOUR: 
  - for non-muons: 0
  - for primary muons: 13
  - for non primary muons: flavour of the mother: abs(pdgId) of heaviest quark, or 15 for tau
     
*/
//
// Original Author:  Nov 16 16:12 (lxplus231.cern.ch)
//         Created:  Sun Nov 16 16:14:09 CET 2008
// $Id: MuonMCClassifier.cc,v 1.9 2013/06/24 12:53:19 speer Exp $
//
//


// system include files
#include <memory>
#include <set>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Association.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "SimMuon/MCTruth/interface/MuonAssociatorByHits.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include <SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h>

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

//
// class decleration
class MuonMCClassifier : public edm::EDProducer {
    public:
        explicit MuonMCClassifier(const edm::ParameterSet&);
        ~MuonMCClassifier();

    private:
        virtual void produce(edm::Event&, const edm::EventSetup&) override;
        /// The RECO objects
        edm::InputTag muons_;

        /// A preselection cut for the muon. 
        /// I pass through pat::Muon so that I can access muon id selectors
        bool hasMuonCut_;
        StringCutObjectSelector<pat::Muon> muonCut_;
 
        /// Track to use
        MuonAssociatorByHits::MuonTrackType trackType_;

        /// The TrackingParticle objects 
        edm::InputTag trackingParticles_;

        /// The Associations
        std::string associatorLabel_;

        /// Cylinder to use to decide if a decay is early or late
        double decayRho_, decayAbsZ_;

        /// Create a link to the generator level particles
        bool linkToGenParticles_; 
        edm::InputTag genParticles_; 

        /// Returns the flavour given a pdg id code
        int flavour(int pdgId) const ;

        /// Write a ValueMap<int> in the event
        template<typename T>
        void writeValueMap(edm::Event &iEvent,
                const edm::Handle<edm::View<reco::Muon> > & handle,
                const std::vector<T> & values,
                const std::string    & label) const ;

        TrackingParticleRef getTpMother(TrackingParticleRef tp) {
            if (tp.isNonnull() && tp->parentVertex().isNonnull() && !tp->parentVertex()->sourceTracks().empty()) {
                return tp->parentVertex()->sourceTracks()[0];
            } else {
                return TrackingParticleRef();
            }
        }

        const HepMC::GenParticle * getGpMother(const HepMC::GenParticle *gp) {
            if (gp != 0) {
                const HepMC::GenVertex *vtx = gp->production_vertex();
                if (vtx != 0 && vtx->particles_in_size() > 0) {
                    return *vtx->particles_in_const_begin();
                }
            }
            return 0;
        }

        /// Find the index of a genParticle given it's barcode. -1 if not found
        int fetch(const edm::Handle<std::vector<int> > & genBarcodes, int barcode) const;

        /// Convert TrackingParticle into GenParticle, save into output collection,
        /// if mother is primary set reference to it,
        /// return index in output collection
        int convertAndPush(const TrackingParticle &tp, 
                           reco::GenParticleCollection &out, 
                           const TrackingParticleRef &momRef, 
                           const edm::Handle<reco::GenParticleCollection> & genParticles, 
                           const edm::Handle<std::vector<int> > & genBarcodes) const ;
};

MuonMCClassifier::MuonMCClassifier(const edm::ParameterSet &iConfig) :
    muons_(iConfig.getParameter<edm::InputTag>("muons")),
    hasMuonCut_(iConfig.existsAs<std::string>("muonPreselection")),
    muonCut_(hasMuonCut_ ? iConfig.getParameter<std::string>("muonPreselection") : ""),
    trackingParticles_(iConfig.getParameter<edm::InputTag>("trackingParticles")),
    associatorLabel_(iConfig.getParameter< std::string >("associatorLabel")),
    decayRho_(iConfig.getParameter<double>("decayRho")),
    decayAbsZ_(iConfig.getParameter<double>("decayAbsZ")),
    linkToGenParticles_(iConfig.getParameter<bool>("linkToGenParticles")),
    genParticles_(linkToGenParticles_ ? iConfig.getParameter<edm::InputTag>("genParticles") : edm::InputTag("NONE"))
{
    std::string trackType = iConfig.getParameter< std::string >("trackType");
    if (trackType == "inner") trackType_ = MuonAssociatorByHits::InnerTk;
    else if (trackType == "outer") trackType_ = MuonAssociatorByHits::OuterTk;
    else if (trackType == "global") trackType_ = MuonAssociatorByHits::GlobalTk;
    else if (trackType == "segments") trackType_ = MuonAssociatorByHits::Segments;
    else throw cms::Exception("Configuration") << "Track type '" << trackType << "' not supported.\n";

    produces<edm::ValueMap<int> >(); 
    produces<edm::ValueMap<int> >("ext"); 
    produces<edm::ValueMap<int> >("flav"); 
    produces<edm::ValueMap<int> >("hitsPdgId"); 
    produces<edm::ValueMap<int> >("momPdgId"); 
    produces<edm::ValueMap<int> >("momFlav"); 
    produces<edm::ValueMap<int> >("momStatus"); 
    produces<edm::ValueMap<int> >("gmomPdgId"); 
    produces<edm::ValueMap<int> >("gmomFlav"); 
    produces<edm::ValueMap<int> >("hmomFlav"); // heaviest mother flavour
    produces<edm::ValueMap<int> >("tpId");
    produces<edm::ValueMap<float> >("prodRho"); 
    produces<edm::ValueMap<float> >("prodZ"); 
    produces<edm::ValueMap<float> >("momRho"); 
    produces<edm::ValueMap<float> >("momZ"); 
    produces<edm::ValueMap<float> >("tpAssoQuality");
    if (linkToGenParticles_) {
        produces<reco::GenParticleCollection>("secondaries");
        produces<edm::Association<reco::GenParticleCollection> >("toPrimaries");
        produces<edm::Association<reco::GenParticleCollection> >("toSecondaries");
    }
}

MuonMCClassifier::~MuonMCClassifier() 
{
}

void
MuonMCClassifier::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    edm::LogVerbatim("MuonMCClassifier") <<"\n sono in MuonMCClassifier !";
  
    edm::Handle<edm::View<reco::Muon> > muons; 
    iEvent.getByLabel(muons_, muons);

    edm::Handle<TrackingParticleCollection> trackingParticles;
    iEvent.getByLabel(trackingParticles_, trackingParticles);

    edm::Handle<reco::GenParticleCollection> genParticles;
    edm::Handle<std::vector<int> > genBarcodes;
    if (linkToGenParticles_) {
        iEvent.getByLabel(genParticles_, genParticles);
        iEvent.getByLabel(genParticles_, genBarcodes);
    }

    edm::ESHandle<TrackAssociatorBase> associatorBase;
    iSetup.get<TrackAssociatorRecord>().get(associatorLabel_, associatorBase);
    const MuonAssociatorByHits * assoByHits = dynamic_cast<const MuonAssociatorByHits *>(associatorBase.product());
    if (assoByHits == 0) throw cms::Exception("Configuration") << "The Track Associator with label '" << associatorLabel_ << "' is not a MuonAssociatorByHits.\n";

    MuonAssociatorByHits::MuonToSimCollection recSimColl;
    MuonAssociatorByHits::SimToMuonCollection simRecColl;
    edm::LogVerbatim("MuonMCClassifier") <<"\n ***************************************************************** ";
    edm::LogVerbatim("MuonMCClassifier") <<  " RECO MUON association, type:  "<< trackType_;
    edm::LogVerbatim("MuonMCClassifier") <<  " ***************************************************************** \n";

    edm::RefToBaseVector<reco::Muon> selMuons;
    if (!hasMuonCut_) {
        // all muons
        selMuons = muons->refVector();
    } else {
        // filter, fill refvectors, associate
        // I pass through pat::Muon so that I can access muon id selectors
        for (size_t i = 0, n = muons->size(); i < n; ++i) {
            edm::RefToBase<reco::Muon> rmu = muons->refAt(i);
            if (muonCut_(pat::Muon(rmu))) selMuons.push_back(rmu);
        }
    }

    edm::RefVector<TrackingParticleCollection> allTPs;
    for (size_t i = 0, n = trackingParticles->size(); i < n; ++i) {
        allTPs.push_back(TrackingParticleRef(trackingParticles,i));
    }

    assoByHits->associateMuons(recSimColl, simRecColl, selMuons, trackType_, allTPs, &iEvent, &iSetup);

    // for global muons without hits on muon detectors, look at the linked standalone muon
    MuonAssociatorByHits::MuonToSimCollection UpdSTA_recSimColl;
    MuonAssociatorByHits::SimToMuonCollection UpdSTA_simRecColl;
    if (trackType_ == MuonAssociatorByHits::GlobalTk) {
      edm::LogVerbatim("MuonMCClassifier") <<"\n ***************************************************************** ";
      edm::LogVerbatim("MuonMCClassifier") <<  " STANDALONE (UpdAtVtx) MUON association ";
      edm::LogVerbatim("MuonMCClassifier") <<  " ***************************************************************** \n";
      assoByHits->associateMuons(UpdSTA_recSimColl, UpdSTA_simRecColl, selMuons, MuonAssociatorByHits::OuterTk, 
				 allTPs, &iEvent, &iSetup);
    }

    typedef MuonAssociatorByHits::MuonToSimCollection::const_iterator r2s_it;
    typedef MuonAssociatorByHits::SimToMuonCollection::const_iterator s2r_it;

    size_t nmu = muons->size();
    edm::LogVerbatim("MuonMCClassifier") <<"\n There are "<<nmu<<" reco::Muons.";

    std::vector<int> classif(nmu, 0), ext(nmu, 0);
    std::vector<int> hitsPdgId(nmu, 0), momPdgId(nmu, 0), gmomPdgId(nmu, 0), momStatus(nmu, 0);
    std::vector<int> flav(nmu, 0),      momFlav(nmu, 0),  gmomFlav(nmu, 0), hmomFlav(nmu, 0);
    std::vector<int> tpId(nmu, -1);
    std::vector<float> prodRho(nmu, 0.0), prodZ(nmu, 0.0), momRho(nmu, 0.0), momZ(nmu, 0.0);
    std::vector<float> tpAssoQuality(nmu, -1);

    std::auto_ptr<reco::GenParticleCollection> secondaries;     // output collection of secondary muons
    std::map<TrackingParticleRef, int>         tpToSecondaries; // map from tp to (index+1) in output collection
    std::vector<int> muToPrimary(nmu, -1), muToSecondary(nmu, -1); // map from input into (index) in output, -1 for null
    if (linkToGenParticles_) secondaries.reset(new reco::GenParticleCollection());

    for(size_t i = 0; i < nmu; ++i) {
        edm::LogVerbatim("MuonMCClassifier") <<"\n reco::Muons # "<<i;
        edm::RefToBase<reco::Muon> mu = muons->refAt(i);
        if (hasMuonCut_ && (std::find(selMuons.begin(), selMuons.end(), mu) == selMuons.end()) ) {
            edm::LogVerbatim("MuonMCClassifier") <<"\t muon didn't pass the selection. classified as -99 and skipped";
            classif[i] = -99; continue;
        }

        TrackingParticleRef        tp;
        edm::RefToBase<reco::Muon> muMatchBack;
        r2s_it match = recSimColl.find(mu);
        s2r_it matchback;
        if (match != recSimColl.end()) {
            edm::LogVerbatim("MuonMCClassifier") <<"\t RtS matched Ok...";
            // match->second is vector, front is first element, first is the ref (second would be the quality)
            tp = match->second.front().first;
            tpId[i]          = tp.isNonnull() ? tp.key() : -1; // we check, even if null refs should not appear here at all
            tpAssoQuality[i] = match->second.front().second;
            s2r_it matchback = simRecColl.find(tp);
            if (matchback != simRecColl.end()) {
                muMatchBack = matchback->second.front().first;
            } else {
                edm::LogWarning("MuonMCClassifier") << "\n***WARNING:  This I do NOT understand: why no match back? *** \n";
            }
        } else if ((trackType_ == MuonAssociatorByHits::GlobalTk) &&
                    mu->isGlobalMuon()) {
            // perform a second attempt, matching with the standalone muon
            r2s_it matchSta = UpdSTA_recSimColl.find(mu);
            if (matchSta != UpdSTA_recSimColl.end()) {
                edm::LogVerbatim("MuonMCClassifier") <<"\t RtS matched Ok... from the UpdSTA_recSimColl ";
                tp    = matchSta->second.front().first;
                tpId[i]          = tp.isNonnull() ? tp.key() : -1; // we check, even if null refs should not appear here at all
                tpAssoQuality[i] = matchSta->second.front().second;
                s2r_it matchback = UpdSTA_simRecColl.find(tp);
                if (matchback != UpdSTA_simRecColl.end()) {
                    muMatchBack = matchback->second.front().first;
                } else {
                    edm::LogWarning("MuonMCClassifier") << "\n***WARNING:  This I do NOT understand: why no match back in UpdSTA? *** \n";
                }
            }
        } 
        if (tp.isNonnull()) {
            bool isGhost = muMatchBack != mu;
            if (isGhost) edm::LogVerbatim("MuonMCClassifier") <<"\t This seems a GHOST ! classif[i] will be < 0";

            hitsPdgId[i] = tp->pdgId();
            prodRho[i]   = tp->vertex().Rho(); 
            prodZ[i]     = tp->vertex().Z();
	    edm::LogVerbatim("MuonMCClassifier") <<"\t TP pdgId = "<<hitsPdgId[i] << ", vertex rho = " << prodRho[i] << ", z = " << prodZ[i];

            // Try to extract mother and grand mother of this muon.
            // Unfortunately, SIM and GEN histories require diffent code :-(
            if (!tp->genParticles().empty()) { // Muon is in GEN
#warning "This file has been modified just to get it to compile without any regard as to whether it still functions as intended"
#ifdef REMOVED_JUST_TO_GET_IT_TO_COMPILE__THIS_CODE_NEEDS_TO_BE_CHECKED
                const HepMC::GenParticle * genMom = getGpMother(tp->genParticle()[0].get());
#else
                const HepMC::GenParticle * genMom = NULL;
#endif
                if (genMom) {
                    momPdgId[i]  = tp->pdgId();
                    momStatus[i] = tp->status();
                    if (genMom->production_vertex()) {
                        const HepMC::ThreeVector & momVtx = genMom->production_vertex()->point3d();
                        momRho[i] = momVtx.perp() * 0.1; momZ[i] = momVtx.z() * 0.1; // HepMC is in mm!
                    }
                    edm::LogVerbatim("MuonMCClassifier") << "\t Particle pdgId = "<<hitsPdgId[i] << " produced at rho = " << prodRho[i] << ", z = " << prodZ[i] << ", has GEN mother pdgId = " << momPdgId[i];
                    const HepMC::GenParticle * genGMom = getGpMother(genMom);
                    if (genGMom) {
                        gmomPdgId[i] = genGMom->pdg_id();
                        edm::LogVerbatim("MuonMCClassifier") << "\t\t mother prod. vertex rho = " << momRho[i] << ", z = " << momZ[i] << ", grand-mom pdgId = " << gmomPdgId[i];
                    }
                    // in this case, we might want to know the heaviest mom flavour
                    for (const HepMC::GenParticle *nMom = genMom; 
                            nMom != 0 && abs(nMom->pdg_id()) >= 100; // stop when we're no longer looking at hadrons or mesons
                            nMom = getGpMother(nMom)) {
                        int flav = flavour(nMom->pdg_id());
                        if (hmomFlav[i] < flav) hmomFlav[i] = flav; 
                        edm::LogVerbatim("MuonMCClassifier") << "\t\t backtracking flavour: mom pdgId = "<<nMom->pdg_id()<< ", flavour = " << flav << ", heaviest so far = " << hmomFlav[i];
                    }
                }
            } else { // Muon is in SIM Only
                TrackingParticleRef simMom = getTpMother(tp);
                if (simMom.isNonnull()) {
                    momPdgId[i] = simMom->pdgId();
                    momRho[i] = simMom->vertex().Rho();
                    momZ[i]   = simMom->vertex().Z();
                    edm::LogVerbatim("MuonMCClassifier") << "\t Particle pdgId = "<<hitsPdgId[i] << " produced at rho = " << prodRho[i] << ", z = " << prodZ[i] << 
                                                            ", has SIM mother pdgId = " << momPdgId[i] << " produced at rho = " << simMom->vertex().Rho() << ", z = " << simMom->vertex().Z();
                    if (!simMom->genParticles().empty()) {
                        momStatus[i] = simMom->genParticles()[0]->status();
#warning "This file has been modified just to get it to compile without any regard as to whether it still functions as intended"
                        gmomPdgId[i] = simMom->pdgId();
                        edm::LogVerbatim("MuonMCClassifier") << "\t\t SIM mother is in GEN (status " << momStatus[i] << "), grand-mom id = " << gmomPdgId[i];
                    } else {
                        momStatus[i] = -1;
                        TrackingParticleRef simGMom = getTpMother(simMom);
                        if (simGMom.isNonnull()) gmomPdgId[i] = simGMom->pdgId();
                        edm::LogVerbatim("MuonMCClassifier") << "\t\t SIM mother is in SIM only, grand-mom id = " << gmomPdgId[i];
                    }
                } else {
                  edm::LogVerbatim("MuonMCClassifier") << "\t Particle pdgId = "<<hitsPdgId[i] << " produced at rho = " << prodRho[i] << ", z = " << prodZ[i] << ", has NO mother!";
                }
            }
            momFlav[i]  = flavour(momPdgId[i]);
            gmomFlav[i] = flavour(gmomPdgId[i]);

            // Check first IF this is a muon at all
            if (abs(tp->pdgId()) != 13) {
                classif[i] = isGhost ? -1 : 1;
                ext[i]     = isGhost ? -1 : 1;
                edm::LogVerbatim("MuonMCClassifier") <<"\t This is not a muon. Sorry. classif[i] = " << classif[i];
                continue;
            }

            // Is this SIM muon also a GEN muon, with a mother?
            if (!tp->genParticles().empty() && (momPdgId[i] != 0)) {
                if (abs(momPdgId[i]) < 100 && (abs(momPdgId[i]) != 15)) {
                    classif[i] = isGhost ? -4 : 4;
                    flav[i] = (abs(momPdgId[i]) == 15 ? 15 : 13);
                    edm::LogVerbatim("MuonMCClassifier") <<"\t This seems PRIMARY MUON ! classif[i] = " << classif[i];
                    ext[i] = 10;
                } else if (momFlav[i] == 4 || momFlav[i] == 5 || momFlav[i] == 15) {
                    classif[i] = isGhost ? -3 : 3;
                    flav[i]    = momFlav[i];
                    if (momFlav[i] == 15)      ext[i] = 9; // tau->mu
                    else if (momFlav[i] == 5)  ext[i] = 8; // b->mu
                    else if (hmomFlav[i] == 5) ext[i] = 7; // b->c->mu
                    else                       ext[i] = 6; // c->mu
                    edm::LogVerbatim("MuonMCClassifier") <<"\t This seems HEAVY FLAVOUR ! classif[i] = " << classif[i];
                } else {
                    classif[i] = isGhost ? -2 : 2;
                    flav[i]    = momFlav[i];
                    edm::LogVerbatim("MuonMCClassifier") <<"\t This seems LIGHT FLAVOUR ! classif[i] = " << classif[i];
                }
            } else {
                classif[i] = isGhost ? -2 : 2;
                flav[i]    = momFlav[i];
                edm::LogVerbatim("MuonMCClassifier") <<"\t This seems LIGHT FLAVOUR ! classif[i] = " << classif[i];
            }
            // extended classification
            if (momPdgId[i] == 0) ext[i] = 2; // if it has no mom, it's not a primary particle so it won't be in ppMuX
            else if (abs(momPdgId[i]) < 100) ext[i] = (momFlav[i] == 15 ? 9 : 10); // primary mu, or tau->mu
            else if (momFlav[i] == 5) ext[i] = 8; // b->mu
            else if (momFlav[i] == 4) ext[i] = (hmomFlav[i] == 5 ? 7 : 6); // b->c->mu and c->mu
            else if (momStatus[i] != -1) { // primary light particle
                int id = abs(momPdgId[i]);
                if (id != /*pi+*/211 && id != /*K+*/321 && id != 130 /*K0L*/)  ext[i] = 5; // other light particle, possibly short-lived
                else if (prodRho[i] < decayRho_ && abs(prodZ[i]) < decayAbsZ_) ext[i] = 4; // decay a la ppMuX (primary pi/K within a cylinder)
                else                                                           ext[i] = 3; // late decay that wouldn't be in ppMuX
            } else ext[i] = 2; // decay of non-primary particle, would not be in ppMuX
            if (isGhost) ext[i] = -ext[i];

            if (linkToGenParticles_ && abs(ext[i]) >= 2) {
                // Link to the genParticle if possible, but not decays in flight (in ppMuX they're in GEN block, but they have wrong parameters)
                if (!tp->genParticles().empty() && abs(ext[i]) >= 5) {
#warning "This file has been modified just to get it to compile without any regard as to whether it still functions as intended"
#ifdef REMOVED_JUST_TO_GET_IT_TO_COMPILE__THIS_CODE_NEEDS_TO_BE_CHECKED
                    muToPrimary[i] = fetch(genBarcodes, tp->genParticle()[0]->barcode());
#endif
                } else {
                    // Don't put the same trackingParticle twice!
                    int &indexPlus1 = tpToSecondaries[tp]; // will create a 0 if the tp is not in the list already
                    if (indexPlus1 == 0) indexPlus1 = convertAndPush(*tp, *secondaries, getTpMother(tp), genParticles, genBarcodes) + 1;
                    muToSecondary[i] = indexPlus1 - 1; 
                }
            }
            edm::LogVerbatim("MuonMCClassifier") <<"\t Extended classification code = " << ext[i];
	}
    }
    
    writeValueMap(iEvent, muons, classif,   "");
    writeValueMap(iEvent, muons, ext,       "ext");
    writeValueMap(iEvent, muons, flav,      "flav");
    writeValueMap(iEvent, muons, tpId,      "tpId");
    writeValueMap(iEvent, muons, hitsPdgId, "hitsPdgId");
    writeValueMap(iEvent, muons, momPdgId,  "momPdgId");
    writeValueMap(iEvent, muons, momStatus, "momStatus");
    writeValueMap(iEvent, muons, momFlav,   "momFlav");
    writeValueMap(iEvent, muons, gmomPdgId, "gmomPdgId");
    writeValueMap(iEvent, muons, gmomFlav,  "gmomFlav");
    writeValueMap(iEvent, muons, hmomFlav,  "hmomFlav");
    writeValueMap(iEvent, muons, prodRho,   "prodRho");
    writeValueMap(iEvent, muons, prodZ,     "prodZ");
    writeValueMap(iEvent, muons, momRho,    "momRho");
    writeValueMap(iEvent, muons, momZ,      "momZ");
    writeValueMap(iEvent, muons, tpAssoQuality, "tpAssoQuality");

    if (linkToGenParticles_) {
        edm::OrphanHandle<reco::GenParticleCollection> secHandle = iEvent.put(secondaries, "secondaries");
        edm::RefProd<reco::GenParticleCollection> priRP(genParticles); 
        edm::RefProd<reco::GenParticleCollection> secRP(secHandle);
        std::auto_ptr<edm::Association<reco::GenParticleCollection> > outPri(new edm::Association<reco::GenParticleCollection>(priRP));
        std::auto_ptr<edm::Association<reco::GenParticleCollection> > outSec(new edm::Association<reco::GenParticleCollection>(secRP));
        edm::Association<reco::GenParticleCollection>::Filler fillPri(*outPri), fillSec(*outSec);
        fillPri.insert(muons, muToPrimary.begin(),   muToPrimary.end());
        fillSec.insert(muons, muToSecondary.begin(), muToSecondary.end());
        fillPri.fill(); fillSec.fill();
        iEvent.put(outPri, "toPrimaries");
        iEvent.put(outSec, "toSecondaries");
    }
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

int
MuonMCClassifier::flavour(int pdgId) const {
    int flav = abs(pdgId);
    // for quarks, leptons and bosons except gluons, take their pdgId
    // muons and taus have themselves as flavour
    if (flav <= 37 && flav != 21) return flav;
    // look for barions
    int bflav = ((flav / 1000) % 10);
    if (bflav != 0) return bflav;
    // look for mesons
    int mflav = ((flav / 100) % 10);
    if (mflav != 0) return mflav;
    return 0;
}

int MuonMCClassifier::fetch(const edm::Handle<std::vector<int> > & genBarcodes, int barcode) const
{
    std::vector<int>::const_iterator it = std::find(genBarcodes->begin(), genBarcodes->end(), barcode);
    return (it == genBarcodes->end()) ? -1 : (it - genBarcodes->begin());
}

// push secondary in collection.
// if it has a primary mother link to it. 
int MuonMCClassifier::convertAndPush(const TrackingParticle &tp, 
                                     reco::GenParticleCollection &out,
                                     const TrackingParticleRef & simMom, 
                                     const edm::Handle<reco::GenParticleCollection> & genParticles,
                                     const edm::Handle<std::vector<int> > & genBarcodes) const {
    out.push_back(reco::GenParticle(tp.charge(), tp.p4(), tp.vertex(), tp.pdgId(), tp.status(), true));
    if (simMom.isNonnull() && !simMom->genParticles().empty()) {
#warning "This file has been modified just to get it to compile without any regard as to whether it still functions as intended"
#ifdef REMOVED_JUST_TO_GET_IT_TO_COMPILE__THIS_CODE_NEEDS_TO_BE_CHECKED
        int momIdx = fetch(genBarcodes, simMom->genParticle()[0]->barcode());
        if (momIdx != -1)  out.back().addMother(reco::GenParticleRef(genParticles, momIdx));
#endif
    }
    return out.size()-1;
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonMCClassifier);
