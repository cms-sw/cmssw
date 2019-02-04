// -*- C++ -*-
//
// Package:    MuonMCClassifier
// Class:      MuonMCClassifier
//
/**\class MuonMCClassifier MuonMCClassifier.cc MuonAnalysis/MuonAssociators/src/MuonMCClassifier.cc


 CLASSIFICATION: For each RECO Muon, match to SIM particle, and then:
  - If the SIM is not a Muon, label as Punchthrough (1) except if it is an electron or positron (11)
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
  - for non primary muons: flavour of the mother: std::abs(pdgId) of heaviest quark, or 15 for tau

*/
//
// Original Author:  G.Petrucciani and G.Abbiendi
//         Created:  Sun Nov 16 16:14:09 CET 2008
//         revised:  3/Aug/2017
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
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "SimDataFormats/Associations/interface/MuonToTrackingParticleAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include <SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h>

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

//
// class decleration
class MuonMCClassifier : public edm::EDProducer {
    public:
        explicit MuonMCClassifier(const edm::ParameterSet&);
        ~MuonMCClassifier() override;

    private:
        void produce(edm::Event&, const edm::EventSetup&) override;
        /// The RECO objects
        edm::EDGetTokenT<edm::View<reco::Muon> > muonsToken_;

        /// A preselection cut for the muon.
        /// I pass through pat::Muon so that I can access muon id selectors
        bool hasMuonCut_;
        StringCutObjectSelector<pat::Muon> muonCut_;

        /// Track to use
        reco::MuonTrackType trackType_;

        /// The TrackingParticle objects
        edm::EDGetTokenT<TrackingParticleCollection> trackingParticlesToken_;

        /// The Associations
        edm::InputTag associatorLabel_;
        edm::EDGetTokenT<reco::MuonToTrackingParticleAssociator> muAssocToken_;

        /// Cylinder to use to decide if a decay is early or late
        double decayRho_, decayAbsZ_;

        /// Create a link to the generator level particles
        bool linkToGenParticles_;
        edm::InputTag genParticles_;
        edm::EDGetTokenT<reco::GenParticleCollection> genParticlesToken_;

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

        /// Convert TrackingParticle into GenParticle, save into output collection,
        /// if mother is primary set reference to it,
        /// return index in output collection
        int convertAndPush(const TrackingParticle &tp,
                           reco::GenParticleCollection &out,
                           const TrackingParticleRef &momRef,
                           const edm::Handle<reco::GenParticleCollection> & genParticles) const ;
};


MuonMCClassifier::MuonMCClassifier(const edm::ParameterSet &iConfig) :
    muonsToken_(consumes<edm::View<reco::Muon> >(iConfig.getParameter<edm::InputTag>("muons"))),
    hasMuonCut_(iConfig.existsAs<std::string>("muonPreselection")),
    muonCut_(hasMuonCut_ ? iConfig.getParameter<std::string>("muonPreselection") : ""),
    trackingParticlesToken_(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("trackingParticles"))),
    muAssocToken_(consumes<reco::MuonToTrackingParticleAssociator>(iConfig.getParameter<edm::InputTag>("associatorLabel"))),
    decayRho_(iConfig.getParameter<double>("decayRho")),
    decayAbsZ_(iConfig.getParameter<double>("decayAbsZ")),
    linkToGenParticles_(iConfig.getParameter<bool>("linkToGenParticles")),
    genParticles_(linkToGenParticles_ ? iConfig.getParameter<edm::InputTag>("genParticles") : edm::InputTag("NONE"))

{
    std::string trackType = iConfig.getParameter< std::string >("trackType");
    if (trackType == "inner") trackType_ = reco::InnerTk;
    else if (trackType == "outer") trackType_ = reco::OuterTk;
    else if (trackType == "global") trackType_ = reco::GlobalTk;
    else if (trackType == "segments") trackType_ = reco::Segments;
    else if (trackType == "glb_or_trk") trackType_ = reco::GlbOrTrk;
    else throw cms::Exception("Configuration") << "Track type '" << trackType << "' not supported.\n";
    if (linkToGenParticles_) {
      genParticlesToken_ = consumes<reco::GenParticleCollection>(genParticles_);
    }

    produces<edm::ValueMap<int> >();
    produces<edm::ValueMap<int> >("ext");
    produces<edm::ValueMap<int> >("flav");
    produces<edm::ValueMap<int> >("hitsPdgId");
    produces<edm::ValueMap<int> >("G4processType"); // Geant process producing the particle
    produces<edm::ValueMap<int> >("momPdgId");
    produces<edm::ValueMap<int> >("momFlav");
    produces<edm::ValueMap<int> >("momStatus");
    produces<edm::ValueMap<int> >("gmomPdgId");
    produces<edm::ValueMap<int> >("gmomFlav");
    produces<edm::ValueMap<int> >("hmomFlav"); // heaviest mother flavour
    produces<edm::ValueMap<int> >("tpId");
    produces<edm::ValueMap<int> >("tpEv");
    produces<edm::ValueMap<int> >("tpBx");    
    produces<edm::ValueMap<float> >("signp");
    produces<edm::ValueMap<float> >("pt");
    produces<edm::ValueMap<float> >("eta");
    produces<edm::ValueMap<float> >("phi");
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
    edm::Handle<edm::View<reco::Muon> > muons;
    iEvent.getByToken(muonsToken_, muons);

    edm::Handle<TrackingParticleCollection> trackingParticles;
    iEvent.getByToken(trackingParticlesToken_, trackingParticles);

    edm::Handle<reco::GenParticleCollection> genParticles;
    if (linkToGenParticles_) {
        iEvent.getByToken(genParticlesToken_, genParticles);
    }

    edm::Handle<reco::MuonToTrackingParticleAssociator> associatorBase;
    iEvent.getByToken(muAssocToken_, associatorBase);
    const reco::MuonToTrackingParticleAssociator * assoByHits = associatorBase.product();

    reco::MuonToSimCollection recSimColl;
    reco::SimToMuonCollection simRecColl;
    edm::LogVerbatim("MuonMCClassifier") <<"\n ***************************************************************** ";
    edm::LogVerbatim("MuonMCClassifier") <<  " RECO MUON association, type:  "<< trackType_;
    edm::LogVerbatim("MuonMCClassifier") <<  " ***************************************************************** \n";

    edm::RefToBaseVector<reco::Muon> selMuons;
    if (!hasMuonCut_) {
        // all muons
        for (size_t i = 0, n = muons->size(); i < n; ++i) {
            edm::RefToBase<reco::Muon> rmu = muons->refAt(i);
            selMuons.push_back(rmu);
        }
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

    assoByHits->associateMuons(recSimColl, simRecColl, selMuons, trackType_, allTPs);

    // for global muons without hits on muon detectors, look at the linked standalone muon
    reco::MuonToSimCollection UpdSTA_recSimColl;
    reco::SimToMuonCollection UpdSTA_simRecColl;
    if (trackType_ == reco::GlobalTk) {
      edm::LogVerbatim("MuonMCClassifier") <<"\n ***************************************************************** ";
      edm::LogVerbatim("MuonMCClassifier") <<  " STANDALONE (UpdAtVtx) MUON association ";
      edm::LogVerbatim("MuonMCClassifier") <<  " ***************************************************************** \n";
      assoByHits->associateMuons(UpdSTA_recSimColl, UpdSTA_simRecColl, selMuons, reco::OuterTk,
				 allTPs);
    }

    typedef reco::MuonToSimCollection::const_iterator r2s_it;
    typedef reco::SimToMuonCollection::const_iterator s2r_it;

    size_t nmu = muons->size();
    edm::LogVerbatim("MuonMCClassifier") <<"\n There are "<<nmu<<" reco::Muons.";

    std::vector<int> classif(nmu, 0), ext(nmu, 0);
    std::vector<int> hitsPdgId(nmu, 0), G4processType(nmu, 0), momPdgId(nmu, 0), gmomPdgId(nmu, 0), momStatus(nmu, 0);
    std::vector<int> flav(nmu, 0), momFlav(nmu, 0), gmomFlav(nmu, 0), hmomFlav(nmu, 0);
    std::vector<int> tpId(nmu, -1), tpBx(nmu, 999), tpEv(nmu, 999);
    std::vector<float> signp(nmu, 0.0), pt(nmu, 0.0), eta(nmu, -99.), phi(nmu, -99.);
    std::vector<float> prodRho(nmu, 0.0), prodZ(nmu, 0.0), momRho(nmu, 0.0), momZ(nmu, 0.0);
    std::vector<float> tpAssoQuality(nmu, -1);

    std::unique_ptr<reco::GenParticleCollection> secondaries;     // output collection of secondary muons
    std::map<TrackingParticleRef, int>         tpToSecondaries; // map from tp to (index+1) in output collection
    std::vector<int> muToPrimary(nmu, -1), muToSecondary(nmu, -1); // map from input into (index) in output, -1 for null
    if (linkToGenParticles_) secondaries.reset(new reco::GenParticleCollection());

    // loop on reco muons
    for(size_t i = 0; i < nmu; ++i) {    
        edm::RefToBase<reco::Muon> mu = muons->refAt(i);
        if (hasMuonCut_ && (std::find(selMuons.begin(), selMuons.end(), mu) == selMuons.end()) ) {
	  LogTrace("MuonMCClassifier") <<"\n reco::Muon # "<<i<<"didn't pass the selection. classified as -99 and skipped";
            classif[i] = -99; continue;
        } 
	else edm::LogVerbatim("MuonMCClassifier") <<"\n reco::Muon # "<<i;

        TrackingParticleRef        tp;
        edm::RefToBase<reco::Muon> muMatchBack;
        r2s_it match = recSimColl.find(mu);
        s2r_it matchback;
        if (match != recSimColl.end()) {
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
        } else if ((trackType_ == reco::GlobalTk) &&
                    mu->isGlobalMuon()) {
            // perform a second attempt, matching with the standalone muon
            r2s_it matchSta = UpdSTA_recSimColl.find(mu);
            if (matchSta != UpdSTA_recSimColl.end()) {
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
        } else {
            edm::LogVerbatim("MuonMCClassifier") <<"\t No matching TrackingParticle is found ";
	}

        if (tp.isNonnull()) {
            bool isGhost = muMatchBack != mu;
            if (isGhost) edm::LogVerbatim("MuonMCClassifier") <<"\t *** This seems a Duplicate muon ! classif[i] will be < 0 ***";

	    // identify signal and pileup TP
	    tpBx[i]      = tp->eventId().bunchCrossing();
	    tpEv[i]      = tp->eventId().event();

            hitsPdgId[i] = tp->pdgId();
            prodRho[i]   = tp->vertex().Rho();
            prodZ[i]     = tp->vertex().Z();

	    // added info on GEANT process producing the TrackingParticle
	    const std::vector<SimVertex> & G4Vs = tp->parentVertex()->g4Vertices();
	    G4processType[i] = G4Vs[0].processType();

	    signp[i] = tp->charge() * tp->p();
	    pt[i]    = tp->pt();
	    eta[i]   = tp->eta();
	    phi[i]   = tp->phi();

            // Try to extract mother and grand mother of this muon.
            // Unfortunately, SIM and GEN histories require diffent code :-(
            if (!tp->genParticles().empty()) { // Muon is in GEN
	      reco::GenParticleRef genp   = tp->genParticles()[0];
	      reco::GenParticleRef genMom = genp->numberOfMothers() > 0 ? genp->motherRef() : reco::GenParticleRef();
	      reco::GenParticleRef mMom = genMom;
	      
	      if (genMom.isNonnull()) {
		if (genMom->pdgId() != tp->pdgId()) {
		  momPdgId[i]  = genMom->pdgId();
		  momStatus[i] = genMom->status();
		  momRho[i] = genMom->vertex().Rho(); 
		  momZ[i] = genMom->vz();
		}
		else {
		  // if mother has the same identity look backwards for the real mother (it may happen in radiative decays) 
		  int jm = 0;
		  while (mMom->pdgId() == tp->pdgId()) {
		    jm++;

		    if (mMom->numberOfMothers() > 0) {
		      mMom = mMom->motherRef();
		    }
		    LogTrace("MuonMCClassifier") 
		      << "\t\t backtracking mother "<<jm<<", pdgId = "<<mMom->pdgId()<<", status= " <<mMom->status();
		  }
		  genMom = mMom; // redefine genMom
		  momPdgId[i]  = genMom->pdgId();
		  momStatus[i] = genMom->status();
		  momRho[i] = genMom->vertex().Rho(); 
		  momZ[i] = genMom->vz();
		}
		edm::LogVerbatim("MuonMCClassifier") 
		  << "\t Particle pdgId = "<<hitsPdgId[i] << ", (Event,Bx) = "<< "(" <<tpEv[i]<<","<<tpBx[i]<<")"
		  <<"\n\t   q*p = "<<signp[i]<<", pT = "<<pt[i]<<", eta = "<<eta[i]<<", phi = "<<phi[i]
		  << "\n\t   produced at vertex rho = " << prodRho[i] << ", z = " << prodZ[i] 
		  << ", (GEANT4 process = "<< G4processType[i]<<")"
		  << "\n\t   has GEN mother pdgId = " << momPdgId[i] << " (status = "<<momStatus[i] << ")";
		
		reco::GenParticleRef genGMom = genMom->numberOfMothers() > 0 ? genMom->motherRef() : reco::GenParticleRef();
	      
		if (genGMom.isNonnull()) {
		  gmomPdgId[i] = genGMom->pdgId();
		  edm::LogVerbatim("MuonMCClassifier") << "\t\t mother prod. vertex rho = " << momRho[i] << ", z = " << momZ[i] 
						       << ", grand-mom pdgId = " << gmomPdgId[i];
		}
		// in this case, we might want to know the heaviest mom flavour
		for (reco::GenParticleRef nMom = genMom;
                     nMom.isNonnull() && std::abs(nMom->pdgId()) >= 100; // stop when we're no longer looking at hadrons or mesons
		     nMom = nMom->numberOfMothers() > 0 ? nMom->motherRef() : reco::GenParticleRef()) {
		  int flav = flavour(nMom->pdgId());
		  if (hmomFlav[i] < flav) hmomFlav[i] = flav;
		  edm::LogVerbatim("MuonMCClassifier") 
		    << "\t\t backtracking flavour: mom pdgId = "<<nMom->pdgId()
		    << ", flavour = " << flav << ", heaviest so far = " << hmomFlav[i];
		}
	      }    //      if (genMom.isNonnull())

	      else {   // mother is null ??
		edm::LogWarning("MuonMCClassifier") 
		  << "\t GenParticle with pdgId = "<<hitsPdgId[i] << ", (Event,Bx) = "<< "(" <<tpEv[i]<<","<<tpBx[i]<<")"
		  <<"\n\t   q*p = "<<signp[i]<<", pT = "<<pt[i]<<", eta = "<<eta[i]<<", phi = "<<phi[i]
		  << "\n\t   produced at vertex rho = " << prodRho[i] << ", z = " << prodZ[i] 
		  << ", (GEANT4 process = "<< G4processType[i]<<")"
		  <<"\n\t   has NO mother!";
	      }
	      
            } else { // Muon is in SIM Only
                TrackingParticleRef simMom = getTpMother(tp);
                if (simMom.isNonnull()) {
                    momPdgId[i] = simMom->pdgId();
                    momRho[i] = simMom->vertex().Rho();
                    momZ[i]   = simMom->vertex().Z();
                    edm::LogVerbatim("MuonMCClassifier") 
		      << "\t Particle pdgId = "<<hitsPdgId[i] << ", (Event,Bx) = "<< "(" <<tpEv[i]<<","<<tpBx[i]<<")"
		      <<"\n\t   q*p = "<<signp[i]<<", pT = "<<pt[i]<<", eta = "<<eta[i]<<", phi = "<<phi[i]
		      << "\n\t   produced at vertex rho = " << prodRho[i] << ", z = " << prodZ[i] 
		      << ", (GEANT4 process = "<< G4processType[i]<<")"
		      <<"\n\t   has SIM mother pdgId = " << momPdgId[i] 
		      << " produced at rho = " << simMom->vertex().Rho() << ", z = " << simMom->vertex().Z();

                    if (!simMom->genParticles().empty()) {
		      momStatus[i] = simMom->genParticles()[0]->status();
		      reco::GenParticleRef genGMom = (simMom->genParticles()[0]->numberOfMothers() > 0 ? simMom->genParticles()[0]->motherRef() : reco::GenParticleRef());
		      if (genGMom.isNonnull()) gmomPdgId[i] = genGMom->pdgId();
		      edm::LogVerbatim("MuonMCClassifier") 
			<< "\t\t SIM mother is in GEN (status " << momStatus[i] << "), grand-mom id = " << gmomPdgId[i];
                    } 
		    else {
		      momStatus[i] = -1;
		      TrackingParticleRef simGMom = getTpMother(simMom);
		      if (simGMom.isNonnull()) gmomPdgId[i] = simGMom->pdgId();
		      edm::LogVerbatim("MuonMCClassifier") << "\t\t SIM mother is in SIM only, grand-mom id = " << gmomPdgId[i];
                    }
                } else {
                  edm::LogVerbatim("MuonMCClassifier") 
		      << "\t Particle pdgId = "<<hitsPdgId[i] << ", (Event,Bx) = "<< "(" <<tpEv[i]<<","<<tpBx[i]<<")"
		      <<"\n\t   q*p = "<<signp[i]<<", pT = "<<pt[i]<<", eta = "<<eta[i]<<", phi = "<<phi[i]
		      << "\n\t   produced at vertex rho = " << prodRho[i] << ", z = " << prodZ[i] 
		      << ", (GEANT4 process = "<< G4processType[i]<<")"
		      <<"\n\t   has NO mother!";
                }
            }
            momFlav[i]  = flavour(momPdgId[i]);
            gmomFlav[i] = flavour(gmomPdgId[i]);

            // Check first IF this is a muon at all
            if (abs(tp->pdgId()) != 13) {
	      if (abs(tp->pdgId()) == 11) {
                classif[i] = isGhost ? -11 : 11;
                ext[i]     = isGhost ? -11 : 11;
                edm::LogVerbatim("MuonMCClassifier") <<"\t This is electron/positron. classif[i] = " << classif[i];
	      }
	      else {
                classif[i] = isGhost ? -1 : 1;
                ext[i]     = isGhost ? -1 : 1;
                edm::LogVerbatim("MuonMCClassifier") <<"\t This is not a muon. Sorry. classif[i] = " << classif[i];
	      }

	      continue;
            }

            // Is this SIM muon also a GEN muon, with a mother?
            if (!tp->genParticles().empty() && (momPdgId[i] != 0)) {
                if (abs(momPdgId[i]) < 100 && (abs(momPdgId[i]) != 15)) {
                    classif[i] = isGhost ? -4 : 4;
		    flav[i] = 13;
                    ext[i]  = 10;
                    edm::LogVerbatim("MuonMCClassifier") <<"\t This seems PRIMARY MUON ! classif[i] = " << classif[i];
                } else if (momFlav[i] == 4 || momFlav[i] == 5 || momFlav[i] == 15) {
                    classif[i] = isGhost ? -3 : 3;
                    flav[i]    = momFlav[i];
                    if (momFlav[i] == 15)      ext[i] = 9; // tau->mu
                    else if (momFlav[i] == 5)  ext[i] = 8; // b->mu
                    else if (hmomFlav[i] == 5) ext[i] = 7; // b->c->mu or b->tau->mu
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
                int id = std::abs(momPdgId[i]);
                if (id != /*pi+*/211 && id != /*K+*/321 && id != 130 /*K0L*/)  ext[i] = 5; // other light particle, possibly short-lived
                else if (prodRho[i] < decayRho_ && std::abs(prodZ[i]) < decayAbsZ_) ext[i] = 4; // decay a la ppMuX (primary pi/K within a cylinder)
                else                                                           ext[i] = 3; // late decay that wouldn't be in ppMuX
            } else ext[i] = 2; // decay of non-primary particle, would not be in ppMuX
            if (isGhost) ext[i] = -ext[i];

            if (linkToGenParticles_ && std::abs(ext[i]) >= 2) {
                // Link to the genParticle if possible, but not decays in flight (in ppMuX they're in GEN block, but they have wrong parameters)
                if (!tp->genParticles().empty() && std::abs(ext[i]) >= 5) {
                    if (genParticles.id() != tp->genParticles().id()) {
                        throw cms::Exception("Configuration") << "Product ID mismatch between the genParticle collection (" << genParticles_ << ", id " << genParticles.id() << ") and the references in the TrackingParticles (id " << tp->genParticles().id() << ")\n";
                    }
                    muToPrimary[i] = tp->genParticles()[0].key();
                } else {
                    // Don't put the same trackingParticle twice!
                    int &indexPlus1 = tpToSecondaries[tp]; // will create a 0 if the tp is not in the list already
                    if (indexPlus1 == 0) indexPlus1 = convertAndPush(*tp, *secondaries, getTpMother(tp), genParticles) + 1;
                    muToSecondary[i] = indexPlus1 - 1;
                }
            }
            edm::LogVerbatim("MuonMCClassifier") <<"\t Extended classification code = " << ext[i];
	}  // if (tp.isNonnull())
    } // end loop on reco muons 

    writeValueMap(iEvent, muons, classif,   "");
    writeValueMap(iEvent, muons, ext,       "ext");
    writeValueMap(iEvent, muons, flav,      "flav");
    writeValueMap(iEvent, muons, tpId,      "tpId");
    writeValueMap(iEvent, muons, tpBx,      "tpBx");
    writeValueMap(iEvent, muons, tpEv,      "tpEv");
    writeValueMap(iEvent, muons, hitsPdgId, "hitsPdgId");
    writeValueMap(iEvent, muons, G4processType, "G4processType");
    writeValueMap(iEvent, muons, momPdgId,  "momPdgId");
    writeValueMap(iEvent, muons, momStatus, "momStatus");
    writeValueMap(iEvent, muons, momFlav,   "momFlav");
    writeValueMap(iEvent, muons, gmomPdgId, "gmomPdgId");
    writeValueMap(iEvent, muons, gmomFlav,  "gmomFlav");
    writeValueMap(iEvent, muons, hmomFlav,  "hmomFlav");
    writeValueMap(iEvent, muons, signp,     "signp");
    writeValueMap(iEvent, muons, pt,        "pt");
    writeValueMap(iEvent, muons, eta,       "eta");
    writeValueMap(iEvent, muons, phi,       "phi");
    writeValueMap(iEvent, muons, prodRho,   "prodRho");
    writeValueMap(iEvent, muons, prodZ,     "prodZ");
    writeValueMap(iEvent, muons, momRho,    "momRho");
    writeValueMap(iEvent, muons, momZ,      "momZ");
    writeValueMap(iEvent, muons, tpAssoQuality, "tpAssoQuality");

    if (linkToGenParticles_) {
        edm::OrphanHandle<reco::GenParticleCollection> secHandle = iEvent.put(std::move(secondaries), "secondaries");
        edm::RefProd<reco::GenParticleCollection> priRP(genParticles);
        edm::RefProd<reco::GenParticleCollection> secRP(secHandle);
        std::unique_ptr<edm::Association<reco::GenParticleCollection> > outPri(new edm::Association<reco::GenParticleCollection>(priRP));
        std::unique_ptr<edm::Association<reco::GenParticleCollection> > outSec(new edm::Association<reco::GenParticleCollection>(secRP));
        edm::Association<reco::GenParticleCollection>::Filler fillPri(*outPri), fillSec(*outSec);
        fillPri.insert(muons, muToPrimary.begin(),   muToPrimary.end());
        fillSec.insert(muons, muToSecondary.begin(), muToSecondary.end());
        fillPri.fill(); fillSec.fill();
        iEvent.put(std::move(outPri), "toPrimaries");
        iEvent.put(std::move(outSec), "toSecondaries");
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
    unique_ptr<ValueMap<T> > valMap(new ValueMap<T>());
    typename edm::ValueMap<T>::Filler filler(*valMap);
    filler.insert(handle, values.begin(), values.end());
    filler.fill();
    iEvent.put(std::move(valMap), label);
}

int
MuonMCClassifier::flavour(int pdgId) const {
    int flav = std::abs(pdgId);
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

// push secondary in collection.
// if it has a primary mother link to it.
int MuonMCClassifier::convertAndPush(const TrackingParticle &tp,
                                     reco::GenParticleCollection &out,
                                     const TrackingParticleRef & simMom,
                                     const edm::Handle<reco::GenParticleCollection> & genParticles) const {
    out.push_back(reco::GenParticle(tp.charge(), tp.p4(), tp.vertex(), tp.pdgId(), tp.status(), true));
    if (simMom.isNonnull() && !simMom->genParticles().empty()) {
         if (genParticles.id() != simMom->genParticles().id()) {
            throw cms::Exception("Configuration") << "Product ID mismatch between the genParticle collection (" << genParticles_ << ", id " << genParticles.id() << ") and the references in the TrackingParticles (id " << simMom->genParticles().id() << ")\n";
         }
         out.back().addMother(simMom->genParticles()[0]);
    }
    return out.size()-1;
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonMCClassifier);
