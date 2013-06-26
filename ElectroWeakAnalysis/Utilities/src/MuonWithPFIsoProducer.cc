#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "MuonAnalysis/MomentumScaleCalibration/interface/MomentumScaleCorrector.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/ResolutionFunction.h"

//
// class declaration
//
class MuonWithPFIsoProducer : public edm::EDProducer {
   public:
      explicit MuonWithPFIsoProducer(const edm::ParameterSet&);
      ~MuonWithPFIsoProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;

      edm::InputTag muonTag_;
      edm::InputTag pfTag_;

      bool usePfMuonsOnly_;

      double trackIsoVeto_;
      double gammaIsoVeto_;
      double neutralHadronIsoVeto_;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/GeometryVector/interface/VectorUtil.h"

/////////////////////////////////////////////////////////////////////////////////////
MuonWithPFIsoProducer::MuonWithPFIsoProducer(const edm::ParameterSet& pset) {

  // What is being produced
      produces<std::vector<reco::Muon> >();

  // Muon collection
      muonTag_ = pset.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"));

  // PF candidate collection
      pfTag_ = pset.getUntrackedParameter<edm::InputTag> ("PFTag", edm::InputTag("particleFlow"));

  // Use only PF muons to get exact consistency with PfMET
      usePfMuonsOnly_ = pset.getUntrackedParameter<bool> ("UsePfMuonsOnly", false);

  // Veto cone
      trackIsoVeto_ = pset.getUntrackedParameter<double> ("TrackIsoVeto", 0.01);
      gammaIsoVeto_ = pset.getUntrackedParameter<double> ("GammaIsoVeto", 0.07);
      neutralHadronIsoVeto_ = pset.getUntrackedParameter<double> ("NeutralHadronIsoVeto", 0.1);

} 

/////////////////////////////////////////////////////////////////////////////////////
MuonWithPFIsoProducer::~MuonWithPFIsoProducer(){
}

/////////////////////////////////////////////////////////////////////////////////////
void MuonWithPFIsoProducer::beginJob() {
}

/////////////////////////////////////////////////////////////////////////////////////
void MuonWithPFIsoProducer::endJob(){
}

/////////////////////////////////////////////////////////////////////////////////////
void MuonWithPFIsoProducer::produce(edm::Event& ev, const edm::EventSetup& iSetup) {

      // Initialize pointer to new output muon collection
      std::auto_ptr<reco::MuonCollection> newmuons (new reco::MuonCollection);

      // Get Muon collection
      edm::Handle<edm::View<reco::Muon> > muonCollection;
      if (!ev.getByLabel(muonTag_, muonCollection)) {
            edm::LogError("") << ">>> Muon collection does not exist !!!";
            ev.put(newmuons);
            return;
      }

      // Get PFCandidate collection
      edm::Handle<edm::View<reco::PFCandidate> > pfCollection;
      if (!ev.getByLabel(pfTag_, pfCollection)) {
            edm::LogError("") << ">>> PFCandidate collection does not exist !!!";
            ev.put(newmuons);
            return;
      }

      // Loop over Pf candidates to find muons and collect deposits in veto, 
      // dR<0.3 and dR<0.5 cones. Interpret "track" as charged particles (e,mu,
      // chraged hadrons). Interpret "em" as photons and also as electromagnetic 
      // energy in HF. Interpret "had" as neutral hadrons and also as hadronic
      // energy in HF. Apply weights if requested at input level.
      // HO energies are not filled. Ditto for jet energies around the muon.
      unsigned int muonCollectionSize = muonCollection->size();
      unsigned int pfCollectionSize = pfCollection->size();
      for (unsigned int i=0; i<muonCollectionSize; i++) {
            edm::RefToBase<reco::Muon> mu = muonCollection->refAt(i);

            // Ask for PfMuon consistency if requested
            bool muonFound = false;

            // Starting bycloning this muon
            reco::Muon* newmu = mu->clone();
            reco::TrackRef tk = mu->innerTrack();

            // Set isolations
            reco::MuonIsolation iso03;
            reco::MuonIsolation iso05;
            // Loop on all candidates
            for (unsigned int j=0; j<pfCollectionSize; j++) {
                  edm::RefToBase<reco::PFCandidate> pf = pfCollection->refAt(j);

                  // Check the muon is in the PF collection when required
                  bool thisIsTheMuon = false;
                  if (tk.isNonnull() && pf->trackRef()==tk) {
                        thisIsTheMuon = true;
                        muonFound = true;
                  }
                         
                  // Get dR. Nothing to add if dR>0.5
                  double deltaR = Geom::deltaR(mu->momentum(),pf->momentum());
                  if (deltaR>0.5) continue;

                  // Fill "tracker" components
                  if (   pf->particleId()==reco::PFCandidate::h
                      || pf->particleId()==reco::PFCandidate::e
                      || pf->particleId()==reco::PFCandidate::mu ) {
                        if (deltaR<trackIsoVeto_ || thisIsTheMuon) {
                              iso05.trackerVetoPt += pf->pt();
                              iso03.trackerVetoPt += pf->pt();
                        } else {
                              iso05.sumPt += pf->pt();
                              iso05.nTracks++;
                              if (deltaR<0.3) {
                                    iso03.sumPt += pf->pt();
                                    iso03.nTracks++;
                              }
                        }
                  // Fill "em" components
                  } else if (   pf->particleId()==reco::PFCandidate::gamma 
                             || pf->particleId()==reco::PFCandidate::egamma_HF) {
                        if (deltaR<gammaIsoVeto_) {
                              iso05.emVetoEt += pf->pt();
                              iso03.emVetoEt += pf->pt();
                        } else {
                              iso05.emEt += pf->pt();
                              if (deltaR<0.3) iso03.emEt += pf->pt();
                        }
                  // Fill "had" components
                  } else if (   pf->particleId()==reco::PFCandidate::h0
                             || pf->particleId()==reco::PFCandidate::h_HF) {
                        if (deltaR<neutralHadronIsoVeto_) {
                              iso05.hadVetoEt += pf->pt();
                              iso03.hadVetoEt += pf->pt();
                        } else {
                              iso05.hadEt += pf->pt();
                              if (deltaR<0.3) iso03.hadEt += pf->pt();
                        }
                  }
            }

            // Do not take this muon (under explicit request) if it is not a PfMuon
            if (usePfMuonsOnly_ && (!muonFound)) continue;

            // Set this isolation information in the new muon
            newmu->setIsolation(iso03,iso05);
            
            // Add new muon to output collection
            newmuons->push_back(*newmu);

      }

      // Add output collection to event
      ev.put(newmuons);
}

DEFINE_FWK_MODULE(MuonWithPFIsoProducer);
