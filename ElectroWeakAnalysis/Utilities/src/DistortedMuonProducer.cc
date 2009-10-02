#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"

//
// class declaration
//
class DistortedMuonProducer : public edm::EDProducer {
   public:
      explicit DistortedMuonProducer(const edm::ParameterSet&);
      ~DistortedMuonProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      edm::InputTag muonTag_;
      edm::InputTag genMatchMapTag_;
      std::vector<double> etaBinEdges_;
      std::vector<double> momentumScaleShift_;
      std::vector<double> uncertaintyOnOneOverPt_; // in [1/GeV]
      std::vector<double> relativeUncertaintyOnPt_;
      std::vector<double> efficiencyRatioOverMC_;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <CLHEP/Random/RandFlat.h>
#include <CLHEP/Random/RandGauss.h>

/////////////////////////////////////////////////////////////////////////////////////
DistortedMuonProducer::DistortedMuonProducer(const edm::ParameterSet& pset) {

  // What is being produced
      produces<std::vector<reco::Track> >();
      produces<std::vector<reco::Muon> >();

  // Input products
      muonTag_ = pset.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"));
      genMatchMapTag_ = pset.getUntrackedParameter<edm::InputTag> ("GenMatchMapTag", edm::InputTag("genMatchMap"));

  // Eta edges
      std::vector<double> defEtaEdges;
      defEtaEdges.push_back(-999999.);
      defEtaEdges.push_back(999999.);
      etaBinEdges_ = pset.getUntrackedParameter<std::vector<double> > ("EtaBinEdges",defEtaEdges);
      unsigned int ninputs_expected = etaBinEdges_.size()-1;

  // Distortions in muon momentum
      std::vector<double> defDistortion;
      defDistortion.push_back(0.);

      momentumScaleShift_ = pset.getUntrackedParameter<std::vector<double> > ("MomentumScaleShift",defDistortion);
      if (momentumScaleShift_.size()==1 && ninputs_expected>1) {
            for (unsigned int i=1; i<ninputs_expected; i++){ momentumScaleShift_.push_back(momentumScaleShift_[0]);}
      }

      uncertaintyOnOneOverPt_ = pset.getUntrackedParameter<std::vector<double> > ("UnertaintyOnOneOverPt",defDistortion); // in [1/GeV]
      if (uncertaintyOnOneOverPt_.size()==1 && ninputs_expected>1) {
            for (unsigned int i=1; i<ninputs_expected; i++){ uncertaintyOnOneOverPt_.push_back(uncertaintyOnOneOverPt_[0]);}
      }

      relativeUncertaintyOnPt_ = pset.getUntrackedParameter<std::vector<double> > ("RelativeUncertaintyOnPt",defDistortion);
      if (relativeUncertaintyOnPt_.size()==1 && ninputs_expected>1) {
            for (unsigned int i=1; i<ninputs_expected; i++){ relativeUncertaintyOnPt_.push_back(relativeUncertaintyOnPt_[0]);}
      }

  // Data/MC efficiency ratios
      std::vector<double> defEfficiencyRatio;
      defEfficiencyRatio.push_back(1.);
      efficiencyRatioOverMC_ = pset.getUntrackedParameter<std::vector<double> > ("EfficiencyRatioOverMC",defEfficiencyRatio);
      if (efficiencyRatioOverMC_.size()==1 && ninputs_expected>1) {
            for (unsigned int i=1; i<ninputs_expected; i++){ efficiencyRatioOverMC_.push_back(efficiencyRatioOverMC_[0]);}
      }

  // Send a warning if there are inconsistencies in vector sizes !!
      if (    momentumScaleShift_.size() != ninputs_expected
           || uncertaintyOnOneOverPt_.size() != ninputs_expected
           || relativeUncertaintyOnPt_.size() != ninputs_expected
           || efficiencyRatioOverMC_.size() != ninputs_expected
         ) {
           edm::LogError("") << "WARNING: DistortedMuonProducer : Size of some parameters do not match the EtaBinEdges vector!!";
      }

} 

/////////////////////////////////////////////////////////////////////////////////////
DistortedMuonProducer::~DistortedMuonProducer(){
}

/////////////////////////////////////////////////////////////////////////////////////
void DistortedMuonProducer::beginJob(const edm::EventSetup&) {
}

/////////////////////////////////////////////////////////////////////////////////////
void DistortedMuonProducer::endJob(){}

/////////////////////////////////////////////////////////////////////////////////////
void DistortedMuonProducer::produce(edm::Event& ev, const edm::EventSetup&) {

      if (ev.isRealData()) return;

      // Muon collection
      edm::Handle<edm::View<reco::Muon> > muonCollection;
      if (!ev.getByLabel(muonTag_, muonCollection)) {
            edm::LogError("") << ">>> Muon collection does not exist !!!";
            return;
      }

      edm::Handle<reco::GenParticleMatch> genMatchMap;
      if (!ev.getByLabel(genMatchMapTag_, genMatchMap)) {
            edm::LogError("") << ">>> Muon-GenParticle match map does not exist !!!";
            return;
      }
  
      unsigned int muonCollectionSize = muonCollection->size();

      std::auto_ptr<reco::MuonCollection> newmuons (new reco::MuonCollection);
      std::auto_ptr<reco::TrackCollection> newtracks (new reco::TrackCollection);
      reco::TrackRefProd trackRefProd = ev.getRefBeforePut<reco::TrackCollection>();

      for (unsigned int i=0; i<muonCollectionSize; i++) {
            // With "View<Muon>": one can use a "RefToBase<Muon>" instead of a "MuonRef"
            edm::RefToBase<reco::Muon> mu = muonCollection->refAt(i);
            // To get a a true "MuonRef" out of it:
            //    reco::MuonRef mu_trueref = mu.castTo<reco::MuonRef>(); 
            if (mu->innerTrack().isNull()) continue;
            reco::TrackRef tk = mu->innerTrack();

            double ptgen = mu->pt();
            reco::GenParticleRef gen = (*genMatchMap)[mu];
            if( !gen.isNull()) {
                  ptgen = gen->pt();
                  LogTrace("") << ">>> Muon-GenParticle match found; ptmu= " << mu->pt() << ", ptgen= " << ptgen;
            } else {
                  LogTrace("") << ">>> MUON-GENPARTICLE MATCH NOT FOUND!!!";
            }

            // Find out which eta bin should be used
            double eta = mu->eta();
            double eff = 0.; // Reject any muon outside [mineta,maxeta]
            double shift = 0.;
            double sigma1 = 0.;
            double sigma2 = 0.;
            unsigned int nbins = etaBinEdges_.size()-1;
            if (eta>etaBinEdges_[0] && eta<etaBinEdges_[nbins]) {
                  for (unsigned int j=1; j<=nbins; ++j) {
                        if (eta>etaBinEdges_[j]) continue;
                        eff = efficiencyRatioOverMC_[j-1];
                        shift = momentumScaleShift_[j-1];
                        sigma1 = uncertaintyOnOneOverPt_[j-1];
                        sigma2 = relativeUncertaintyOnPt_[j-1];
                        break;
                  }
            }

            // Reject muons according to efficiency ratio
            double rndf = CLHEP::RandFlat::shoot();
            if (rndf>eff) continue;

            // Gaussian Random numbers for smearing
            double rndg1 = CLHEP::RandGauss::shoot();
            double rndg2 = CLHEP::RandGauss::shoot();
            
            // New track
            double pttk = tk->pt();
            pttk += ptgen * ( shift + sigma1*rndg1*ptgen + sigma2*rndg2);
            double pxtk = pttk*tk->px()/tk->pt();
            double pytk = pttk*tk->py()/tk->pt();
            double pztk = tk->pz();
            reco::TrackBase::Vector tkmom(pxtk,pytk,pztk);
            reco::Track* newtk = new reco::Track(tk->chi2(), tk->ndof(), tk->referencePoint(), tkmom, tk->charge(), tk->covariance());
            newtk->setExtra(tk->extra());
            newtk->setHitPattern(tk->extra()->recHits());
            newtracks->push_back(*newtk);

            // New muon
            double ptmu = mu->pt();
            ptmu += ptgen * ( shift + sigma1*rndg1*ptgen + sigma2*rndg2);
            reco::Muon* newmu = mu->clone();
            newmu->setP4 (
                  reco::Particle::PolarLorentzVector (
                        ptmu, mu->eta(), mu->phi(), mu->mass()
                  )
            );
            newmu->setInnerTrack(reco::TrackRef(trackRefProd,newtracks->size()-1));
            newmuons->push_back(*newmu);

      }

      ev.put(newtracks);
      ev.put(newmuons);
}

DEFINE_FWK_MODULE(DistortedMuonProducer);
