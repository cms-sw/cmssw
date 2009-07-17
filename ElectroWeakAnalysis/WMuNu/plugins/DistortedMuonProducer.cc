#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <CLHEP/Random/RandEngine.h>
#include <CLHEP/Random/RandFlat.h>
#include <CLHEP/Random/RandGauss.h>

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
      double momentumScaleShift_;
      double uncertaintyOnOneOverPt_; // in [1/GeV]
      double relativeUncertaintyOnPt_;

      CLHEP::RandEngine* fRandEngine;
      CLHEP::RandFlat* fRandFlat;
      CLHEP::RandGauss* fRandGauss;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

/////////////////////////////////////////////////////////////////////////////////////
DistortedMuonProducer::DistortedMuonProducer(const edm::ParameterSet& pset) :
 muonTag_(pset.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"))),
 momentumScaleShift_(pset.getUntrackedParameter<double> ("MomentumScaleShift", 1.e-3)),
 uncertaintyOnOneOverPt_(pset.getUntrackedParameter<double> ("UnertaintyOnOneOverPt", 2.e-4)), // in [1/GeV]
 relativeUncertaintyOnPt_(pset.getUntrackedParameter<double> ("RelativeUncertaintyOnPt", 1.e-3))
{
      produces<std::vector<reco::Track> >();
      produces<std::vector<reco::Muon> >();
      fRandEngine = new CLHEP::RandEngine(123456789);
      fRandFlat = new CLHEP::RandFlat(fRandEngine);
      fRandGauss = new CLHEP::RandGauss(fRandEngine);
} 

/////////////////////////////////////////////////////////////////////////////////////
DistortedMuonProducer::~DistortedMuonProducer(){
      //delete fRandEngine;
      //delete fRandFlat;
      //delete fRandGauss;
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
      unsigned int muonCollectionSize = muonCollection->size();

      std::auto_ptr<reco::MuonCollection> newmuons (new reco::MuonCollection);
      std::auto_ptr<reco::TrackCollection> newtracks (new reco::TrackCollection);
      reco::TrackRefProd trackRefProd = ev.getRefBeforePut<reco::TrackCollection>();

      int muindex = 0;
      for (unsigned int i=0; i<muonCollectionSize; i++) {
            const reco::Muon& mu = muonCollection->at(i);
            if (mu.innerTrack().isNull()) continue;
            reco::TrackRef tk = mu.innerTrack();

            double rndg1 = fRandGauss->shoot();
            double rndg2 = fRandGauss->shoot();

            // New track
            double pttk = tk->pt();
            // Next line must be modified: it should use ptmu_gen instead
            pttk += pttk * (  momentumScaleShift_
                            + uncertaintyOnOneOverPt_ * rndg1*pttk
                            + relativeUncertaintyOnPt_ * rndg2);
            double pxtk = pttk*tk->px()/tk->pt();
            double pytk = pttk*tk->py()/tk->pt();
            double pztk = tk->pz();
            reco::TrackBase::Vector tkmom(pxtk,pytk,pztk);
            reco::Track* newtk = new reco::Track(tk->chi2(), tk->ndof(), tk->referencePoint(), tkmom, tk->charge(), tk->covariance());
            newtk->setExtra(tk->extra());
            newtk->setHitPattern(tk->extra()->recHits());
            newtracks->push_back(*newtk);

            // New muon
            double ptmu = mu.pt();
            // Next line must be modified: it should use ptmu_gen instead
            ptmu += ptmu * (  momentumScaleShift_
                            + uncertaintyOnOneOverPt_ * rndg1*ptmu
                            + relativeUncertaintyOnPt_ * rndg2);
            reco::Muon* newmu = mu.clone();
            newmu->setP4 (
                  reco::Particle::PolarLorentzVector (
                        ptmu, mu.eta(), mu.phi(), mu.mass()
                  )
            );
            newmu->setInnerTrack(reco::TrackRef(trackRefProd,newtracks->size()-1));
            newmuons->push_back(*newmu);

            muindex++;

      }

      ev.put(newtracks);
      ev.put(newmuons);
}

DEFINE_FWK_MODULE(DistortedMuonProducer);
