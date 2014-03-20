#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/METReco/interface/MET.h"

//
// class declaration
//
class DistortedMETProducer : public edm::EDProducer {
   public:
      explicit DistortedMETProducer(const edm::ParameterSet&);
      ~DistortedMETProducer();

   private:
      virtual void beginJob() override ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;

      edm::EDGetTokenT<edm::View<reco::MET> > metToken_;
      double metScaleShift_; // relative shift (0. => no shift)
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/METReco/interface/METFwd.h"

/////////////////////////////////////////////////////////////////////////////////////
DistortedMETProducer::DistortedMETProducer(const edm::ParameterSet& pset) {

  // What is being produced
      produces<std::vector<reco::MET> >();

  // Input products
      metToken_ = consumes<edm::View<reco::MET> >(pset.getUntrackedParameter<edm::InputTag> ("MetTag", edm::InputTag("met")));
  // Distortions in MET in Gev**{-1/2}
      metScaleShift_ = pset.getUntrackedParameter<double> ("MetScaleShift",1.e-3);

}

/////////////////////////////////////////////////////////////////////////////////////
DistortedMETProducer::~DistortedMETProducer(){
}

/////////////////////////////////////////////////////////////////////////////////////
void DistortedMETProducer::beginJob() {
}

/////////////////////////////////////////////////////////////////////////////////////
void DistortedMETProducer::endJob(){}

/////////////////////////////////////////////////////////////////////////////////////
void DistortedMETProducer::produce(edm::Event& ev, const edm::EventSetup&) {

      if (ev.isRealData()) return;

      // MET collection
      edm::Handle<edm::View<reco::MET> > metCollection;
      if (!ev.getByToken(metToken_, metCollection)) {
            edm::LogError("") << ">>> MET collection does not exist !!!";
            return;
      }
      edm::RefToBase<reco::MET> met = metCollection->refAt(0);

      std::auto_ptr<reco::METCollection> newmetCollection (new reco::METCollection);

      double met_et = met->et() * (1. + metScaleShift_);
      double sum_et = met->sumEt() * (1. + metScaleShift_);
      double met_phi = met->phi();
      double met_ex = met_et*cos(met_phi);
      double met_ey = met_et*sin(met_phi);
      reco::Particle::LorentzVector met_p4(met_ex, met_ey, 0., met_et);
      reco::Particle::Point met_vtx(0.,0.,0.);
      reco::MET* newmet = new reco::MET(sum_et, met_p4, met_vtx);

      newmetCollection->push_back(*newmet);

      ev.put(newmetCollection);
}

DEFINE_FWK_MODULE(DistortedMETProducer);
