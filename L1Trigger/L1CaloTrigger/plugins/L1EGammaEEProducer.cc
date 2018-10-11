#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "L1Trigger/L1CaloTrigger/interface/L1EGammaEECalibrator.h"


class L1EGammaEEProducer : public edm::EDProducer {
   public:
      explicit L1EGammaEEProducer(const edm::ParameterSet&);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);

      edm::EDGetToken multiclusters_token_;
      L1EGammaEECalibrator calibrator_;
};

L1EGammaEEProducer::L1EGammaEEProducer(const edm::ParameterSet& iConfig) :
  multiclusters_token_(consumes<l1t::HGCalMulticlusterBxCollection>(iConfig.getParameter<edm::InputTag>("Multiclusters"))),
  calibrator_(iConfig.getParameter<edm::ParameterSet>("calibrationConfig")) {
    produces< BXVector<l1t::EGamma> >("L1EGammaCollectionBXVWithCuts");
}


void L1EGammaEEProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  float minEt_ = 0;

  std::unique_ptr< BXVector<l1t::EGamma> > l1EgammaBxCollection(new l1t::EGammaBxCollection);


  // retrieve clusters 3D
  edm::Handle<l1t::HGCalMulticlusterBxCollection> multiclusters_h;
  iEvent.getByToken(multiclusters_token_, multiclusters_h);
  const l1t::HGCalMulticlusterBxCollection& multiclusters = *multiclusters_h;

  // here we loop on the TPGs
  for(auto cl3d = multiclusters.begin(0); cl3d != multiclusters.end(0); cl3d++) {
     // std::cout << "-- CL3D is EG: " <<  cl3d->hwQual() << "   "<< cl3d->eta() <<"   "<< cl3d->pt()<<std::endl;
     if(cl3d->hwQual()) {
       if(cl3d->et() > minEt_) {
         int hw_quality = 1; // baseline EG ID passed
         if(fabs(cl3d->eta()) >= 1.52) {
           hw_quality = 2; // baseline EG ID passed + cleanup of transition region
         }

         float calib_factor = calibrator_.calibrationFactor(cl3d->pt(), cl3d->eta());
         l1t::EGamma eg=l1t::EGamma(reco::Candidate::PolarLorentzVector(cl3d->pt()/calib_factor, cl3d->eta(), cl3d->phi(), 0.));
         eg.setHwQual(hw_quality);
         eg.setHwIso(1);
         l1EgammaBxCollection->push_back(0,eg);
      }
    }
  }

  iEvent.put(std::move(l1EgammaBxCollection),"L1EGammaCollectionBXVWithCuts");
}



DEFINE_FWK_MODULE(L1EGammaEEProducer);
