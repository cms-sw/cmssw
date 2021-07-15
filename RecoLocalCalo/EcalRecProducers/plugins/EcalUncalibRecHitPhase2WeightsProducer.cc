#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"

#include "DataFormats/EcalDigi/interface/EcalConstants.h"

class EcalUncalibRecHitPhase2WeightsProducer: public edm::stream::EDProducer<>{


public:
  explicit EcalUncalibRecHitPhase2WeightsProducer(const edm::ParameterSet& ps);//, edm::ConsumesCollector& c);
  void produce(edm::Event& evt, const edm::EventSetup& es) override;

private:
  
  float tRise_;
  float tFall_;
  std::vector<double> weights_;


  edm::EDGetTokenT<EBDigiCollectionPh2> ebDigiCollectionToken_;
  std::string hitCollection_;

};

EcalUncalibRecHitPhase2WeightsProducer::EcalUncalibRecHitPhase2WeightsProducer(const edm::ParameterSet& ps){//, edm::ConsumesCollector& c){
    
  hitCollection_ = ps.getParameter<std::string>("EBhitCollection");
  produces<EBUncalibratedRecHitCollection>(hitCollection_);
  
  tRise_ = ps.getParameter<double>("tRise");
  tFall_ = ps.getParameter<double>("tFall");
  
  ebDigiCollectionToken_ = consumes<EBDigiCollectionPh2>(ps.getParameter<edm::InputTag>("BarrelDigis"));

  weights_ = ps.getParameter< std::vector<double> >("weights");
}

void EcalUncalibRecHitPhase2WeightsProducer::produce(edm::Event& evt, const edm::EventSetup& es){
  
  // retrieve digis
  edm::Handle<EBDigiCollectionPh2> pdigis_;
  
  evt.getByToken(ebDigiCollectionToken_, pdigis_);
  
  const EBDigiCollectionPh2* pdigis = nullptr;
  pdigis = pdigis_.product();

  // prepare output
  auto ebUncalibRechits = std::make_unique<EBUncalibratedRecHitCollection>();
  
  for (auto itdg = pdigis->begin(); itdg != pdigis->end(); ++itdg) {
    
    EBDataFrame digi(*itdg);
    EcalDataFrame_Ph2 dataFrame(*itdg);
    DetId detId(digi.id());
    
    bool g1 = false; 

    std::vector<float> timetrace;
    std::vector<float> adctrace;

    int nSamples = digi.size();

    timetrace.reserve(nSamples);
    adctrace.reserve(nSamples);
    
    float amp = 0;
    float gratio;
    
    for (int sample = 0; sample < nSamples; ++sample){
      
      EcalLiteDTUSample thisSample = dataFrame[sample];
      gratio = ecalPh2::gains[thisSample.gainId()];
      adctrace.push_back(thisSample.adc()*gratio);

      amp = amp + adctrace[sample]*weights_[sample];
      
      if (thisSample.gainId()==1) g1=true;
      
      timetrace.push_back(sample);
      
    }// loop on samples
     
    float amp_e= 1;
    float t0_e = 0;
  
    EcalUncalibratedRecHit rhit(detId, amp, 0., 0., 0., 0); // rhit(detIt, amp, pedestal, t0, chi2, flags)
    rhit.setAmplitudeError(amp_e);
    rhit.setJitterError(t0_e);
    if (g1) rhit.setFlagBit(EcalUncalibratedRecHit::kHasSwitchToGain1); 
    
    ebUncalibRechits->push_back(rhit);
    
  }// loop on digis    
  
  
  evt.put(std::move(ebUncalibRechits), hitCollection_);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EcalUncalibRecHitPhase2WeightsProducer);