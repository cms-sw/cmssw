#include "RecoJets/JetProducers/plugins/FixedGridRhoProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace std;

FixedGridRhoProducer::FixedGridRhoProducer(const edm::ParameterSet& iConfig) {
  pfCandidatesTag_ = iConfig.getParameter<edm::InputTag>("pfCandidatesTag");
  string etaRegion = iConfig.getParameter<string>("EtaRegion");
  if (etaRegion=="Central") myEtaRegion = FixedGridEnergyDensity::Central;
  else if (etaRegion=="Forward") myEtaRegion = FixedGridEnergyDensity::Forward;
  else if (etaRegion=="All") myEtaRegion = FixedGridEnergyDensity::All;
  else {
    edm::LogWarning("FixedGridRhoProducer") << "Wrong EtaRegion parameter: " << etaRegion << ". Using EtaRegion = Central";  
    myEtaRegion = FixedGridEnergyDensity::Central;
  }
  produces<double>();

  input_pfcoll_token_ = consumes<reco::PFCandidateCollection>(pfCandidatesTag_);

}

FixedGridRhoProducer::~FixedGridRhoProducer(){} 

void FixedGridRhoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

   edm::Handle<reco::PFCandidateCollection> pfColl;
   iEvent.getByToken(input_pfcoll_token_, pfColl);

   algo = new FixedGridEnergyDensity(pfColl.product());

   double result = algo->fixedGridRho(myEtaRegion);
   std::auto_ptr<double> output(new double(result));
   iEvent.put(output);

   delete algo;
 
}

DEFINE_FWK_MODULE(FixedGridRhoProducer);
