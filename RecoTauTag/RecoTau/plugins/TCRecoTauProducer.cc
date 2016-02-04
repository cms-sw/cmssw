/* class CaloRecoTauProducer
 * EDProducer of the TCTauCollection, starting from the CaloRecoTauCollection,
 * authors: Sami Lehti (sami.lehti@cern.ch)
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTauTag/RecoTau/interface/TCTauAlgorithm.h"

using namespace edm;
using namespace reco;

class TCRecoTauProducer : public edm::EDProducer {
    public:
        explicit TCRecoTauProducer(const edm::ParameterSet& iConfig);
        ~TCRecoTauProducer();

        virtual void produce(edm::Event&,const edm::EventSetup&);

    private:
        edm::InputTag    caloRecoTauProducer;
        TCTauAlgorithm*  tcTauAlgorithm;
};

TCRecoTauProducer::TCRecoTauProducer(const edm::ParameterSet& iConfig){
  	caloRecoTauProducer = iConfig.getParameter<edm::InputTag>("CaloRecoTauProducer");
	tcTauAlgorithm = new TCTauAlgorithm(iConfig);

  	produces<CaloTauCollection>();
}
TCRecoTauProducer::~TCRecoTauProducer(){
  	delete tcTauAlgorithm;
}
  
void TCRecoTauProducer::produce(edm::Event& iEvent,const edm::EventSetup& iSetup){

  	std::auto_ptr<CaloTauCollection> tcTauCollection(new CaloTauCollection);

	edm::Handle<CaloTauCollection> theCaloTauHandle;
	iEvent.getByLabel(caloRecoTauProducer,theCaloTauHandle);

	tcTauAlgorithm->eventSetup(iEvent,iSetup);

        if(theCaloTauHandle.isValid()){
          const CaloTauCollection & caloTaus = *(theCaloTauHandle.product());
          CaloTauCollection::const_iterator iTau;
          for(iTau = caloTaus.begin(); iTau != caloTaus.end(); iTau++){
		CaloTau theTCTau = *iTau;
		theTCTau.setP4(tcTauAlgorithm->recalculateEnergy(theTCTau));
		tcTauCollection->push_back(theTCTau);
	  }
	}

   	iEvent.put(tcTauCollection);
}

DEFINE_FWK_MODULE(TCRecoTauProducer);
