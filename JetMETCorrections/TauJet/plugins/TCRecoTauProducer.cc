#include "JetMETCorrections/TauJet/interface/TCRecoTauProducer.h"

using namespace edm;
using namespace reco;

TCRecoTauProducer::TCRecoTauProducer(const ParameterSet& iConfig){
  	caloRecoTauProducer = iConfig.getParameter<InputTag>("CaloRecoTauProducer");
	tcTauCorrector = new TCTauCorrector(iConfig);

  	produces<CaloTauCollection>();
}
TCRecoTauProducer::~TCRecoTauProducer(){
  	delete tcTauCorrector;
}
  
void TCRecoTauProducer::produce(Event& iEvent,const EventSetup& iSetup){

  	auto_ptr<CaloTauCollection> tcTauCollection(new CaloTauCollection);

	Handle<CaloTauCollection> theCaloTauHandle;
	iEvent.getByLabel(caloRecoTauProducer,theCaloTauHandle);

	tcTauCorrector->eventSetup(iEvent,iSetup);

        if(theCaloTauHandle.isValid()){
          const CaloTauCollection & caloTaus = *(theCaloTauHandle.product());
          CaloTauCollection::const_iterator iTau;
          for(iTau = caloTaus.begin(); iTau != caloTaus.end(); iTau++){
		CaloTau theTCTau = *iTau;
		theTCTau.setP4(tcTauCorrector->correctedP4(theTCTau));
		//if(theTCTau.pt() > 0) 
		tcTauCollection->push_back(theTCTau);
	  }
	}

   	iEvent.put(tcTauCollection);
}

DEFINE_FWK_MODULE(TCRecoTauProducer);
