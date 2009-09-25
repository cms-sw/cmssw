#include "JetMETCorrections/TauJet/interface/TCRecoTauProducer.h"

using namespace edm;
using namespace reco;

TCRecoTauProducer::TCRecoTauProducer(const ParameterSet& iConfig){
  	caloRecoTauProducer = iConfig.getParameter<InputTag>("CaloRecoTauProducer");
	tcTauCorrector = new TCTauCorrector(iConfig);
	tauJetCorrector = new TauJetCorrector(iConfig);

  	produces<CaloTauCollection>();
}
TCRecoTauProducer::~TCRecoTauProducer(){
  	delete tcTauCorrector;
}
  
void TCRecoTauProducer::produce(Event& iEvent,const EventSetup& iSetup){
std::cout << "check TCRecoTauProducer::produce " << std::endl;
  	auto_ptr<CaloTauCollection> resultCaloTau(new CaloTauCollection);

	Handle<CaloTauCollection> theCaloTauHandle;
	iEvent.getByLabel(caloRecoTauProducer,theCaloTauHandle);

	tcTauCorrector->eventSetup(iEvent,iSetup);

        if(theCaloTauHandle.isValid()){
          const CaloTauCollection & caloTaus = *(theCaloTauHandle.product());
          CaloTauCollection::const_iterator iTau;
          for(iTau = caloTaus.begin(); iTau != caloTaus.end(); iTau++){
		CaloTau theCaloTau = *iTau;
		double tauJetCorrection = tauJetCorrector->correction(iTau->p4());
                theCaloTau.setP4(iTau->p4()*tauJetCorrection);

		double tcTauCorrection = tcTauCorrector->correction(theCaloTau);
		theCaloTau.setP4(theCaloTau.p4()*tcTauCorrection);
		resultCaloTau->push_back(theCaloTau);
	  }
	}

   	iEvent.put(resultCaloTau);
}
