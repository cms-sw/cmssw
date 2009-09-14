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

#include "JetMETCorrections/TauJet/interface/TCTauCorrector.h"
#include "JetMETCorrections/TauJet/interface/TauJetCorrector.h"

using namespace edm;
using namespace reco;

class TCRecoTauProducer : public edm::EDProducer {
    public:
  	explicit TCRecoTauProducer(const edm::ParameterSet& iConfig);
  	~TCRecoTauProducer(){}
  	virtual void produce(edm::Event&,const edm::EventSetup&);
    private:
  	edm::InputTag 	 caloRecoTauProducer;
        auto_ptr<TCTauCorrector>  tcTauCorrector;
        auto_ptr<TauJetCorrector> tauJetCorrector;
};

TCRecoTauProducer::TCRecoTauProducer(const ParameterSet& iConfig):tcTauCorrector(new TCTauCorrector(iConfig)),
                                                                  tauJetCorrector(new TauJetCorrector(iConfig))
{
   caloRecoTauProducer = iConfig.getParameter<InputTag>("CaloRecoTauProducer");
   produces<CaloTauCollection>();
}
  
void TCRecoTauProducer::produce(Event& iEvent,const EventSetup& iSetup){

   auto_ptr<CaloTauCollection> resultCaloTau(new CaloTauCollection);

   Handle<CaloTauCollection> theCaloTauHandle;
   iEvent.getByLabel(caloRecoTauProducer,theCaloTauHandle);

   tcTauCorrector->eventSetup(iEvent,iSetup);

   if(theCaloTauHandle.isValid()){
      const CaloTauCollection & caloTaus = *(theCaloTauHandle.product());
      CaloTauCollection::const_iterator iTau;
      for(iTau = caloTaus.begin(); iTau != caloTaus.end(); ++iTau)
      {
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

DEFINE_FWK_MODULE(TCRecoTauProducer);
