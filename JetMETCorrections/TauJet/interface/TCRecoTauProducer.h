#ifndef RecoTauTag_RecoTau_TCRecoTauProducer
#define RecoTauTag_RecoTau_TCRecoTauProducer

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

class TCRecoTauProducer : public edm::EDProducer {
    public:
  	explicit TCRecoTauProducer(const edm::ParameterSet& iConfig);
  	~TCRecoTauProducer();

  	virtual void produce(edm::Event&,const edm::EventSetup&);

    private:
  	edm::InputTag 	 caloRecoTauProducer;
        TCTauCorrector*  tcTauCorrector;
        TauJetCorrector* tauJetCorrector;
};
#endif
