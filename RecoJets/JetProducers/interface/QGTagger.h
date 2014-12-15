#ifndef JetProducers_QGTagger_h
#define JetProducers_QGTagger_h
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "RecoJets/JetAlgorithms/interface/QGLikelihoodCalculator.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

class QGTagger : public edm::EDProducer{
   public:
      explicit QGTagger(const edm::ParameterSet&);
      ~QGTagger(){};
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      template <class jetCollection> void produceForJetCollection(edm::Event& iEvent, const edm::EventSetup& iSetup, const jetCollection& jets);
      template <class jetClass> void calcVariables(const jetClass *jet, edm::Handle<reco::VertexCollection> vC);
      template <class jetCollection, typename T> void putInEvent(std::string, const jetCollection&, std::vector<T>*, edm::Event&);


      // member data
      edm::InputTag jetsInputTag;
      edm::EDGetTokenT<reco::PFJetCollection> jets_token;
      edm::EDGetTokenT<pat::JetCollection> patJets_token;
      edm::EDGetTokenT<reco::JetCorrector> jetCorrector_token;
      edm::EDGetTokenT<reco::VertexCollection> vertex_token;
      edm::EDGetTokenT<double> rho_token;
      edm::InputTag jetCorrector_inputTag;
      std::string jetsLabel, systLabel;
      bool useQC, usePatJets, useJetCorr, produceSyst;
      QGLikelihoodCalculator *qgLikelihood;
      float pt, axis2, ptD;
      int mult;
};

#endif
