#ifndef JetProducers_QGTagger_h
#define JetProducers_QGTagger_h
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "RecoJets/JetAlgorithms/interface/QGLikelihoodCalculator.h"

template <class jetClass> class QGTagger : public edm::EDProducer{
   public:
      explicit QGTagger(const edm::ParameterSet&);
      ~QGTagger(){ delete qgLikelihood;};
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      typedef std::vector<jetClass> jetCollection;

      virtual void produce(edm::Event&, const edm::EventSetup&);
      void calcVariables(const jetClass*, int&, float&, float&, edm::Handle<reco::VertexCollection>&);
      template <typename T> void putInEvent(std::string, const edm::Handle<jetCollection>&, std::vector<T>*, edm::Event&);

      edm::EDGetTokenT<jetCollection> 		jetsToken;
      edm::EDGetTokenT<reco::JetCorrector> 	jetCorrectorToken;
      edm::EDGetTokenT<reco::VertexCollection> 	vertexToken;
      edm::EDGetTokenT<double> 			rhoToken;
      std::string 				jetsLabel, systLabel;
      const bool 				useQC, useJetCorr, produceSyst;
      QGLikelihoodCalculator *			qgLikelihood;
};

#endif
