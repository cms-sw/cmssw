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

template <class jetClass> class QGTagger : public edm::EDProducer{
   public:
      explicit QGTagger(const edm::ParameterSet&);
      ~QGTagger(){};
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      typedef std::vector<jetClass> jetCollection;

      virtual void produce(edm::Event&, const edm::EventSetup&);
      void calcVariables(const jetClass *jet, edm::Handle<reco::VertexCollection> vC);
      template <typename T> void putInEvent(std::string, const edm::Handle<jetCollection>&, std::vector<T>*, edm::Event&);

      edm::EDGetTokenT<jetCollection> jets_token;
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
