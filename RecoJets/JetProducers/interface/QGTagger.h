#ifndef JetProducers_QGTagger_h
#define JetProducers_QGTagger_h
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "RecoJets/JetAlgorithms/interface/QGLikelihoodCalculator.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

class QGTagger : public edm::EDProducer{
   public:
      explicit QGTagger(const edm::ParameterSet&);
      ~QGTagger(){};
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      template <class jetClass> void calcVariables(const jetClass *jet, edm::Handle<reco::VertexCollection> vC);
      template <typename T> void putInEvent(std::string, edm::Handle<reco::PFJetCollection>, std::vector<T>*, edm::Event&);


      // member data
      edm::InputTag srcJets, srcRho, srcVertexC;
      edm::EDGetTokenT<reco::PFJetCollection> jets_token;
      edm::EDGetTokenT<reco::VertexCollection> vertex_token;
      edm::EDGetTokenT<double> rho_token;
      std::string jetsLabel, jecService, systLabel;
      bool useJEC, produceSyst;
      QGLikelihoodCalculator *qgLikelihood;
      float pt, axis2, ptD;
      int mult;
      const JetCorrector *JEC;
};

#endif
