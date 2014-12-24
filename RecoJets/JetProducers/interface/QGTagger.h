#ifndef JetProducers_QGTagger_h
#define JetProducers_QGTagger_h
#include <tuple>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "RecoJets/JetAlgorithms/interface/QGLikelihoodCalculator.h"

class QGTagger : public edm::EDProducer{
   public:
      explicit QGTagger(const edm::ParameterSet&);
      ~QGTagger(){ delete qgLikelihood;};
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      std::tuple<int, float, float> calcVariables(const reco::Jet*, edm::Handle<reco::VertexCollection>&);
      template <typename T> void putInEvent(std::string, const edm::Handle<edm::View<reco::Jet>>&, std::vector<T>*, edm::Event&);
      bool isPackedCandidate(const reco::Candidate* candidate);

      edm::EDGetTokenT<edm::View<reco::Jet>> 	jetsToken;
      edm::EDGetTokenT<reco::JetCorrector> 	jetCorrectorToken;
      edm::EDGetTokenT<reco::VertexCollection> 	vertexToken;
      edm::EDGetTokenT<double> 			rhoToken;
      std::string 				jetsLabel, systLabel;
      const bool 				useQC, useJetCorr, produceSyst;
      bool					weStillNeedToCheckJetCandidates, weAreUsingPackedCandidates;
      QGLikelihoodCalculator *			qgLikelihood;
};

#endif
