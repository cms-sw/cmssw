#ifndef JetProducers_QGTagger_h
#define JetProducers_QGTagger_h
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "JetMETCorrections/Objects/interface/JetCorrector.h"

#include "RecoJets/JetAlgorithms/interface/QGLikelihoodCalculator.h"

class QGTagger : public edm::EDProducer {
   public:
      explicit QGTagger(const edm::ParameterSet&);
      ~QGTagger(){};
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      template <class jetClass> void calcVariables(const jetClass *jet, edm::Handle<reco::VertexCollection> vC);
      void putInEvent(std::string, edm::Handle<reco::PFJetCollection>, std::vector<float>*, edm::Event&);
      void putInEvent(std::string, edm::Handle<reco::PFJetCollection>, std::vector<int>*, edm::Event&);


      // ----------member data -------------------------
      edm::InputTag srcJets, srcRhoIso;
      std::string jecService;
      TString dataDir;
      bool useCHS;
      QGLikelihoodCalculator *qgLikelihood;
      float pt, axis2, ptD;
      int mult;
      const JetCorrector *JEC;
};

#endif
