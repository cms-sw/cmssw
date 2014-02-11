#ifndef QGTagger_h
#define QGTagger_h
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "JetMETCorrections/Objects/interface/JetCorrector.h"

#include "RecoJets/JetAlgorithms/interface/QGLikelihoodCalculator.h"
#include "RecoJets/JetAlgorithms/interface/QGMLPCalculator.h"


class QGTagger : public edm::EDProducer {
   public:
      explicit QGTagger(const edm::ParameterSet&);
      ~QGTagger(){};
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      template <class jetClass> void calcVariables(const jetClass *jet, edm::Handle<reco::VertexCollection> vC, TString type);

      // ----------member data -------------------------
      edm::InputTag src, srcRho, srcRhoIso;
      std::string jecService;
      TString dataDir;
      Bool_t useCHS, isPatJet;
      QGLikelihoodCalculator *qgLikelihood;
      QGMLPCalculator *qgMLP;
      std::map<TString, Float_t> variables;
      const JetCorrector *JEC;           
};

#endif
