#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HLTrigger/JetMET/interface/MVAJetPuId.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/JetReco/interface/PileupJetIdentifier.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

class MVAJetPuIdProducer : public edm::EDProducer {
public:
   explicit MVAJetPuIdProducer(const edm::ParameterSet&);
   ~MVAJetPuIdProducer();

   static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
   virtual void produce(edm::Event&, const edm::EventSetup&) override;
     

   void initJetEnergyCorrector(const edm::EventSetup &iSetup, bool isData);

   edm::InputTag jets_, vertexes_, jetids_, rho_;
   std::string jec_;
   bool runMvas_, produceJetIds_, inputIsCorrected_, applyJec_;
   std::vector<std::pair<std::string, MVAJetPuId *> > algos_;
   
   bool residualsFromTxt_;
   edm::FileInPath residualsTxt_;
   FactorizedJetCorrector *jecCor_;
   std::vector<JetCorrectorParameters> jetCorPars_;

       edm::EDGetTokenT<edm::View<reco::Jet> > input_jet_token_;
       edm::EDGetTokenT<reco::VertexCollection> input_vertex_token_;
       edm::EDGetTokenT<edm::ValueMap<StoredPileupJetIdentifier> > input_vm_pujetid_token_;
       edm::EDGetTokenT<double> input_rho_token_;

};

