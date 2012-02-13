#include "PhysicsTools/PatUtils/interface/SmearedJetProducerT.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "RecoMET/METAlgorithms/interface/SigInputObj.h"
#include "RecoMET/METAlgorithms/interface/significanceAlgo.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "PhysicsTools/PatUtils/interface/PATJetCorrExtractor.h"

namespace SmearedJetProducer_namespace
{
  template <>
  class GenJetMatcherT<pat::Jet>
  {
    public:

     GenJetMatcherT(const edm::ParameterSet&) {}
     ~GenJetMatcherT() {}

     const reco::GenJet* operator()(const pat::Jet& jet, edm::Event* evt = 0) const
     {
       return jet.genJet();
     }
  };

  template <>
  class JetResolutionExtractorT<pat::Jet>
  {
    public:

     JetResolutionExtractorT(const edm::ParameterSet& cfg) 
       : jetResolutions_(cfg)
     {}
     ~JetResolutionExtractorT() {}

     double operator()(const pat::Jet& jet) const
     {
       if ( jet.isPFJet() ) {
	 reco::PFJet pfJet(jet.p4(), jet.vertex(), jet.pfSpecific(), jet.getJetConstituents());
	 metsig::SigInputObj pfJetResolution = jetResolutions_.evalPFJet(&pfJet);
	 if ( pfJetResolution.get_energy() > 0. ) {
	   return jet.energy()*(pfJetResolution.get_sigma_e()/pfJetResolution.get_energy());
	 } else {
	   return 0.;
	 }
       } else {
	 throw cms::Exception("SmearedJetProducer::produce")
	   << " Jets of type other than PF not supported yet !!\n";
       }
     }

     metsig::SignAlgoResolutions jetResolutions_;
  };
}

typedef SmearedJetProducerT<pat::Jet, PATJetCorrExtractor> SmearedPATJetProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SmearedPATJetProducer);
