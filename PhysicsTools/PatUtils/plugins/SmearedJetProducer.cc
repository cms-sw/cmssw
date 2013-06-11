#include "PhysicsTools/PatUtils/interface/SmearedJetProducerT.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "JetMETCorrections/Type1MET/interface/JetCorrExtractorT.h"

#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "DataFormats/METReco/interface/SigInputObj.h"
#include "RecoMET/METAlgorithms/interface/significanceAlgo.h"

namespace SmearedJetProducer_namespace
{
  template <>
  class JetResolutionExtractorT<reco::PFJet>
  {
    public:

     JetResolutionExtractorT(const edm::ParameterSet& cfg) 
       : jetResolutions_(cfg)
     {}
     ~JetResolutionExtractorT() {}

     double operator()(const reco::PFJet& jet) const
     {
       metsig::SigInputObj pfJetResolution = jetResolutions_.evalPFJet(&jet);
       if ( pfJetResolution.get_energy() > 0. ) {
	 return jet.energy()*(pfJetResolution.get_sigma_e()/pfJetResolution.get_energy());
       } else {
	 return 0.;
       }
     }

     metsig::SignAlgoResolutions jetResolutions_;
  };
}

typedef SmearedJetProducerT<reco::CaloJet, JetCorrExtractorT<reco::CaloJet> > SmearedCaloJetProducer;
typedef SmearedJetProducerT<reco::PFJet, JetCorrExtractorT<reco::PFJet> > SmearedPFJetProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SmearedCaloJetProducer);
DEFINE_FWK_MODULE(SmearedPFJetProducer);
