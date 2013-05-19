#include "PhysicsTools/PatUtils/interface/SmearedJetProducerT.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "DataFormats/METReco/interface/SigInputObj.h"
#include "RecoMET/METAlgorithms/interface/significanceAlgo.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "PhysicsTools/PatUtils/interface/PATJetCorrExtractor.h"

namespace SmearedJetProducer_namespace
{
  template <>
  class GenJetMatcherT<pat::Jet>
  {
    public:

     GenJetMatcherT(const edm::ParameterSet& cfg)
       : dRmaxGenJetMatch_(0)
     {
       TString dRmaxGenJetMatch_formula = cfg.getParameter<std::string>("dRmaxGenJetMatch").data();
       dRmaxGenJetMatch_formula.ReplaceAll("genJetPt", "x");
       dRmaxGenJetMatch_ = new TFormula("dRmaxGenJetMatch", dRmaxGenJetMatch_formula.Data());
     }
     ~GenJetMatcherT()
     {
       delete dRmaxGenJetMatch_;
     }

     const reco::GenJet* operator()(const pat::Jet& jet, edm::Event* evt = 0) const
     {
       const reco::GenJet* retVal = 0;

       // CV: apply matching criterion which is tighter than PAT default,
       //     in order to avoid "accidental" matches for which the difference between genJetPt and recJetPt is large 
       //    (the large effect of such bad matches on the MEt smearing is "unphysical",
       //     because the large difference between genJetPt and recJetPt results from the matching
       //     and not from the particle/jet reconstruction)
       //retVal = jet.genJet();
       if ( jet.genJet() ) {
	 const reco::GenJet* genJet = jet.genJet();
	 double dR = deltaR(jet.p4(), genJet->p4());
	 if ( dR < dRmaxGenJetMatch_->Eval(genJet->pt()) ) retVal = genJet;
       }
       
       return retVal;
     }

    private:
    
     TFormula* dRmaxGenJetMatch_;
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
