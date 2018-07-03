#include "JetMETCorrections/Type1MET/interface/JetCleanerForType1METT.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

#include "PhysicsTools/PatUtils/interface/PATJetCorrExtractor.h"

namespace JetCleanerForType1MET_namespace
{
  template <>
  class InputTypeCheckerT<pat::Jet, PATJetCorrExtractor>
  {
    public:

     void operator()(const pat::Jet& jet) const
     {
       // check that pat::Jet is of PF-type
       if ( !jet.isPFJet() )
	 throw cms::Exception("InvalidInput")
	   << "Input pat::Jet is not of PF-type !!\n";
     }
     bool isPatJet(const pat::Jet& jet) const {
       return true;
     }
  };

  template <>
  class RawJetExtractorT<pat::Jet>
  {
    public:

     RawJetExtractorT(){}
     reco::Candidate::LorentzVector operator()(const pat::Jet& jet) const
     {
       if ( jet.jecSetsAvailable() ) return jet.correctedP4("Uncorrected");
       else return jet.p4();
     }
  };
}

typedef JetCleanerForType1METT<pat::Jet, PATJetCorrExtractor> PATJetCleanerForType1MET;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATJetCleanerForType1MET);


