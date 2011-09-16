#include "JetMETCorrections/Type1MET/interface/CaloJetMETcorrInputProducerT.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

#include "PhysicsTools/PatUtils/interface/PATJetCorrExtractor.h"

namespace CaloJetMETcorrInputProducer_namespace
{
  template <>
  class InputTypeCheckerT<pat::Jet>
  {
    public:

    void operator()(const pat::Jet& jet) const 
     {
       // check that pat::Jet is of Calo-type
       if ( !jet.isCaloJet() )
	 throw cms::Exception("InvalidInput")
	   << "Input pat::Jet is not of Calo-type !!\n";
     } 
  };
}

typedef CaloJetMETcorrInputProducerT<pat::Jet> PATCaloJetMETcorrInputProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATCaloJetMETcorrInputProducer);
