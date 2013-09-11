
/** \class CorrectedCaloMETProducer
 *
 * Instantiate CorrectedMETProducer template for CaloMET
 *
 * NOTE: This file also defines concrete implementation of CorrectedMETFactory template
 *       specific to CaloMET
 *
 *
 * \authors Michael Schmitt, Richard Cavanaugh, The University of Florida
 *          Florent Lacroix, University of Illinois at Chicago
 *          Christian Veelken, LLR
 *
 *
 *
 */

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CorrMETData.h"

#include "JetMETCorrections/Type1MET/interface/CorrectedMETProducerT.h"

namespace CorrectedMETProducer_namespace
{
  template <>
  class CorrectedMETFactoryT<reco::CaloMET>
  {
   public:

    reco::CaloMET operator()(const reco::CaloMET& rawMEt, const CorrMETData& correction) const
    {
      std::vector<CorrMETData> corrections = rawMEt.mEtCorr();
      corrections.push_back(correction);
      return reco::CaloMET(rawMEt.getSpecific(), 
			   correctedSumEt(rawMEt, correction), 
			   corrections,
			   correctedP4(rawMEt, correction), 
			   rawMEt.vertex());
    }
  };
}

typedef CorrectedMETProducerT<reco::CaloMET> CorrectedCaloMETProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CorrectedCaloMETProducer);

