
/** \class CorrectedPATMETProducer
 *
 * Instantiate CorrectedMETProducer template for pat::MET (PF or Calo)
 *
 * NOTE: This file also defines concrete implementation of CorrectedMETFactory template
 *       specific to pat::MET
 *
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: CorrectedPATMETProducer.cc,v 1.1 2011/09/16 08:09:52 veelken Exp $
 *
 */

#include "DataFormats/PatCandidates/interface/MET.h"

#include "JetMETCorrections/Type1MET/interface/CorrectedMETProducerT.h"

namespace CorrectedMETProducer_namespace
{
  template <>
  class CorrectedMETFactoryT<pat::MET>
  {
   public:

    pat::MET operator()(const pat::MET& rawMEt, const CorrMETData& correction) const
    {
      pat::MET correctedMEt(rawMEt);
      // CV: cannot set sumEt data-member to corrected value
      correctedMEt.setP4(correctedP4(rawMEt, correction));
      return correctedMEt;
    }
  };
}

typedef CorrectedMETProducerT<pat::MET> CorrectedPATMETProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CorrectedPATMETProducer);

