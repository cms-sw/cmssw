
/** \class CorrectedPFMETProducer
 *
 * Instantiate CorrectedMETProducer template for PFMET
 *
 * NOTE: This file also defines concrete implementation of CorrectedMETFactory template
 *       specific to PFMET
 *
 *
 * \authors Michael Schmitt, Richard Cavanaugh, The University of Florida
 *          Florent Lacroix, University of Illinois at Chicago
 *          Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: CorrectedPFMETProducer.cc,v 1.1 2011/09/13 14:35:34 veelken Exp $
 *
 */

#include "DataFormats/METReco/interface/PFMET.h"

#include "JetMETCorrections/Type1MET/interface/CorrectedMETProducerT.h"

namespace CorrectedMETProducer_namespace
{
  template <>
  class CorrectedMETFactoryT<reco::PFMET>
  {
   public:

    reco::PFMET operator()(const reco::PFMET& rawMEt, const CorrMETData& correction) const
    {
      return reco::PFMET(rawMEt.getSpecific(), 
			 correctedSumEt(rawMEt, correction), 
			 correctedP4(rawMEt, correction), 
			 rawMEt.vertex());
    }
  };
}

typedef CorrectedMETProducerT<reco::PFMET> CorrectedPFMETProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CorrectedPFMETProducer);

