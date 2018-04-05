#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

/* 
 * class PFRecoTauDiscriminatioLogicalAndProducer
 *
 * Applies a boolean operator (AND or OR) to a set 
 * of PFTauDiscriminators.  Note that the set of PFTauDiscriminators
 * is accessed/combined in the base class (the Prediscriminants).
 *
 * This class merely exposes this behavior directly by 
 * returning true for all taus that pass the prediscriminants
 *
 * revised : Mon Aug 31 12:59:50 PDT 2009
 * Authors : Michele Pioppi, Evan Friis (UC Davis)
 */

using namespace reco;

class PFTauDiscriminatorLogicalAndProducer : public PFTauDiscriminationProducerBase {
   public:
      explicit PFTauDiscriminatorLogicalAndProducer(const edm::ParameterSet&);
      ~PFTauDiscriminatorLogicalAndProducer() override{};
      double discriminate(const PFTauRef& pfTau) const override;
   private:
      double passResult_;
};

PFTauDiscriminatorLogicalAndProducer::PFTauDiscriminatorLogicalAndProducer(const edm::ParameterSet& iConfig):PFTauDiscriminationProducerBase(iConfig)
{
   passResult_               = iConfig.getParameter<double>("PassValue");
   prediscriminantFailValue_ = iConfig.getParameter<double>("FailValue"); //defined in base class
}

double
PFTauDiscriminatorLogicalAndProducer::discriminate(const PFTauRef& pfTau) const 
{
   // if this function is called on a tau, it is has passed (in the base class)
   // the set of prediscriminants, using the prescribed boolean operation.  thus 
   // we only need to return TRUE
   return passResult_;
}

DEFINE_FWK_MODULE(PFTauDiscriminatorLogicalAndProducer);
