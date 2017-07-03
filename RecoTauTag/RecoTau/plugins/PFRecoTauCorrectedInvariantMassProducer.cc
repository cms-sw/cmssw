/* 
 * class PFRecoTauCorrectedInvariantMassProducer
 * created : April 16, 2009
 * revised : ,
 * Authors : Evan K. Friis, (UC Davis), Simone Gennai (SNS)
 *
 * Associates the invariant mass reconstruced in the PFTauDecayMode production 
 * to its underlying PFTau, in PFTau discriminator format.
 *
 * The following corrections are applied in the PFTauDecayMode production
 *
 * 1) UE filtering
 * 2) Merging of PFGammas into candidate pi zeros
 * 3) Application of mass hypothesis (charged/neutral pion) to constituents
 *
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "DataFormats/TauReco/interface/PFTauDecayMode.h"
#include "DataFormats/TauReco/interface/PFTauDecayModeAssociation.h"

namespace {
using namespace reco;

class PFRecoTauCorrectedInvariantMassProducer : public PFTauDiscriminationProducerBase  {
   public:
      explicit PFRecoTauCorrectedInvariantMassProducer(const edm::ParameterSet& iConfig):PFTauDiscriminationProducerBase(iConfig){   
         PFTauDecayModeProducer_     = iConfig.getParameter<edm::InputTag>("PFTauDecayModeProducer");
      }
      ~PFRecoTauCorrectedInvariantMassProducer() override{} 
      double discriminate(const PFTauRef& pfTau) const override;
      void beginEvent(const edm::Event& event, const edm::EventSetup& evtSetup) override;
   private:
      edm::InputTag PFTauDecayModeProducer_;
      edm::Handle<PFTauDecayModeAssociation> theDMAssoc;
};

void PFRecoTauCorrectedInvariantMassProducer::beginEvent(const edm::Event& event, const edm::EventSetup& evtSetup)
{
   // Get the PFTau Decay Modes
   event.getByLabel(PFTauDecayModeProducer_, theDMAssoc);
}

double PFRecoTauCorrectedInvariantMassProducer::discriminate(const PFTauRef& thePFTauRef) const
{
   const PFTauDecayMode& theDecayMode = (*theDMAssoc)[thePFTauRef];
   return theDecayMode.mass();
}
}
DEFINE_FWK_MODULE(PFRecoTauCorrectedInvariantMassProducer);
