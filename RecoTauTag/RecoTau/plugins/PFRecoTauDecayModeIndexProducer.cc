/* 
 * class PFRecoTauDecayModeIndexProducer
 * created : April 16, 2009
 * revised : ,
 * Authors : Evan K. Friis, (UC Davis), Simone Gennai (SNS)
 *
 * Associates the decay mode index (see enum in DataFormats/TauReco/interface/PFTauDecayMode.h)
 * reconstruced in the PFTauDecayMode production to its underlying PFTau, in PFTau discriminator format
 *
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include "DataFormats/TauReco/interface/PFTauDecayMode.h"
#include "DataFormats/TauReco/interface/PFTauDecayModeAssociation.h"

namespace {

using namespace reco;

class PFRecoTauDecayModeIndexProducer final : public PFTauDiscriminationProducerBase {
   public:
      explicit PFRecoTauDecayModeIndexProducer(const edm::ParameterSet& iConfig):PFTauDiscriminationProducerBase(iConfig) {   
         PFTauDecayModeProducer_     = iConfig.getParameter<edm::InputTag>("PFTauDecayModeProducer");
      }
      ~PFRecoTauDecayModeIndexProducer() override{} 
      double discriminate(const PFTauRef& thePFTauRef) const override ;
      void beginEvent(const edm::Event& evt, const edm::EventSetup& evtSetup) override;
   private:
      edm::InputTag PFTauDecayModeProducer_;
      edm::Handle<PFTauDecayModeAssociation> decayModes_; // holds the PFTauDecayModes for the current event
};


void PFRecoTauDecayModeIndexProducer::beginEvent(const edm::Event& event, const edm::EventSetup& evtSetup)
{
   // Get the PFTau Decay Modes
   event.getByLabel(PFTauDecayModeProducer_, decayModes_);
}

double PFRecoTauDecayModeIndexProducer::discriminate(const PFTauRef& thePFTauRef) const
{
   int theDecayModeIndex = -1;

   const PFTauDecayMode& theDecayMode = (*decayModes_)[thePFTauRef];

   // retrieve decay mode
   theDecayModeIndex = static_cast<int>(theDecayMode.getDecayMode()); 

   if (theDecayModeIndex < 0) theDecayModeIndex = -1;

   return theDecayModeIndex;
}
  
}

DEFINE_FWK_MODULE(PFRecoTauDecayModeIndexProducer);
