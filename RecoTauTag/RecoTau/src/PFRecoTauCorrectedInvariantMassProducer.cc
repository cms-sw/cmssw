#include "RecoTauTag/RecoTau/interface/PFRecoTauCorrectedInvariantMassProducer.h"

void PFRecoTauCorrectedInvariantMassProducer::produce(Event& iEvent,const EventSetup& iEventSetup){
   // Get the PFTaus
   Handle<PFTauCollection> thePFTauCollection;
   iEvent.getByLabel(PFTauProducer_,thePFTauCollection);

   // Get the PFTau Decay Modes
   Handle<PFTauDecayModeAssociation> theDMAssoc;
   iEvent.getByLabel(PFTauDecayModeProducer_, theDMAssoc);

   auto_ptr<PFTauDiscriminator> decayModeIndexCollection(new PFTauDiscriminator(PFTauRefProd(thePFTauCollection)));

   //loop over the PFTau candidates
   for(size_t iPFTau = 0; iPFTau < thePFTauCollection->size(); ++iPFTau) {
      int theCorrectedInvariantMass = -1;
      PFTauRef thePFTauRef(thePFTauCollection,iPFTau);
      if (thePFTauRef.isNonnull()) {
         const PFTauDecayMode& theDecayMode = (*theDMAssoc)[thePFTauRef];
         // retrieve decay mode
         theCorrectedInvariantMass = static_cast<int>(theDecayMode.mass()); 
      }
      decayModeIndexCollection->setValue(iPFTau,theCorrectedInvariantMass);
   }
   iEvent.put(decayModeIndexCollection);

}
   
