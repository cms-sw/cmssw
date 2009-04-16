#include "RecoTauTag/RecoTau/interface/PFRecoTauDecayModeIndexProducer.h"

void PFRecoTauDecayModeIndexProducer::produce(Event& iEvent,const EventSetup& iEventSetup){
   // Get the PFTaus
   Handle<PFTauCollection> thePFTauCollection;
   iEvent.getByLabel(PFTauProducer_,thePFTauCollection);

   // Get the PFTau Decay Modes
   Handle<PFTauDecayModeAssociation> theDMAssoc;
   iEvent.getByLabel(PFTauDecayModeProducer_, theDMAssoc);

   auto_ptr<PFTauDiscriminator> decayModeIndexCollection(new PFTauDiscriminator(PFTauRefProd(thePFTauCollection)));

   //loop over the PFTau candidates
   for(size_t iPFTau = 0; iPFTau < thePFTauCollection->size(); ++iPFTau) {
      int theDecayModeIndex = -1;
      PFTauRef thePFTauRef(thePFTauCollection,iPFTau);
      if (thePFTauRef.isNonnull()) {
         const PFTauDecayMode& theDecayMode = (*theDMAssoc)[thePFTauRef];
         // retrieve decay mode
         theDecayModeIndex = static_cast<int>(theDecayMode.getDecayMode()); 
         if (theDecayModeIndex < 0) theDecayModeIndex = -1;
      }
      decayModeIndexCollection->setValue(iPFTau,theDecayModeIndex);
   }
   iEvent.put(decayModeIndexCollection);

}
   
