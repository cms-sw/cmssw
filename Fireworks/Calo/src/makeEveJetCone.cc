#include "TEveJetCone.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "Fireworks/Core/interface/Context.h"

namespace fireworks 
{
TEveJetCone* makeEveJetCone(const reco::Jet& iData, const fireworks::Context& context)
{
   TEveJetCone* jet = new TEveJetCone();
   jet->SetApex(TEveVector(iData.vertex().x(),iData.vertex().y(),iData.vertex().z()));

   reco::Jet::Constituents c = iData.getJetConstituents();
   bool haveData = true;
   for ( reco::Jet::Constituents::const_iterator itr = c.begin(); itr != c.end(); ++itr )
   {
      if ( !itr->isAvailable() ) {
         haveData = false;
         break;
      }
   }

   double eta_size = 0.2;
   double phi_size = 0.2;
   if ( haveData ){
      eta_size = sqrt(iData.etaetaMoment());
      phi_size = sqrt(iData.phiphiMoment());
   }

   static const float offr = 5;
   static const float offz = offr/tan(context.caloTransAngle());
   if (iData.eta() < context.caloMaxEta())
      jet->SetCylinder(context.caloR1(false) -offr, context.caloZ1(false)-offz);
   else
      jet->SetCylinder(context.caloR2(false) -offr, context.caloZ2(false)-offz);


   jet-> AddEllipticCone(iData.eta(), iData.phi(), eta_size, phi_size);
   jet->SetPickable(kTRUE);
   return jet;
}
}       
