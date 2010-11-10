#include "FWPFEcalRecHitRPProxyBuilder.h"

//______________________________________________________________________________________________________
void
FWPFEcalRecHitRPProxyBuilder::scaleProduct( TEveElementList *parent, FWViewType::EType type, const FWViewContext *vc )
{
   FWViewEnergyScale *caloScale = vc->getEnergyScale( "Calo" );
   typedef std::vector<FWPFRhoPhiRecHit*> rpRecHits;
   unsigned int index = 0;

   for( rpRecHits::iterator i = towers.begin(); i != towers.end(); ++i )
   {
      float value = caloScale->getPlotEt() ? (*i)->GetEt() : (*i)->GetEnergy();
      TEveScalableStraightLineSet *ls = (*i)->GetLineSet();
      towers[index]->updateScale( ls, caloScale->getValToHeight() * value, index );
      index++;
   }
}

//______________________________________________________________________________________________________
TEveVector
FWPFEcalRecHitRPProxyBuilder::calculateCentre( const float *vertices )
{
   TEveVector centre;
   
   for( unsigned int i = 0; i < 8; i++ )
   {
      int j = i * 3;
      centre.fX += vertices[j];
      centre.fY += vertices[j+1];            // Total x,y,z values
      centre.fZ += vertices[j+2];
   }

   centre *= 1.0f / 8.0f;                  // Actually calculate the centre point

   return centre;
}

//______________________________________________________________________________________________________
void
FWPFEcalRecHitRPProxyBuilder::build( const FWEventItem *iItem, TEveElementList *product, const FWViewContext *vc )
{
   for( unsigned int index = 0; index < static_cast<unsigned int>( iItem->size() ); index++ )
   {
      TEveCompound *itemHolder = createCompound();
      product->AddElement( itemHolder );

      bool added = false;
      float E;
      double lPhi, rPhi;
      const EcalRecHit &iData = modelData( index );
      const float *vertices = item()->getGeom()->getCorners( iData.detid() );

      TEveVector centre = calculateCentre( vertices );
      TEveVector lVec = TEveVector( vertices[0], vertices[1], 0 );   // Bottom left corner of tower
      TEveVector rVec = TEveVector( vertices[9], vertices[10], 0 );   // Bottom right corner of tower
      
      lPhi = lVec.Phi();
      rPhi = rVec.Phi();
      E = iData.energy();

      if( index == 0 )
      {
         FWPFRhoPhiRecHit *rh = new FWPFRhoPhiRecHit( this, itemHolder, vc, centre, E, lPhi, rPhi, true );
         towers.push_back( rh );
         continue;
      }

      for( unsigned int i = 0; i < towers.size(); i++ )
      {   // Small range to catch rounding inaccuracies etc.
         double phi = towers[i]->GetlPhi();
         if( lPhi == phi || ( ( lPhi < phi + 0.0005 ) && ( lPhi > phi - 0.0005 ) ) )
         {
            towers[i]->Add( this, itemHolder, vc, E );
            added = true;
            break;
         }
      }
   
      if( !added )
      {
         FWPFRhoPhiRecHit *rh = new FWPFRhoPhiRecHit( this, itemHolder, vc, centre, E, lPhi, rPhi, true );
         towers.push_back( rh );
      }
   }
}

//______________________________________________________________________________________________________
REGISTER_FWPROXYBUILDER( FWPFEcalRecHitRPProxyBuilder, EcalRecHit, "Ecal RecHit PF", FWViewType::kRhoPhiBit );
