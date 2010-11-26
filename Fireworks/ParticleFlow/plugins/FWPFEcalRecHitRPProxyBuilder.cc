#include "FWPFEcalRecHitRPProxyBuilder.h"

//______________________________________________________________________________________________________
void
FWPFEcalRecHitRPProxyBuilder::scaleProduct( TEveElementList *parent, FWViewType::EType type, const FWViewContext *vc )
{
   typedef std::vector<FWPFRhoPhiRecHit*> rpRecHits;
   unsigned int index = 0;
   
   for( rpRecHits::iterator i = towers.begin(); i != towers.end(); ++i )
   {
      towers[index]->updateScale( vc );
      index++;
   }
}

//______________________________________________________________________________________________________
void
FWPFEcalRecHitRPProxyBuilder::cleanLocal()
{
   typedef std::vector<FWPFRhoPhiRecHit*> rpRecHits;
   for( rpRecHits::iterator i = towers.begin(); i != towers.end(); ++i )
   {
      (*i)->clean();
   }

   towers.clear();
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
float
FWPFEcalRecHitRPProxyBuilder::calculateEt( const TEveVector &centre, float E )
{
   TEveVector vec = centre;
   float et;

   vec.Normalize();
   vec *= E;
   et = vec.Perp();

   return et;
}

//______________________________________________________________________________________________________
void
FWPFEcalRecHitRPProxyBuilder::build( const FWEventItem *iItem, TEveElementList *product, const FWViewContext *vc )
{
   cleanLocal();  // Called so that previous data isn't used when new view is added

   for( unsigned int index = 0; index < static_cast<unsigned int>( iItem->size() ); index++ )
   {
      TEveCompound *itemHolder = createCompound();
      product->AddElement( itemHolder );

      bool added = false;
      float E, et;
      float ecalR = context().caloR1();
      Double_t lPhi, rPhi;
      const EcalRecHit &iData = modelData( index );
      const float *vertices = item()->getGeom()->getCorners( iData.detid() );

      TEveVector centre = calculateCentre( vertices );
      TEveVector lVec = TEveVector( vertices[0], vertices[1], 0 );   // Bottom left corner of tower
      TEveVector rVec = TEveVector( vertices[9], vertices[10], 0 );  // Bottom right corner of tower
      
      lPhi = lVec.Phi();
      rPhi = rVec.Phi();
      E = iData.energy();
      et = calculateEt( centre, E );

      for( unsigned int i = 0; i < towers.size(); i++ )
      {   // Small range to catch rounding inaccuracies etc.
         Double_t phi = towers[i]->getlPhi();
         if( ( lPhi == phi ) || ( ( lPhi < phi + 0.0005 ) && ( lPhi > phi - 0.0005 ) ) )
         {
            towers[i]->addChild( this, itemHolder, vc, E, et );
            added = true;
            break;
         }
      }
   
      if( !added )
      {
         rVec.fX = ecalR * cos( rPhi ); rVec.fY = ecalR * sin( rPhi );
         lVec.fX = ecalR * cos( lPhi ); lVec.fY = ecalR * sin( lPhi );
         std::vector<TEveVector> bCorners(2);
         bCorners[0] = lVec;
         bCorners[1] = rVec;

         FWPFRhoPhiRecHit *rh = new FWPFRhoPhiRecHit( this, itemHolder, vc, E, et, lPhi, rPhi, bCorners );
         context().voteMaxEtAndEnergy(et, E);
         towers.push_back( rh );
      }
   }
}

//______________________________________________________________________________________________________
REGISTER_FWPROXYBUILDER( FWPFEcalRecHitRPProxyBuilder, EcalRecHit, "PF Ecal RecHit", FWViewType::kRhoPhiPFBit );
