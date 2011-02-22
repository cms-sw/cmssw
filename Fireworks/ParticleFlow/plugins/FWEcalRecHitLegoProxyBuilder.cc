#include "FWEcalRecHitLegoProxyBuilder.h"

void
FWEcalRecHitLegoProxyBuilder::localModelChanges( const FWModelId &iId, TEveElement *iCompound,
												FWViewType::EType viewType, const FWViewContext *vc)
{
	iCompound->CSCApplyMainColorToMatchingChildren();
	TEveElement *square = iCompound->FindChild("Square");
	if( square )
		square->SetMainColor( kBlack );	// Always set the square on top to black
}

float
FWEcalRecHitLegoProxyBuilder::calculateET( const std::vector<TEveVector> &vertices, float E )
{
	TEveVector vec;
	float e_t = 0;

	for( size_t i = 0; i < vertices.size(); i++ )
	{
		vec.fX += vertices[i].fX;
		vec.fY += vertices[i].fY;		// Get the total for x, y, z values
		vec.fZ += vertices[i].fZ;
	}

	vec *= 1.0f / 8.0f;					// Actually calculate centre point for vector
	vec.Normalize();
	vec *= E;
	e_t = vec.Perp();

	return e_t;
}

void
FWEcalRecHitLegoProxyBuilder::build( const FWEventItem *iItem, TEveElementList *product, const FWViewContext* )
{
   size_t size = iItem->size();
   int iSize = static_cast<int>(size);
   float max = 0;
   float etData[iSize];
   std::vector< std::vector<TEveVector> > recHitData(0);
   std::vector<TEveVector> null(8);


   for( int index = 0; index < iSize; index++ )
   {
      const EcalRecHit &iData = modelData(index);
      const float* corners = item()->getGeom()->getCorners( iData.detid() );
      std::vector<TEveVector> etaphiCorners(8);
      float e_t = 0;

      if(corners == 0 )
      {
         recHitData.push_back( null );
         etData[index] = 0;
         continue;
      }

      for( int i = 4; i < 8; i++ )
      {
         int j = (i-4)*3;
         TEveVector cv = TEveVector(corners[j], corners[j+1], corners[j+2]);
         etaphiCorners[i].fX = cv.Eta();	// Conversion of rechit X/Y values for plotting in Eta/Phi
         etaphiCorners[i].fY = cv.Phi();
         etaphiCorners[i].fZ = 0.1;					// Small z offset so that the tower is slightly above the 'ground'

         etaphiCorners[i-4] = etaphiCorners[i];		// Front face can simply be plotted exactly over the top of the back face
         etaphiCorners[i-4].fZ = 0.0;				// There needs to be a difference in facet Z values for normalization
      }

      e_t = calculateET( etaphiCorners, iData.energy() );
      if( e_t > max ) { max = e_t; }
      recHitData.push_back( etaphiCorners );			// Store the newly calculated data to remove the need for re-processing
      etData[index] = e_t;
   }

   for( int i = 0; i < iSize; i++ )
   {
      if( recHitData[i] == null ) { continue; }		// Don't do anything if the current index is NULL data

      TEveCompound *itemHolder = createCompound();
      product->AddElement( itemHolder );

      LegoRecHit *recHit = new LegoRecHit( recHitData[i].size(), recHitData[i], itemHolder, this, etData[i], max );
      recHit->setSquareColor( kBlack );
   }
}

REGISTER_FWPROXYBUILDER( FWEcalRecHitLegoProxyBuilder, EcalRecHit, "Ecal RecHit PF", FWViewType::kLegoBit );
