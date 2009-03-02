#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CLHEP/Vector/RotationInterfaces.h" 
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "Alignment/CommonAlignment/interface/AlignableDet.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


//__________________________________________________________________________________________________
AlignableDet::AlignableDet( const GeomDet* geomDet, bool addComponents ) : 
  AlignableComposite( geomDet ), 
  theAlignmentPositionError(0)
{
  // Take over APE from geometry _before_ creating daughters,
  // otherwise would overwrite their APEs!
  if (geomDet->alignmentPositionError()) {
    this->setAlignmentPositionError(*(geomDet->alignmentPositionError()));
  }

  if (addComponents) {
    if ( geomDet->components().size() == 0 ) { // Is a DetUnit
      throw cms::Exception("BadHierarchy") << "[AlignableDet] GeomDet with DetId " 
                                           << geomDet->geographicalId().rawId() 
                                           << " has no components, use AlignableDetUnit.\n";
    } else { // Push back all components
      const std::vector<const GeomDet*>& geomDets = geomDet->components();
      for (std::vector<const GeomDet*>::const_iterator idet = geomDets.begin(); 
            idet != geomDets.end(); ++idet) {
        const GeomDetUnit *unit = dynamic_cast<const GeomDetUnit*>(*idet);
        if (!unit) {
          throw cms::Exception("BadHierarchy") 
            << "[AlignableDet] component not GeomDetUnit, call with addComponents==false" 
            << " and build hierarchy yourself.\n";  // e.g. AlignableDTChamber
        }
        this->addComponent(new AlignableDetUnit(unit));
      }
    }
    // Ensure that the surface is not screwed up by addComponent, it must stay the GeomDet's one:
    theSurface = AlignableSurface(geomDet->surface());
  } // end addComponents  
}


//__________________________________________________________________________________________________
AlignableDet::~AlignableDet()
{

  delete theAlignmentPositionError;

}


//__________________________________________________________________________________________________
void AlignableDet::setAlignmentPositionError(const AlignmentPositionError& ape)
{

  if ( !theAlignmentPositionError )
	theAlignmentPositionError = new AlignmentPositionError( ape );
  else 
	*theAlignmentPositionError = ape;

  AlignableComposite::setAlignmentPositionError( ape );

}


//__________________________________________________________________________________________________
void AlignableDet::addAlignmentPositionError(const AlignmentPositionError& ape)
{

  if ( !theAlignmentPositionError ) {
    theAlignmentPositionError = new AlignmentPositionError( ape );
  } else {
    *theAlignmentPositionError += ape;
  }

  AlignableComposite::addAlignmentPositionError( ape );

}

//__________________________________________________________________________________________________
void AlignableDet::addAlignmentPositionErrorFromRotation(const RotationType& rot ) 
{

  // average error calculated by movement of a local point at
  // (xWidth/2,yLength/2,0) caused by the rotation rot
  GlobalVector localPositionVector = surface().toGlobal( LocalVector(.5 * surface().width(), .5 * surface().length(), 0.) );

  LocalVector::BasicVectorType lpvgf = localPositionVector.basicVector();
  GlobalVector gv( rot.multiplyInverse(lpvgf) - lpvgf );

  AlignmentPositionError  ape( gv.x(),gv.y(),gv.z() );
  this->addAlignmentPositionError( ape );

  AlignableComposite::addAlignmentPositionErrorFromRotation( rot );

}

//__________________________________________________________________________________________________
void AlignableDet::addAlignmentPositionErrorFromLocalRotation(const RotationType& rot )
{

  RotationType globalRot = globalRotation().multiplyInverse(rot*globalRotation());
  this->addAlignmentPositionErrorFromRotation(globalRot);

  AlignableComposite::addAlignmentPositionErrorFromLocalRotation( rot );

}



//__________________________________________________________________________________________________
Alignments* AlignableDet::alignments() const
{

  Alignments* m_alignments = new Alignments();
  RotationType rot( this->globalRotation() );

  // Get position, rotation, detId
  Hep3Vector clhepVector( globalPosition().x(), globalPosition().y(), globalPosition().z() );
  HepRotation clhepRotation( HepRep3x3( rot.xx(), rot.xy(), rot.xz(),
					rot.yx(), rot.yy(), rot.yz(),
					rot.zx(), rot.zy(), rot.zz() ) );
  uint32_t detId = this->geomDetId().rawId();
  
  AlignTransform transform( clhepVector, clhepRotation, detId );
  
  // Add to alignments container
  m_alignments->m_align.push_back( transform );

  // Add components recursively (if it is not already an alignable det unit)
  std::vector<Alignable*> comp = this->components();
  if ( comp.size() > 1 ) {
    for ( std::vector<Alignable*>::iterator i=comp.begin(); i!=comp.end(); i++ )
      {
	Alignments* tmpAlignments = (*i)->alignments();
	std::copy( tmpAlignments->m_align.begin(), tmpAlignments->m_align.end(), 
		   std::back_inserter(m_alignments->m_align) );
	delete tmpAlignments;
      }
  }


  return m_alignments;

}


//__________________________________________________________________________________________________
AlignmentErrors* AlignableDet::alignmentErrors( void ) const
{


  AlignmentErrors* m_alignmentErrors = new AlignmentErrors();

  // Add associated alignment position error
  uint32_t detId = this->geomDetId().rawId();
  HepSymMatrix clhepSymMatrix(3,0);
  if ( theAlignmentPositionError ) // Might not be set
    clhepSymMatrix= theAlignmentPositionError->globalError().matrix();
  AlignTransformError transformError( clhepSymMatrix, detId );
  m_alignmentErrors->m_alignError.push_back( transformError );
  
  // Add components recursively (if it is not already an alignable det unit)
  std::vector<Alignable*> comp = this->components();
  if ( comp.size() > 1 ) {
    for ( std::vector<Alignable*>::iterator i=comp.begin(); i!=comp.end(); i++ )
      {
		AlignmentErrors* tmpAlignmentErrors = (*i)->alignmentErrors();
		std::copy( tmpAlignmentErrors->m_alignError.begin(),
			   tmpAlignmentErrors->m_alignError.end(), 
			   std::back_inserter(m_alignmentErrors->m_alignError) );
		delete tmpAlignmentErrors;
      }
  }
  
  return m_alignmentErrors;

}
