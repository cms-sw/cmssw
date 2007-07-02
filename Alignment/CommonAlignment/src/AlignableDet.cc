#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "Alignment/CommonAlignment/interface/AlignableDet.h"


//__________________________________________________________________________________________________
AlignableDet::AlignableDet( const GeomDet* geomDet ) : 
  AlignableComposite( geomDet ), 
  theAlignmentPositionError(0)
{
  
  // Behaviour depends on level of components:
  // Check if the AlignableDet is a CompositeDet or a DetUnit
  // In both cases, we have to down-cast these GeomDets to GeomDetUnits
  
  if ( geomDet->components().size() == 0 ) // Is a DetUnit
    {
      const GeomDetUnit* tmpGeomDetUnit = dynamic_cast<const GeomDetUnit*>( geomDet );
      theDetUnits.push_back( new AlignableDetUnit( tmpGeomDetUnit ) );
    }
  else // Is a compositeDet: push back all components
    {
      std::vector< const GeomDet*> geomDets = geomDet->components();
      for ( std::vector<const GeomDet*>::iterator idet=geomDets.begin(); 
			idet != geomDets.end(); idet++ )
		{
		  const GeomDetUnit* tmpGeomDetUnit = dynamic_cast<const GeomDetUnit*>( *idet );
		  if ( tmpGeomDetUnit ) // Just check down-cast worked...
			theDetUnits.push_back( new AlignableDetUnit( tmpGeomDetUnit ) );
		}
    }

  // Also store width and length of geomdet surface
  theWidth  = geomDet->surface().bounds().width();
  theLength = geomDet->surface().bounds().length();

}


//__________________________________________________________________________________________________
AlignableDet::~AlignableDet() {

  delete theAlignmentPositionError;
  for ( std::vector<AlignableDetUnit*>::iterator iter = theDetUnits.begin();
		iter != theDetUnits.end(); iter++ )
	delete *iter;

}


//__________________________________________________________________________________________________
std::vector<Alignable*> AlignableDet::components() const 
{

  std::vector<Alignable*> result;
  
  result.insert( result.end(), theDetUnits.begin(), theDetUnits.end() );

  return result;

}


//__________________________________________________________________________________________________
AlignableDetUnit &AlignableDet::detUnit(int i) 
{

  if ( i >= size() ) 
    throw cms::Exception("LogicError") << "DetUnit index (" << i << ") out of range";

  return *theDetUnits[i];

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

  if ( !theAlignmentPositionError ) this->setAlignmentPositionError( ape );
  else 
    this->setAlignmentPositionError( *theAlignmentPositionError += ape );

  AlignableComposite::addAlignmentPositionError( ape );

}

//__________________________________________________________________________________________________
void AlignableDet::addAlignmentPositionErrorFromRotation(const RotationType& rot ) 
{

  // average error calculated by movement of a local point at
  // (xWidth/2,yLength/2,0) caused by the rotation rot
  const GlobalVector localPositionVector = this->globalPosition()
    - this->surface().toGlobal( Local3DPoint(theWidth/2.0, theLength/2.0, 0.) );

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
  if ( comp.size() > 1 )
    for ( std::vector<Alignable*>::iterator i=comp.begin(); i!=comp.end(); i++ )
      {
		Alignments* tmpAlignments = (*i)->alignments();
		std::copy( tmpAlignments->m_align.begin(), tmpAlignments->m_align.end(), 
				   std::back_inserter(m_alignments->m_align) );
		delete tmpAlignments;
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
  if ( comp.size() > 1 )
    for ( std::vector<Alignable*>::iterator i=comp.begin(); i!=comp.end(); i++ )
      {
		AlignmentErrors* tmpAlignmentErrors = (*i)->alignmentErrors();
		std::copy( tmpAlignmentErrors->m_alignError.begin(), tmpAlignmentErrors->m_alignError.end(), 
				   std::back_inserter(m_alignmentErrors->m_alignError) );
		delete tmpAlignmentErrors;
      }
  
  return m_alignmentErrors;

}
