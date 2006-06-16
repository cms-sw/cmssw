#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "Alignment/CommonAlignment/interface/AlignableDet.h"


//__________________________________________________________________________________________________
AlignableDet::AlignableDet( GeomDet* geomDet ) : AlignableComposite( geomDet )
{
  
  // Behaviour depends on level of components:
  // Check if the AlignableDet is a CompositeDet or a DetUnit
  // In both cases, we have to down-cast these GeomDets to GeomDetUnits
  // and cast away the const...
  
  if ( geomDet->components().size() == 0 ) // Is a DetUnit
	{
	  const GeomDetUnit* tmpGeomDetUnit = dynamic_cast<const GeomDetUnit*>( geomDet );
	  theDetUnits.push_back( 
							new AlignableDetUnit( const_cast<GeomDetUnit*>(tmpGeomDetUnit) ) 
							);
	}
  else // Is a compositeDet: push back all components
	{
	  std::vector< const GeomDet*> geomDets = geomDet->components();
	  for ( std::vector<const GeomDet*>::iterator idet=geomDets.begin(); 
			idet != geomDets.end(); idet++ )
		{
		  
		  const GeomDetUnit* tmpGeomDetUnit = dynamic_cast<const GeomDetUnit*>( *idet );
		  if ( tmpGeomDetUnit ) // Just check down-cast worked...
			theDetUnits.push_back(
								  new AlignableDetUnit( const_cast<GeomDetUnit*>(tmpGeomDetUnit) )
								  );
		}
	}

}


//__________________________________________________________________________________________________
AlignableDet::~AlignableDet() {};


//__________________________________________________________________________________________________
std::vector<Alignable*> AlignableDet::components() const 
{

  std::vector<Alignable*> result;
  
  result.insert( result.end(), theDetUnits.begin(), theDetUnits.end() );

  return result;

}


//__________________________________________________________________________________________________
AlignableDetUnit &AlignableDet::geomDetUnit(int i) 
{

  if ( i >= size() ) 
    throw cms::Exception("LogicError") << "DetUnit index (" << i << ") out of range";

  return *theDetUnits[i];

}


//__________________________________________________________________________________________________
void AlignableDet::setAlignmentPositionError(const AlignmentPositionError& ape)
{

  for ( std::vector<AlignableDetUnit*>::iterator i=theDetUnits.begin(); 
		i!=theDetUnits.end();i++ )
    (*i)->setAlignmentPositionError(ape);

}


//__________________________________________________________________________________________________
Alignments* AlignableDet::alignments() const
{

  Alignments* m_alignments = new Alignments();
  RotationType rot( theGeomDet->rotation() );

  // Get associated geomDet's alignments (position, rotation, detId)
  Hep3Vector clhepVector( globalPosition().x(), globalPosition().y(), globalPosition().z() );
  HepRotation clhepRotation( HepRep3x3( rot.xx(), rot.xy(), rot.xz(),
										rot.yx(), rot.yy(), rot.yz(),
										rot.zx(), rot.zy(), rot.zz() )
							 );
  uint32_t detId = this->geomDet()->geographicalId().rawId();
  
  // TEMPORARILY also include alignment error
  HepSymMatrix clhepSymMatrix(0);
  if ( this->geomDet()->alignmentPositionError() ) // Might not be set
	clhepSymMatrix= this->geomDet()->alignmentPositionError()->globalError().matrix();
  
  AlignTransform transform( clhepVector, clhepRotation, clhepSymMatrix, detId );
  
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
	  }

  
  return m_alignments;

}


//__________________________________________________________________________________________________
AlignmentErrors* AlignableDet::alignmentErrors( void ) const
{


  AlignmentErrors* m_alignmentErrors = new AlignmentErrors();

  // Add associated geomDet alignment position error
  uint32_t detId = this->geomDet()->geographicalId().rawId();
  HepSymMatrix clhepSymMatrix(0);
  if ( this->geomDet()->alignmentPositionError() ) // Might not be set
	{
	  clhepSymMatrix= 
		this->geomDet()->alignmentPositionError()->globalError().matrix();
	}
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
	  }
  
  return m_alignmentErrors;

}
