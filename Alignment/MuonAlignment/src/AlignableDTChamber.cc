/** \file
 *
 *  $Date: 2006/8/4 10:10:07 $
 *  $Revision: 1.0 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */
 

#include "Alignment/MuonAlignment/interface/AlignableDTChamber.h"


// The constructor gets all components and stores them as AlignableDets
AlignableDTChamber::AlignableDTChamber( const GeomDet* geomDet ) : 
  AlignableComposite( geomDet ),
  theAlignmentPositionError(0)
{

  // Retrieve components
  if ( geomDet->components().size() ) 
	{
	  std::vector<const GeomDet*> m_SuperLayers = geomDet->components();
	  for ( std::vector<const GeomDet*>::iterator iGeomDet = m_SuperLayers.begin(); 
			iGeomDet != m_SuperLayers.end(); iGeomDet++ )
		theDets.push_back( new AlignableDet( *iGeomDet ) );
	}

  // Also store width and length of geomdet surface
  theWidth  = geomDet->surface().bounds().width();
  theLength = geomDet->surface().bounds().length();

}


// Destructor: delete all AlignableDet objects
AlignableDTChamber::~AlignableDTChamber() 
{

  // Properly delete all elements of the vector (they are pointers!)
  for ( std::vector<AlignableDet*>::iterator iter = theDets.begin(); 
		iter != theDets.end(); iter++)
    delete *iter;

  delete theAlignmentPositionError;

}



// Return all components of the chamber (as Alignables)
std::vector<Alignable*> AlignableDTChamber::components() const 
{

  std::vector<Alignable*> result; 
  result.insert( result.end(), theDets.begin(), theDets.end() );
  return result;
  
}


// Return Alignable DT Chamber at given index
AlignableDet& AlignableDTChamber::det( int i )
{

  if (i >= size() ) 
	throw cms::Exception("LogicError") << "Det index (" << i << ") out of range";

  return *theDets[i];
  
}


//__________________________________________________________________________________________________
void AlignableDTChamber::setAlignmentPositionError(const AlignmentPositionError& ape)
{

  if ( !theAlignmentPositionError )
	theAlignmentPositionError = new AlignmentPositionError( ape );
  else 
	*theAlignmentPositionError = ape;

  AlignableComposite::setAlignmentPositionError( ape );

}


//__________________________________________________________________________________________________
void AlignableDTChamber::addAlignmentPositionError(const AlignmentPositionError& ape)
{

  if ( !theAlignmentPositionError ) this->setAlignmentPositionError( ape );
  else 
    this->setAlignmentPositionError( *theAlignmentPositionError += ape );

  AlignableComposite::addAlignmentPositionError( ape );

}

//__________________________________________________________________________________________________
void AlignableDTChamber::addAlignmentPositionErrorFromRotation(const RotationType& rot ) 
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
void AlignableDTChamber::addAlignmentPositionErrorFromLocalRotation(const RotationType& rot )
{

  RotationType globalRot = globalRotation().multiplyInverse(rot*globalRotation());
  this->addAlignmentPositionErrorFromRotation(globalRot);

  AlignableComposite::addAlignmentPositionErrorFromLocalRotation( rot );

}


//__________________________________________________________________________________________________
Alignments* AlignableDTChamber::alignments() const
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

  // Add components recursively
  std::vector<Alignable*> comp = this->components();
  for ( std::vector<Alignable*>::iterator i=comp.begin(); i!=comp.end(); i++ )
	{
	  Alignments* tmpAlignments = (*i)->alignments();
	  std::copy( tmpAlignments->m_align.begin(), tmpAlignments->m_align.end(), 
				 std::back_inserter(m_alignments->m_align) );
	}
  
  return m_alignments;

}


//__________________________________________________________________________________________________
AlignmentErrors* AlignableDTChamber::alignmentErrors( void ) const
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
  for ( std::vector<Alignable*>::iterator i=comp.begin(); i!=comp.end(); i++ )
	{
	  AlignmentErrors* tmpAlignmentErrors = (*i)->alignmentErrors();
	  std::copy( tmpAlignmentErrors->m_alignError.begin(), tmpAlignmentErrors->m_alignError.end(), 
				 std::back_inserter(m_alignmentErrors->m_alignError) );
	}
  
  return m_alignmentErrors;

}


/// Printout the Dets in the DT chamber
std::ostream &operator << ( std::ostream &os, const AlignableDTChamber & r )
{

  os << "    This DTChamber contains " << r.theDets.size() << " units" << std::endl ;
  os << "    position = " << r.globalPosition() << std::endl;
  os << "    (phi, r, z)= (" << r.globalPosition().phi() << "," 
     << r.globalPosition().perp() << "," << r.globalPosition().z();
  os << "), orientation:" << std::endl<< r.globalRotation() << std::endl;

  os << "    total displacement and rotation: " << r.displacement() << std::endl;
  os << r.rotation() << std::endl;

  for ( std::vector<AlignableDet*>::const_iterator idet = r.theDets.begin(); 
		idet != r.theDets.end(); idet++) 
	{
	  for ( int i=0; i<(*idet)->size();i++) 
		{
		  os << "     Det position, phi, r: " 
			 << (*idet)->detUnit(i).globalPosition() << " , "
			 << (*idet)->detUnit(i).globalPosition().phi() << " , "
			 << (*idet)->detUnit(i).globalPosition().perp() << std::endl; 
		  os << "     local  position, phi, r: " 
			 << r.surface().toLocal( (*idet)->detUnit(i).globalPosition() )        << " , "
			 << r.surface().toLocal( (*idet)->detUnit(i).globalPosition() ).phi()  << " , "
			 << r.surface().toLocal( (*idet)->detUnit(i).globalPosition() ).perp() << std::endl; 
		}
	}
  return os;

}
