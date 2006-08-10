#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/CommonAlignmentParametrization/interface/KarimakiAlignmentDerivatives.h"
#include "Alignment/CommonAlignmentParametrization/interface/AlignmentTransformations.h"

#include <string>

// This class's header
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"


//__________________________________________________________________________________________________
AlignmentParameterStore::AlignmentParameterStore( std::vector<Alignable*> alivec ) :
  theAlignables(alivec)
{


  theNavigator = new AlignableNavigator( alivec );
  theTrackerAlignableId = new TrackerAlignableId;

  // Fill detId <-> Alignable map
  for ( std::vector<Alignable*>::iterator it = alivec.begin();
		it != alivec.end(); it++ )
	{
	  DetIds tmpDetIds = findDetIds(*it);
	  for ( DetIds::iterator iDetId = tmpDetIds.begin();
			iDetId != tmpDetIds.end(); iDetId++ )
		theActiveAlignablesByDetId[ *iDetId ] = *it;
	}

  edm::LogInfo("AlignmentParameterStore") << "Created navigator map with "
										  << theNavigator->size() << " elements";
  

}


//__________________________________________________________________________________________________
CompositeAlignmentParameters 
AlignmentParameterStore::selectParameters( const std::vector<AlignableDet*>& alignabledets ) const
{

  std::vector<Alignable*> alignables;
  std::map <AlignableDet*,Alignable*> alidettoalimap;
  std::map <Alignable*,int> aliposmap;
  std::map <Alignable*,int> alilenmap;
  int nparam=0;

  // iterate over AlignableDet's
  for(std::vector<AlignableDet*>::const_iterator 
		iad=alignabledets.begin(); iad!=alignabledets.end(); iad++ ) 
	{

	  unsigned int detId = (*iad)->geomDetId().rawId();
	  Alignable* ali = alignableFromDetId( detId );
	  if ( ali ) 
		{
		  alidettoalimap[ *iad ]=ali; // Add to map
		  // Check if Alignable already there, insert into vector if not
		  if ( find(alignables.begin(),alignables.end(),ali) == alignables.end() ) 
			{
			  alignables.push_back(ali);
			  AlignmentParameters* ap = ali->alignmentParameters();
			  nparam += ap->numSelected();
			}
		}
	}

  AlgebraicVector selpar( nparam, 0 );
  AlgebraicSymMatrix selcov( nparam, 0 );

  int ipos=1; // Position within selpar,selcov; starts from 1!
 
  // Now, run through again and fill parameters
  for(std::vector<Alignable*>::const_iterator 
		it=alignables.begin(); it!=alignables.end(); it++) 
	{
	  AlignmentParameters* ap = (*it)->alignmentParameters();
	  AlgebraicVector thisselpar = ap->selectedParameters();
	  AlgebraicSymMatrix thisselcov = ap->selectedCovariance();
	  int npar = thisselpar.num_row();
	  selpar.sub(ipos,thisselpar);
	  selcov.sub(ipos,thisselcov);
	  // Look for correlations between alignables
	  int jpos=1;
	  for( std::vector<Alignable*>::const_iterator it2 = alignables.begin(); 
		   it2 != it; it2++ ) 
		{
		  AlignmentParameters* ap2 = (*it2)->alignmentParameters();
		  int npar2=ap2->selectedParameters().num_row();
		  AlgebraicMatrix covmat = correlations(*it,*it2);
		  if (covmat.num_row()>0)
			for (int i=0;i<npar;i++)
			  for (int j=0;j<npar2;j++)
				selcov[(ipos-1)+i][(jpos-1)+j]=covmat[i][j];
		  jpos +=npar2;
		}
    aliposmap[*it]=ipos;
    alilenmap[*it]=npar;
    ipos +=npar;
  }

  CompositeAlignmentParameters aap( selpar, selcov, alignables, alidettoalimap,
									aliposmap, alilenmap );
  return aap;

}


//__________________________________________________________________________________________________
void AlignmentParameterStore::updateParameters( const CompositeAlignmentParameters& aap )
{

  std::vector<Alignable*> alignables = aap.components();
  AlgebraicVector parameters = aap.parameters();
  AlgebraicSymMatrix covariance = aap.covariance();

  int ipar=1; // NOTE: .sub indices start from 1

  // Loop over alignables
  for( std::vector<Alignable*>::const_iterator it=alignables.begin(); 
	   it != alignables.end(); it++ ) 
	{
	  AlignmentParameters* ap =(*it)->alignmentParameters();
	  int nsel=ap->numSelected();
	  // Update parameters and local covariance   
	  AlgebraicVector subvec=parameters.sub(ipar,ipar+nsel-1);
	  AlgebraicSymMatrix subcov=covariance.sub(ipar,ipar+nsel-1);
	  AlignmentParameters* apnew = ap->cloneFromSelected(subvec,subcov);
	  (*it)->setAlignmentParameters(apnew);
	  
	  // Now update correlations between detectors
	  int ipar2=1;
	  for( std::vector<Alignable*>::const_iterator it2 = alignables.begin(); 
		   it2 != it; it2++ ) 
		{
		  AlignmentParameters* ap2 =(*it2)->alignmentParameters();
		  int nsel2=ap2->numSelected();
		  AlgebraicMatrix suboffcov(nsel,nsel2);
		  for (int i=0;i<nsel;i++)
			for (int j=0;j<nsel2;j++)
			  suboffcov[i][j]=covariance[(ipar-1)+i][(ipar2-1)+j];
		
		  // Need to develop mechanism to control when to add correlation ...
		  if ( true )
			setCorrelations(*it,*it2,suboffcov);

		  ipar2 += nsel2;
		}
	  ipar+=nsel;
	}

}


//__________________________________________________________________________________________________
std::vector<Alignable*> AlignmentParameterStore::validAlignables(void) const
{ 

  std::vector<Alignable*> result;
  for (std::vector<Alignable*>::const_iterator iali = theAlignables.begin();
	   iali != theAlignables.end(); iali++)
	if ( (*iali)->alignmentParameters()->isValid() ) result.push_back(*iali);

  edm::LogInfo("AlignmentParameterStore") << "Valid alignables: " << result.size()
										  << "out of " << theAlignables.size();
  return result;

}


//__________________________________________________________________________________________________
AlgebraicMatrix AlignmentParameterStore::correlations( Alignable* ap1, Alignable* ap2 ) const
{

  bool transpose = false;
  if (ap2<ap1) 
	{
	  std::swap(ap1,ap2); 
	  transpose = true; 
	}

  AlgebraicMatrix matrix;
  Correlations::const_iterator ic = theCorrelations.find( std::make_pair(ap1,ap2) );
  if (ic != theCorrelations.end()) 
	if ( transpose ) matrix = (*ic).second.T();
	else matrix = (*ic).second;

  return matrix;

}


//__________________________________________________________________________________________________
void AlignmentParameterStore::setCorrelations(Alignable* ap1, Alignable* ap2, 
											  const AlgebraicMatrix& mat )
{
  
  AlgebraicMatrix mat2;
  if (ap2<ap1) 
	{ 
	  std::swap(ap1,ap2); 
	  mat2=mat.T(); 
	}
  else 
	mat2=mat;
  
  theCorrelations[ std::make_pair(ap1,ap2) ] = mat2;

}



//__________________________________________________________________________________________________
Alignable* 
AlignmentParameterStore::alignableFromGeomDet( const GeomDet* geomDet ) const
{
  return alignableFromDetId( geomDet->geographicalId().rawId() );
}


//__________________________________________________________________________________________________
Alignable* 
AlignmentParameterStore::alignableFromAlignableDet( const AlignableDet* alignableDet ) const
{
  return alignableFromDetId( alignableDet->geomDetId().rawId() );
}


//__________________________________________________________________________________________________
Alignable* 
AlignmentParameterStore::alignableFromDetId( const unsigned int& detId ) const
{

  ActiveAlignablesByDetIdMap::const_iterator iali = theActiveAlignablesByDetId.find( detId );
  if ( iali != theActiveAlignablesByDetId.end() ) 
	return (*iali).second;
  else return 0;

}


//__________________________________________________________________________________________________
AlignmentParameterStore::DetIds 
AlignmentParameterStore::findDetIds(Alignable* alignable)
{

  DetIds result;
  AlignableDet* alidet = dynamic_cast<AlignableDet*>( alignable );
  if (alidet !=0) result.push_back( alignable->geomDetId().rawId() );
  std::vector<Alignable*> comp = alignable->components();
  if ( comp.size() > 1 )
	for ( std::vector<Alignable*>::const_iterator ib = comp.begin(); ib != comp.end(); ib++ ) 
	  {
		DetIds tmpDetIds = findDetIds(*ib);
		std::copy( tmpDetIds.begin(), tmpDetIds.end(), std::back_inserter( result ) );
	  }
  return result;

}


//__________________________________________________________________________________________________
std::vector<AlignableDet*> 
AlignmentParameterStore::alignableDetsFromHits( const std::vector<TrackingRecHit>& hitvec )
{

  std::vector<AlignableDet*> alidetvec;
  for ( std::vector<TrackingRecHit>::const_iterator ih=hitvec.begin();
		ih!=hitvec.end(); ih++ ) 
	{
	  AlignableDet* aliDet 
		= dynamic_cast<AlignableDet*>( theNavigator->alignableFromDetId(ih->geographicalId()) );
	  if ( aliDet )
		alidetvec.push_back( aliDet );
	  else
		throw cms::Exception("BadAssociation") << "Couldn't find AlignableDet"
											   << " associated to hit";
	}
  return alidetvec;

}


//__________________________________________________________________________________________________
void AlignmentParameterStore::applyParameters(void)
{

  for (std::vector<Alignable*>::const_iterator iali = theAlignables.begin();
	   iali != theAlignables.end(); iali++) 
    applyParameters(*iali);

}


//__________________________________________________________________________________________________
void AlignmentParameterStore::applyParameters(Alignable* alignable)
{

  // Get alignment parameters
  RigidBodyAlignmentParameters* ap = 
	dynamic_cast<RigidBodyAlignmentParameters*>( alignable->alignmentParameters() );

  if ( !ap )
	throw cms::Exception("BadAlignable") 
	  << "applyParameters: provided alignable does not have rigid body alignment parameters";

  // Translation in local frame
  AlgebraicVector shift = ap->translation();

  // Translation local->global
  LocalPoint l0 = Local3DPoint( 0.0,  0.0, 0.0);
  LocalPoint l1 = Local3DPoint(shift[0], shift[1], shift[2]);
  GlobalPoint g0 = alignable->surface().toGlobal( l0 );
  GlobalPoint g1 = alignable->surface().toGlobal( l1 );
  GlobalVector dg = g1-g0;
  alignable->move(dg);

  // Rotation in local frame
  AlgebraicVector rota = ap->rotation();
  if ( fabs(rota[0]) > 1e-5 || fabs(rota[1]) > 1e-5 || fabs(rota[2]) > 1e-5 ) 
	{
	  AlignmentTransformations alignTransform;
	  Surface::RotationType rot  = alignTransform.rotationType( alignTransform.rotMatrix3(rota) );
	  Surface::RotationType rot2 = 
		alignTransform.localToGlobalMatrix( rot, 
											alignable->globalRotation() );
	  alignable->rotateInGlobalFrame(rot2);
	}

}


//__________________________________________________________________________________________________
void AlignmentParameterStore::resetParameters(void)
{

  // Erase contents of correlation map
  theCorrelations.erase(theCorrelations.begin(),theCorrelations.end());

  // Iterate over alignables in the store and reset parameters
  for ( std::vector<Alignable*>::const_iterator iali = theAlignables.begin();
		iali != theAlignables.end(); iali++ )
    resetParameters(*iali);

}


//__________________________________________________________________________________________________
void AlignmentParameterStore::resetParameters( Alignable* ali )
{
  if ( ali ) 
	{
	  // Get alignment parameters for this alignable
	  AlignmentParameters* ap = ali->alignmentParameters();
	  if ( ap ) 
		{
		  int npar=ap->numSelected();
		
		  AlgebraicVector par(npar,0);
		  AlgebraicSymMatrix cov(npar,0);
		  AlignmentParameters* apnew = ap->cloneFromSelected(par,cov);
		  ali->setAlignmentParameters(apnew);
		  apnew->setValid(false);
		}
	  else 
		edm::LogError("BadArgument") 
		  << "resetParameters: alignable has no alignment parameter";
	}
  else
	edm::LogError("BadArgument") << "resetParameters argument is NULL";

}


//__________________________________________________________________________________________________
void AlignmentParameterStore::acquireRelativeParameters(void)
{

  AlignmentTransformations alignTransform;
  for ( std::vector<Alignable*>::const_iterator iali = theAlignables.begin();
		iali != theAlignables.end(); iali++) 
	{
	  RigidBodyAlignmentParameters* ap = 
		dynamic_cast<RigidBodyAlignmentParameters*>( (*iali)->alignmentParameters() );
	  if ( !ap )
		throw cms::Exception("BadAlignable") 
		  << "acquireRelativeParameters: "
		  << "provided alignable does not have rigid body alignment parameters";

	  AlgebraicVector par( ap->size(),0 );
	  AlgebraicSymMatrix cov( ap->size(), 0 );
	  
	  // Get displacement and transform global->local
	  LocalVector dloc = (*iali)->surface().toLocal( (*iali)->displacement() );
	  par[0]=dloc.x();
	  par[1]=dloc.y();
	  par[2]=dloc.z();
	  
	  // Global rel rotation
	  Surface::RotationType rot = (*iali)->rotation();
	  // Global abs rotation
	  Surface::RotationType detrot = (*iali)->surface().rotation();

	  // Global euler angles
	  AlgebraicVector euglob = alignTransform.eulerAngles( rot,0 );

	  // Transform to local euler angles
	  AlgebraicVector euloc = alignTransform.globalToLocalEulerAngles( euglob, detrot );
	  par[3]=euloc[0];
	  par[4]=euloc[1];
	  par[5]=euloc[2];
	  
	  // Clone parameters
	  RigidBodyAlignmentParameters* apnew = ap->clone(par,cov);
	  
	  (*iali)->setAlignmentParameters(apnew);
	}

}


//__________________________________________________________________________________________________
// Get type/layer from Alignable
// type: -6   -5   -4   -3   -2    -1     1     2    3    4    5    6
//      TEC- TOB- TID- TIB- PxEC- PxBR- PxBr+ PxEC+ TIB+ TID+ TOB+ TEC+
// Layers start from zero
std::pair<int,int> AlignmentParameterStore::typeAndLayer(Alignable* ali)
{
  return theTrackerAlignableId->typeAndLayerFromAlignable( ali );
}


//__________________________________________________________________________________________________
void AlignmentParameterStore::
applyAlignableAbsolutePositions( const Alignables& alivec, 
								 const AlignablePositions& newpos, 
								 int& ierr )
{
  unsigned int nappl=0;
  ierr=0;

  // Iterate over list of alignables
  for ( Alignables::const_iterator iali = alivec.begin(); 
		iali != alivec.end(); iali++ ) 
	{
	  Alignable* ali = *iali;
	  unsigned int detId = theTrackerAlignableId->alignableId(ali);
	  int typeId = theTrackerAlignableId->alignableTypeId(ali);

	  // Find corresponding entry in AlignablePositions
	  bool found=false;
	  for ( AlignablePositions::const_iterator ipos = newpos.begin();
		    ipos != newpos.end(); ipos++ ) 
		if ( detId == ipos->id() && typeId == ipos->objId() ) 
		  if ( found )
			edm::LogError("DuplicatePosition")
			  << "New positions for alignable found more than once!";
		  else
			{
			  // New position/rotation
			  GlobalPoint pnew = ipos->pos();
			  Surface::RotationType rnew = ipos->rot();
			  // Current position / rotation
			  GlobalPoint pold = ali->surface().position();
			  Surface::RotationType rold = ali->surface().rotation();
				
			  // shift needed to move from current to new position
			  GlobalVector shift = pnew - pold;
			  ali->move( shift );
			  edm::LogInfo("NewPosition") << "moving by" << shift;
				
			  // Delta-rotation needed to rotate from current to new rotation
			  int ierr;
			  AlignmentTransformations alignTransform;
			  Surface::RotationType rot = 
				alignTransform.rotationType(alignTransform.algebraicMatrix(rold).inverse(ierr)) 
				* rnew;
			  if ( ierr )
				edm::LogError("InversionError") << "Matrix inversion failed: not rotating";
			  else
				{ 
				  // 'Repair' matrix for rounding errors 
				  Surface::RotationType rotfixed = alignTransform.rectify(rot);
				  ali->rotateInGlobalFrame(rotfixed);
				  AlgebraicMatrix mrot = alignTransform.algebraicMatrix( rotfixed );
				  edm::LogInfo("NewRotation") << "rotating by: " << mrot;
				}
				
			  // add position error
			  // AlignmentPositionError ape(shift.x(),shift.y(),shift.z());
			  // (*iali)->addAlignmentPositionError(ape);
			  // (*iali)->addAlignmentPositionErrorFromRotation(rot);
				
			  found=true;
			  nappl++;
			}
	}

  if ( nappl< newpos.size() )
	edm::LogError("Mismatch") << "Applied only " << nappl << " new positions" 
							  << " out of " << newpos.size();

  edm::LogInfo("NewPositions") << "Applied new positions for " << nappl
							   << " out of " << alivec.size() <<" alignables.";

}


//__________________________________________________________________________________________________
void AlignmentParameterStore::
applyAlignableRelativePositions( const Alignables& alivec, 
								 const AlignableShifts& shifts, int& ierr )
{

  unsigned int nappl=0;
  ierr=0;

  // Iterate over list of alignables
  for ( Alignables::const_iterator iali = alivec.begin(); 
		iali != alivec.end(); iali++) 
	{

	  unsigned int detId = theTrackerAlignableId->alignableId( *iali );
	  int typeId=theTrackerAlignableId->alignableTypeId( *iali );

	  // Find corresponding entry in AlignableShifts
	  bool found = false;
	  for ( AlignableShifts::const_iterator ipos = shifts.begin();
			ipos != shifts.end(); ipos++ ) 
		{
		  if ( detId == ipos->id() && typeId == ipos->objId() ) 
			if ( found )
			  edm::LogError("DuplicatePosition")
				<< "New positions for alignable found more than once!";
			else
			  {
				// New position/rotation shift
				GlobalVector pnew = ipos->pos();
				Surface::RotationType rnew = ipos->rot();
				
				(*iali)->move(pnew);
				(*iali)->rotateInGlobalFrame(rnew);
				
				// Add position error
				//AlignmentPositionError ape(pnew.x(),pnew.y(),pnew.z());
				//(*iali)->addAlignmentPositionError(ape);
				//(*iali)->addAlignmentPositionErrorFromRotation(rnew);
				
				found=true;
				nappl++;
			  }
		}
	}
  
  if ( nappl < shifts.size() )
	edm::LogError("Mismatch") << "Applied only " << nappl << " new positions" 
							  << " out of " << shifts.size();

  edm::LogInfo("NewPositions") 
	<< "Applied new positions for " << nappl << " alignables.";

}



//__________________________________________________________________________________________________
void AlignmentParameterStore::attachAlignmentParameters( const Parameters& parvec, int& ierr )
{
  attachAlignmentParameters( theAlignables, parvec, ierr);
}



//__________________________________________________________________________________________________
void AlignmentParameterStore::attachAlignmentParameters( const Alignables& alivec, 
														 const Parameters& parvec, int& ierr )
{

  int ipass = 0;
  int ifail = 0;
  ierr = 0;

  // Iterate over alignables
  for ( Alignables::const_iterator iali = alivec.begin();
		iali != alivec.end(); iali++ ) 
	{
	  // Iterate over Parameters
	  bool found=false;
	  for ( Parameters::const_iterator ipar = parvec.begin();
			ipar != parvec.end(); ipar++) 
		{
		  // Get new alignment parameters
		  RigidBodyAlignmentParameters* ap =
			dynamic_cast<RigidBodyAlignmentParameters*>(*ipar); 

		  // Check if parameters belong to alignable 
		  if ( ap->alignable() == (*iali) )
			{
			  if (!found) 
				{
				  (*iali)->setAlignmentParameters(ap);
				  ipass++;
				  found=true;
				} 
			  else 
				edm::LogError("DuplicateParameters")
				  <<"More than one parameters for Alignable";
			}
		}
	  if (!found) ifail++;
	}
  if (ifail>0) ierr=-1;
  
  edm::LogInfo("attachAlignmentParameters") 
	<< " Parameters, Alignables: "<< parvec.size() <<"," << alivec.size()
	<< "\n pass,fail: " << ipass <<","<< ifail;
}


//__________________________________________________________________________________________________
void AlignmentParameterStore::attachCorrelations( const Correlations& cormap, 
												  bool overwrite, int& ierr )
{
  attachCorrelations( theAlignables, cormap, overwrite, ierr );
}


//__________________________________________________________________________________________________
void AlignmentParameterStore::attachCorrelations( const Alignables& alivec, 
												  const Correlations& cormap, 
												  bool overwrite,int& ierr )
{

  ierr=0;
  int icount=0;

  // Iterate over correlations
  for ( Correlations::const_iterator icor = cormap.begin(); 
		icor!=cormap.end(); icor++ ) 
	{
	  AlgebraicMatrix mat=(*icor).second;
	  Alignable* ali1 = (*icor).first.first;
	  Alignable* ali2 = (*icor).first.second;

	  // Check if alignables exist
	  if ( find(alivec.begin(),alivec.end(),ali1) != alivec.end() && 
		   find(alivec.begin(),alivec.end(),ali2) != alivec.end() )
		// Check if correlations already existing between these alignables
		if ( correlations(ali1,ali2).num_row() == 0 || (overwrite) ) 
		  {
			setCorrelations(ali1,ali2,mat);
			icount++;
		  }
		else 
		  edm::LogWarning("AlreadyExists") 
			<< "Correlation existing and not overwritten";
	  else 
		edm::LogWarning("IgnoreCorrelation") 
		  << "Ignoring correlation with no alignables!";
	}

  edm::LogInfo("attachCorrelations") 
	<< " Alignables,Correlations: " << alivec.size() <<","<< cormap.size() 
	<< "\n applied: " << icount ;

}


//__________________________________________________________________________________________________
void AlignmentParameterStore::
attachUserVariables( const Alignables& alivec,
					 const std::vector<AlignmentUserVariables*>& uvarvec, int& ierr )
{

  ierr=0;

  edm::LogInfo("DumpArguments") << "size of alivec:   "  << alivec.size()
								<< "\nsize of uvarvec: " << uvarvec.size();

  std::vector<AlignmentUserVariables*>::const_iterator iuvar=uvarvec.begin();

  for ( Alignables::const_iterator iali=alivec.begin(); iali!=alivec.end();
		iali++, iuvar++ ) 
	{
	  AlignmentParameters* ap = (*iali)->alignmentParameters();
	  AlignmentUserVariables* uvarnew = (*iuvar);
	  ap->setUserVariables(uvarnew);
	}

}


//__________________________________________________________________________________________________
void AlignmentParameterStore::setAlignmentPositionError( const Alignables& alivec, 
														 double valshift, double valrot )
{

  bool first=true;
  for ( Alignables::const_iterator iali = alivec.begin(); 
		iali != alivec.end(); iali++ ) 
	{
	  if (valshift>0) {
		AlignmentPositionError ape(valshift,valshift,valshift);
		(*iali)->addAlignmentPositionError(ape);
		if (first)
		  edm::LogInfo("StoreAPE") << "Store APE from shift: " << valshift;
	  }
	  if (valrot>0) {
		AlignmentTransformations alignTransform;
		AlgebraicVector r(3);
		r[0]=valrot; r[1]=valrot; r[2]=valrot;
		Surface::RotationType aperot 
		  = alignTransform.rotationType( alignTransform.rotMatrix3(r) );
		(*iali)->addAlignmentPositionErrorFromRotation(aperot);
		if (first) 
		  edm::LogInfo("StoreAPE") << "Store APE from rotation: " << valrot;
	  }
	  first=false;
	}
}
