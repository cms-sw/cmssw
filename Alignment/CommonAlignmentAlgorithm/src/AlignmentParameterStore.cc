/**
 * \file AlignmentParameterStore.cc
 *
 *  $Revision: 1.16 $
 *  $Date: 2007/05/15 17:56:15 $
 *  (last update by $Author: cklae $)
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/FrameToFrameDerivative.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentCorrelationsStore.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentExtendedCorrelationsStore.h"

// This class's header
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"

//__________________________________________________________________________________________________
AlignmentParameterStore::AlignmentParameterStore( const Alignables &alis,
						  const edm::ParameterSet& config ) :
  theAlignables(alis)
{
  if (config.getUntrackedParameter<bool>("UseExtendedCorrelations")) {
    theCorrelationsStore = new AlignmentExtendedCorrelationsStore
      (config.getParameter<edm::ParameterSet>("ExtendedCorrelationsConfig"));
  } else {
    theCorrelationsStore = new AlignmentCorrelationsStore();
  }

  theTrackerAlignableId = new TrackerAlignableId;

  edm::LogInfo("Alignment") << "@SUB=AlignmentParameterStore"
                            << "Created with " << theAlignables.size() << " alignables.";
}

//__________________________________________________________________________________________________
AlignmentParameterStore::~AlignmentParameterStore()
{
  delete theCorrelationsStore;
  delete theTrackerAlignableId;
}

//__________________________________________________________________________________________________
CompositeAlignmentParameters
AlignmentParameterStore::selectParameters( const std::vector<AlignableDet*>& alignabledets ) const
{
  std::vector<AlignableDetOrUnitPtr> detOrUnits;
  detOrUnits.reserve(alignabledets.size());

  std::vector<AlignableDet*>::const_iterator it, iEnd;
  for ( it = alignabledets.begin(), iEnd = alignabledets.end(); it != iEnd; ++it)
    detOrUnits.push_back(AlignableDetOrUnitPtr(*it));

  return this->selectParameters(detOrUnits);
}

//__________________________________________________________________________________________________
CompositeAlignmentParameters
AlignmentParameterStore::selectParameters( const std::vector<AlignableDetOrUnitPtr>& alignabledets ) const
{

  std::vector<Alignable*> alignables;
  std::map <AlignableDetOrUnitPtr,Alignable*> alidettoalimap;
  std::map <Alignable*,int> aliposmap;
  std::map <Alignable*,int> alilenmap;
  int nparam=0;

  // iterate over AlignableDet's
  for( std::vector<AlignableDetOrUnitPtr>::const_iterator iad = alignabledets.begin();
       iad != alignabledets.end(); ++iad ) 
  {
    Alignable* ali = alignableFromAlignableDet( *iad );
    if ( ali ) 
    {
      alidettoalimap[ *iad ] = ali; // Add to map
      // Check if Alignable already there, insert into vector if not
      if ( find(alignables.begin(),alignables.end(),ali) == alignables.end() ) 
      {
	alignables.push_back(ali);
	AlignmentParameters* ap = ali->alignmentParameters();
	nparam += ap->numSelected();
      }
    }
  }

  AlgebraicVector* selpar = new AlgebraicVector( nparam, 0 );
  AlgebraicSymMatrix* selcov = new AlgebraicSymMatrix( nparam, 0 );

  // Fill in parameters and corresponding covariance matricess
  int ipos = 1; // NOTE: .sub indices start from 1
  std::vector<Alignable*>::const_iterator it1;
  for( it1 = alignables.begin(); it1 != alignables.end(); ++it1 ) 
  {
    AlignmentParameters* ap = (*it1)->alignmentParameters();
    selpar->sub( ipos, ap->selectedParameters() );
    selcov->sub( ipos, ap->selectedCovariance() );
    int npar = ap->numSelected();
    aliposmap[*it1]=ipos;
    alilenmap[*it1]=npar;
    ipos +=npar;
  }

  // Fill in the correlations. Has to be an extra loop, because the
  // AlignmentExtendedCorrelationsStore (if used) needs the
  // alignables' covariance matrices already present.
  ipos = 1;
  for( it1 = alignables.begin(); it1 != alignables.end(); ++it1 ) 
  {
    int jpos=1;

    // Look for correlations between alignables
    std::vector<Alignable*>::const_iterator it2;
    for( it2 = alignables.begin(); it2 != it1; ++it2 ) 
    {
      theCorrelationsStore->correlations( *it1, *it2, *selcov, ipos-1, jpos-1 );
      jpos += (*it2)->alignmentParameters()->numSelected();
    }

    ipos += (*it1)->alignmentParameters()->numSelected();
  }

  AlignmentParametersData::DataContainer data( new AlignmentParametersData( selpar, selcov ) );
  CompositeAlignmentParameters aap( data, alignables, alidettoalimap, aliposmap, alilenmap );

  return aap;
}


//__________________________________________________________________________________________________
CompositeAlignmentParameters
AlignmentParameterStore::selectParameters( const std::vector<Alignable*>& alignables ) const
{

  std::vector<Alignable*> selectedAlignables;
  std::map <AlignableDetOrUnitPtr,Alignable*> alidettoalimap; // This map won't be filled!!!
  std::map <Alignable*,int> aliposmap;
  std::map <Alignable*,int> alilenmap;
  int nparam=0;

  // iterate over Alignable's
  std::vector<Alignable*>::const_iterator ita;
  for ( ita = alignables.begin(); ita != alignables.end(); ++ita ) 
  {
    // Check if Alignable already there, insert into vector if not
    if ( find(selectedAlignables.begin(), selectedAlignables.end(), *ita) == selectedAlignables.end() ) 
    {
      selectedAlignables.push_back( *ita );
      AlignmentParameters* ap = (*ita)->alignmentParameters();
      nparam += ap->numSelected();
    }
  }

  AlgebraicVector* selpar = new AlgebraicVector( nparam, 0 );
  AlgebraicSymMatrix* selcov = new AlgebraicSymMatrix( nparam, 0 );

  // Fill in parameters and corresponding covariance matrices
  int ipos = 1; // NOTE: .sub indices start from 1
  std::vector<Alignable*>::const_iterator it1;
  for( it1 = selectedAlignables.begin(); it1 != selectedAlignables.end(); ++it1 ) 
  {
    AlignmentParameters* ap = (*it1)->alignmentParameters();
    selpar->sub( ipos, ap->selectedParameters() );
    selcov->sub( ipos, ap->selectedCovariance() );
    int npar = ap->numSelected();
    aliposmap[*it1]=ipos;
    alilenmap[*it1]=npar;
    ipos +=npar;
  }

  // Fill in the correlations. Has to be an extra loop, because the
  // AlignmentExtendedCorrelationsStore (if used) needs the
  // alignables' covariance matrices already present.
  ipos = 1;
  for( it1 = selectedAlignables.begin(); it1 != selectedAlignables.end(); ++it1 ) 
  {
    int jpos=1;

    // Look for correlations between alignables
    std::vector<Alignable*>::const_iterator it2;
    for( it2 = selectedAlignables.begin(); it2 != it1; ++it2 ) 
    {
      theCorrelationsStore->correlations( *it1, *it2, *selcov, ipos-1, jpos-1 );
      jpos += (*it2)->alignmentParameters()->numSelected();
    }

    ipos += (*it1)->alignmentParameters()->numSelected();
  }

  AlignmentParametersData::DataContainer data( new AlignmentParametersData( selpar, selcov ) );
  CompositeAlignmentParameters aap( data, selectedAlignables, alidettoalimap, aliposmap, alilenmap );

  return aap;
}


//__________________________________________________________________________________________________
void AlignmentParameterStore::updateParameters( const CompositeAlignmentParameters& aap )
{

  std::vector<Alignable*> alignables = aap.components();
  const AlgebraicVector& parameters = aap.parameters();
  const AlgebraicSymMatrix& covariance = aap.covariance();

  int ipos = 1; // NOTE: .sub indices start from 1

  // Loop over alignables
  for( std::vector<Alignable*>::const_iterator it=alignables.begin(); it != alignables.end(); ++it ) 
  {
    // Update parameters and local covariance   
    AlignmentParameters* ap = (*it)->alignmentParameters();
    int nsel = ap->numSelected();
    AlgebraicVector subvec = parameters.sub( ipos, ipos+nsel-1 );
    AlgebraicSymMatrix subcov = covariance.sub( ipos, ipos+nsel-1 );
    AlignmentParameters* apnew = ap->cloneFromSelected( subvec, subcov );
    (*it)->setAlignmentParameters( apnew );
	  
    // Now update correlations between detectors
    int jpos = 1;
    for( std::vector<Alignable*>::const_iterator it2 = alignables.begin(); it2 != it; ++it2 ) 
    {
      theCorrelationsStore->setCorrelations( *it, *it2, covariance, ipos-1, jpos-1 );
      jpos += (*it2)->alignmentParameters()->numSelected();
    }

    ipos+=nsel;
  }

}


//__________________________________________________________________________________________________
std::vector<Alignable*> AlignmentParameterStore::validAlignables(void) const
{ 
  std::vector<Alignable*> result;
  for (std::vector<Alignable*>::const_iterator iali = theAlignables.begin();
       iali != theAlignables.end(); ++iali)
    if ( (*iali)->alignmentParameters()->isValid() ) result.push_back(*iali);

  LogDebug("Alignment") << "@SUB=AlignmentParameterStore::validAlignables"
                        << "Valid alignables: " << result.size()
                        << "out of " << theAlignables.size();
  return result;
}

//__________________________________________________________________________________________________
Alignable* AlignmentParameterStore::alignableFromAlignableDet( AlignableDetOrUnitPtr alignableDet ) const
{
  Alignable *mother = alignableDet;
  while (mother) {
    if (mother->alignmentParameters()) return mother;
    mother = mother->mother();
  }

  return 0;
}

//__________________________________________________________________________________________________
void AlignmentParameterStore::applyParameters(void)
{
  std::vector<Alignable*>::const_iterator iali;
  for ( iali = theAlignables.begin(); iali != theAlignables.end(); ++iali) 
    applyParameters( *iali );
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
  AlgebraicVector shift = ap->translation(); // fixme: should be LocalVector

  // Translation local->global
  align::LocalVector lv(shift[0], shift[1], shift[2]);
  alignable->move( alignable->surface().toGlobal(lv) );

  // Rotation in local frame
  align::EulerAngles angles = ap->rotation();
  alignable->rotateInLocalFrame( align::toMatrix(angles) );
}


//__________________________________________________________________________________________________
void AlignmentParameterStore::resetParameters(void)
{
  // Erase contents of correlation map
  theCorrelationsStore->resetCorrelations();

  // Iterate over alignables in the store and reset parameters
  std::vector<Alignable*>::const_iterator iali;
  for ( iali = theAlignables.begin(); iali != theAlignables.end(); ++iali )
    resetParameters( *iali );
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
      edm::LogError("BadArgument") << "@SUB=AlignmentParameterStore::resetParameters"
				   << "alignable has no alignment parameter";
  }
  else
    edm::LogError("BadArgument") << "@SUB=AlignmentParameterStore::resetParameters"
                                 << "argument is NULL";
}


//__________________________________________________________________________________________________
void AlignmentParameterStore::acquireRelativeParameters(void)
{

  unsigned int nAlignables = theAlignables.size();

  for (unsigned int i = 0; i < nAlignables; ++i)
  {
    Alignable* ali = theAlignables[i];

    RigidBodyAlignmentParameters* ap = 
      dynamic_cast<RigidBodyAlignmentParameters*>( ali->alignmentParameters() );

    if ( !ap )
      throw cms::Exception("BadAlignable") 
	<< "acquireRelativeParameters: "
	<< "provided alignable does not have rigid body alignment parameters";

    AlgebraicVector par( ap->size(),0 );
    AlgebraicSymMatrix cov( ap->size(), 0 );
	  
    // Get displacement and transform global->local
    align::LocalVector dloc = ali->surface().toLocal( ali->displacement() );
    par[0]=dloc.x();
    par[1]=dloc.y();
    par[2]=dloc.z();

    // Transform to local euler angles
    align::EulerAngles euloc = align::toAngles( ali->surface().toLocal( ali->rotation() ) );
    par[3]=euloc(1);
    par[4]=euloc(2);
    par[5]=euloc(3);
	  
    // Clone parameters
    RigidBodyAlignmentParameters* apnew = ap->clone(par,cov);
	  
    ali->setAlignmentParameters(apnew);
  }
}


//__________________________________________________________________________________________________
// Get type/layer from Alignable
// type: -6   -5   -4   -3   -2    -1     1     2    3    4    5    6
//      TEC- TOB- TID- TIB- PxEC- PxBR- PxBr+ PxEC+ TIB+ TID+ TOB+ TEC+
// Layers start from zero
std::pair<int,int> AlignmentParameterStore::typeAndLayer(const Alignable* ali) const
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
  for ( Alignables::const_iterator iali = alivec.begin(); iali != alivec.end(); ++iali ) 
  {
    Alignable* ali = *iali;
    unsigned int detId = theTrackerAlignableId->alignableId(ali);
    int typeId = theTrackerAlignableId->alignableTypeId(ali);

    // Find corresponding entry in AlignablePositions
    bool found=false;
    for ( AlignablePositions::const_iterator ipos = newpos.begin(); ipos != newpos.end(); ++ipos ) 
      if ( detId == ipos->id() && typeId == ipos->objId() ) 
	if ( found )
	  edm::LogError("DuplicatePosition")
	    << "New positions for alignable found more than once!";
	else
	{
	  // New position/rotation
	  const align::PositionType& pnew = ipos->pos();
	  const align::RotationType& rnew = ipos->rot();
	  // Current position / rotation
	  const align::PositionType& pold = ali->globalPosition();
	  const align::RotationType& rold = ali->globalRotation();
				
	  // shift needed to move from current to new position
	  align::GlobalVector posDiff = pnew - pold;
	  align::RotationType rotDiff = rold.multiplyInverse(rnew);
	  align::rectify(rotDiff); // correct for rounding errors 
	  ali->move( posDiff );
	  ali->rotateInGlobalFrame( rotDiff );
	  LogDebug("NewPosition") << "moving by:" << posDiff;
	  LogDebug("NewRotation") << "rotating by:\n" << rotDiff;

	  // add position error
	  // AlignmentPositionError ape(shift.x(),shift.y(),shift.z());
	  // (*iali)->addAlignmentPositionError(ape);
	  // (*iali)->addAlignmentPositionErrorFromRotation(rot);
				
	  found=true;
	  ++nappl;
	}
  }

  if ( nappl< newpos.size() )
    edm::LogError("Mismatch") << "Applied only " << nappl << " new positions" 
			      << " out of " << newpos.size();

  LogDebug("NewPositions") << "Applied new positions for " << nappl
                           << " out of " << alivec.size() <<" alignables.";

}


//__________________________________________________________________________________________________
void AlignmentParameterStore::
applyAlignableRelativePositions( const Alignables& alivec, const AlignableShifts& shifts, int& ierr )
{

  ierr=0;
  unsigned int nappl=0;
  unsigned int nAlignables = alivec.size();

  for (unsigned int i = 0; i < nAlignables; ++i)
  {
    Alignable* ali = alivec[i];

    unsigned int detId = theTrackerAlignableId->alignableId(ali);
    int typeId=theTrackerAlignableId->alignableTypeId(ali);

    // Find corresponding entry in AlignableShifts
    bool found = false;
    for ( AlignableShifts::const_iterator ipos = shifts.begin(); ipos != shifts.end(); ++ipos ) 
    {
      if ( detId == ipos->id() && typeId == ipos->objId() ) 
	if ( found )
	  edm::LogError("DuplicatePosition")
	    << "New positions for alignable found more than once!";
	else
	{
	  ali->move( ipos->pos() );
	  ali->rotateInGlobalFrame( ipos->rot() );
				
	  // Add position error
	  //AlignmentPositionError ape(pnew.x(),pnew.y(),pnew.z());
	  //ali->addAlignmentPositionError(ape);
	  //ali->addAlignmentPositionErrorFromRotation(rnew);

	  found=true;
	  ++nappl;
	}
    }
  }
  
  if ( nappl < shifts.size() )
    edm::LogError("Mismatch") << "Applied only " << nappl << " new positions" 
			      << " out of " << shifts.size();

  LogDebug("NewPositions") << "Applied new positions for " << nappl << " alignables.";
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
  for ( Alignables::const_iterator iali = alivec.begin(); iali != alivec.end(); ++iali ) 
  {
    // Iterate over Parameters
    bool found=false;
    for ( Parameters::const_iterator ipar = parvec.begin(); ipar != parvec.end(); ++ipar) 
    {
      // Get new alignment parameters
      RigidBodyAlignmentParameters* ap = dynamic_cast<RigidBodyAlignmentParameters*>(*ipar); 

      // Check if parameters belong to alignable 
      if ( ap->alignable() == (*iali) )
      {
	if (!found) 
	{
          (*iali)->setAlignmentParameters(ap);
          ++ipass;
          found=true;
        } 
        else edm::LogError("DuplicateParameters") << "More than one parameters for Alignable";
      }
    }
    if (!found) ++ifail;
  }
  if (ifail>0) ierr=-1;
  
  LogDebug("attachAlignmentParameters") << " Parameters, Alignables: " << parvec.size() << ","
                                        << alivec.size() << "\n pass,fail: " << ipass << ","<< ifail;
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
                                                  bool overwrite, int& ierr )
{
  ierr=0;
  int icount=0;

  // Iterate over correlations
  for ( Correlations::const_iterator icor = cormap.begin(); icor!=cormap.end(); ++icor ) 
  {
    AlgebraicMatrix mat=(*icor).second;
    Alignable* ali1 = (*icor).first.first;
    Alignable* ali2 = (*icor).first.second;

    // Check if alignables exist
    if ( find( alivec.begin(), alivec.end(), ali1 ) != alivec.end() && 
         find( alivec.begin(), alivec.end(), ali2 ) != alivec.end() )
    {
      // Check if correlations already existing between these alignables
      if ( !theCorrelationsStore->correlationsAvailable(ali1,ali2) || (overwrite) ) 
       {
         theCorrelationsStore->setCorrelations(ali1,ali2,mat);
         ++icount;
       }
      else edm::LogInfo("AlreadyExists") << "Correlation existing and not overwritten";
    }
    else edm::LogInfo("IgnoreCorrelation") << "Ignoring correlation with no alignables!";
  }

  LogDebug( "attachCorrelations" ) << " Alignables,Correlations: " << alivec.size() <<","<< cormap.size() 
                                   << "\n applied: " << icount ;

}


//__________________________________________________________________________________________________
void AlignmentParameterStore::
attachUserVariables( const Alignables& alivec,
                     const std::vector<AlignmentUserVariables*>& uvarvec, int& ierr )
{
  ierr=0;

  LogDebug("DumpArguments") << "size of alivec:   "  << alivec.size()
                            << "\nsize of uvarvec: " << uvarvec.size();

  std::vector<AlignmentUserVariables*>::const_iterator iuvar=uvarvec.begin();

  for ( Alignables::const_iterator iali=alivec.begin(); iali!=alivec.end(); ++iali, ++iuvar ) 
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
  unsigned int nAlignables = alivec.size();

  for (unsigned int i = 0; i < nAlignables; ++i)
  {
    Alignable* ali = alivec[i];

    // First reset APE	 
    AlignmentPositionError nulApe(0,0,0);	 
    ali->setAlignmentPositionError(nulApe);

    // Set APE from displacement
    AlignmentPositionError ape(valshift,valshift,valshift);
    if ( valshift > 0. ) ali->addAlignmentPositionError(ape);
    else ali->setAlignmentPositionError(ape);

    // Set APE from rotation
    align::EulerAngles r(3);
    r(1)=valrot; r(2)=valrot; r(3)=valrot;
    ali->addAlignmentPositionErrorFromRotation( align::toMatrix(r) );
  }

  LogDebug("StoreAPE") << "Store APE from shift: " << valshift;
  LogDebug("StoreAPE") << "Store APE from rotation: " << valrot;
}

//__________________________________________________________________________________________________
bool AlignmentParameterStore
::hierarchyConstraints(const Alignable *ali, const Alignables &aliComps,
		       std::vector<std::vector<ParameterId> > &paramIdsVecOut,
		       std::vector<std::vector<float> > &factorsVecOut,
		       float epsilon) const
{
  // Weak point:
  // Ignores constraints between non-subsequent levels in case the parameter is not considered in
  // the intermediate level, e.g. global z for dets and layers is aligned, but not for rods!
  if (!ali || !ali->alignmentParameters()) return false;

  const std::vector<bool> &aliSel= ali->alignmentParameters()->selector();
  paramIdsVecOut.clear();
  factorsVecOut.clear();
  FrameToFrameDerivative f2fDerivMaker;

  bool firstComp = true;
  for (Alignables::const_iterator iComp = aliComps.begin(), iCompE = aliComps.end();
       iComp != iCompE; ++iComp) {
    const AlgebraicMatrix f2fDeriv(f2fDerivMaker.frameToFrameDerivative(*iComp, ali));
    const std::vector<bool> &aliCompSel = (*iComp)->alignmentParameters()->selector();
    for (unsigned int iParMast = 0, iParMastUsed = 0; iParMast < aliSel.size(); ++iParMast) {
      if (!aliSel[iParMast]) continue; // nothing to constrain if no parameter at higher level
      if (firstComp) { // fill output with empty arrays 
	paramIdsVecOut.push_back(std::vector<ParameterId>());
	factorsVecOut.push_back(std::vector<float>());
      }
      for (int iParComp = 0; iParComp < f2fDeriv.num_col(); ++iParComp) {
// 	if (aliCompSel[iParMast] && aliCompSel[iParComp]) {
	if (aliCompSel[iParComp]) {
	  const float factor = f2fDeriv[iParMast][iParComp]; // switch col/row? GF: Should be fine.
	  if (fabs(factor) > epsilon) {
	    paramIdsVecOut[iParMastUsed].push_back(ParameterId(*iComp, iParComp));
	    factorsVecOut[iParMastUsed].push_back(factor);
	  }
	}
      }
      ++iParMastUsed;
    }
    firstComp = false;
  } // end loop on components

  return true;
}
