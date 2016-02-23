/**
 * \file AlignmentParameterStore.cc
 *
 *  $Revision: 1.31 $
 *  $Date: 2011/05/23 20:50:32 $
 *  (last update by $Author: mussgill $)
 */

// This class's header should be first
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/ParametersToParametersDerivatives.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentExtendedCorrelationsStore.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

//__________________________________________________________________________________________________
AlignmentParameterStore::AlignmentParameterStore( const align::Alignables &alis,
						  const edm::ParameterSet& config ) :
  theAlignables(alis)
{
  if (config.getUntrackedParameter<bool>("UseExtendedCorrelations")) {
    theCorrelationsStore = new AlignmentExtendedCorrelationsStore
      (config.getParameter<edm::ParameterSet>("ExtendedCorrelationsConfig"));
  } else {
    theCorrelationsStore = new AlignmentCorrelationsStore();
  }

  edm::LogInfo("Alignment") << "@SUB=AlignmentParameterStore"
                            << "Created with " << theAlignables.size() << " alignables.";

  // set hierarchy vs averaging constraints
  theTypeOfConstraints = NONE;
  const std::string cfgStrTypeOfConstraints(config.getParameter<std::string>("TypeOfConstraints"));
  if( cfgStrTypeOfConstraints == "hierarchy" ) {
    theTypeOfConstraints = HIERARCHY_CONSTRAINTS;
  } else if( cfgStrTypeOfConstraints == "approximate_averaging" ) {
    theTypeOfConstraints = APPROX_AVERAGING_CONSTRAINTS;
    edm::LogWarning("Alignment") << "@SUB=AlignmentParameterStore"
				 << "\n\n\n******* WARNING ******************************************\n"
				 << "Using approximate implementation of averaging constraints."
				 << "This is not recommended."
				 << "Consider to use 'hierarchy' constraints:"
				 << "  AlignmentProducer.ParameterStore.TypeOfConstraints = cms.string('hierarchy')\n\n\n";
  } else {
    edm::LogError("BadArgument") << "@SUB=AlignmentParameterStore"
				 << "Unknown type of hierarchy constraints '" << cfgStrTypeOfConstraints << "'"; 
  }
}

//__________________________________________________________________________________________________
AlignmentParameterStore::~AlignmentParameterStore()
{
  delete theCorrelationsStore;
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

  align::Alignables alignables;
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
  align::Alignables::const_iterator it1;
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
    align::Alignables::const_iterator it2;
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
AlignmentParameterStore::selectParameters( const align::Alignables& alignables ) const
{

  align::Alignables selectedAlignables;
  std::map <AlignableDetOrUnitPtr,Alignable*> alidettoalimap; // This map won't be filled!!!
  std::map <Alignable*,int> aliposmap;
  std::map <Alignable*,int> alilenmap;
  int nparam=0;

  // iterate over Alignable's
  align::Alignables::const_iterator ita;
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
  align::Alignables::const_iterator it1;
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
    align::Alignables::const_iterator it2;
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
void AlignmentParameterStore::updateParameters( const CompositeAlignmentParameters& aap, bool updateCorrelations )
{

  align::Alignables alignables = aap.components();
  const AlgebraicVector& parameters = aap.parameters();
  const AlgebraicSymMatrix& covariance = aap.covariance();

  int ipos = 1; // NOTE: .sub indices start from 1

  // Loop over alignables
  for( align::Alignables::const_iterator it=alignables.begin(); it != alignables.end(); ++it ) 
  {
    // Update parameters and local covariance   
    AlignmentParameters* ap = (*it)->alignmentParameters();
    int nsel = ap->numSelected();
    AlgebraicVector subvec = parameters.sub( ipos, ipos+nsel-1 );
    AlgebraicSymMatrix subcov = covariance.sub( ipos, ipos+nsel-1 );
    AlignmentParameters* apnew = ap->cloneFromSelected( subvec, subcov );
    (*it)->setAlignmentParameters( apnew );
	  
    // Now update correlations between detectors
    if ( updateCorrelations )
    {
      int jpos = 1;
      for( align::Alignables::const_iterator it2 = alignables.begin(); it2 != it; ++it2 ) 
      {
	theCorrelationsStore->setCorrelations( *it, *it2, covariance, ipos-1, jpos-1 );
	jpos += (*it2)->alignmentParameters()->numSelected();
      }
    }

    ipos+=nsel;
  }

}


//__________________________________________________________________________________________________
align::Alignables AlignmentParameterStore::validAlignables(void) const
{ 
  align::Alignables result;
  for (align::Alignables::const_iterator iali = theAlignables.begin();
       iali != theAlignables.end(); ++iali)
    if ( (*iali)->alignmentParameters()->isValid() ) result.push_back(*iali);

  LogDebug("Alignment") << "@SUB=AlignmentParameterStore::validAlignables"
                        << "Valid alignables: " << result.size()
                        << "out of " << theAlignables.size();
  return result;
}

//__________________________________________________________________________________________________
Alignable* AlignmentParameterStore::alignableFromAlignableDet( const AlignableDetOrUnitPtr& _alignableDet ) const
{
  AlignableDetOrUnitPtr alignableDet = _alignableDet;
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
  align::Alignables::const_iterator iali;
  for ( iali = theAlignables.begin(); iali != theAlignables.end(); ++iali) 
    applyParameters( *iali );
}


//__________________________________________________________________________________________________
void AlignmentParameterStore::applyParameters(Alignable* alignable)
{

  AlignmentParameters *pars = (alignable ? alignable->alignmentParameters() : 0);
  if (!pars) {
    throw cms::Exception("BadAlignable") 
      << "applyParameters: provided alignable does not have alignment parameters";
  }
  pars->apply();
}


//__________________________________________________________________________________________________
void AlignmentParameterStore::resetParameters(void)
{
  // Erase contents of correlation map
  theCorrelationsStore->resetCorrelations();

  // Iterate over alignables in the store and reset parameters
  align::Alignables::const_iterator iali;
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
void AlignmentParameterStore::cacheTransformations(void)
{
  align::Alignables::const_iterator iali;
  for ( iali = theAlignables.begin(); iali != theAlignables.end(); ++iali) 
    (*iali)->cacheTransformation();
}


//__________________________________________________________________________________________________
void AlignmentParameterStore::restoreCachedTransformations(void)
{
  align::Alignables::const_iterator iali;
  for ( iali = theAlignables.begin(); iali != theAlignables.end(); ++iali) 
    (*iali)->restoreCachedTransformation();
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
std::pair<int,int> AlignmentParameterStore::typeAndLayer(const Alignable* ali, const TrackerTopology* tTopo) const
{
  return TrackerAlignableId().typeAndLayerFromDetId( ali->id(), tTopo );
}


//__________________________________________________________________________________________________
void AlignmentParameterStore::
applyAlignableAbsolutePositions( const align::Alignables& alivec, 
                                 const AlignablePositions& newpos, 
                                 int& ierr )
{
  unsigned int nappl=0;
  ierr=0;

  // Iterate over list of alignables
  for (align::Alignables::const_iterator iali = alivec.begin(); iali != alivec.end(); ++iali) { 
    Alignable* ali = *iali;
    align::ID id = ali->id();
    align::StructureType typeId = ali->alignableObjectId();

    // Find corresponding entry in AlignablePositions
    bool found=false;
    for (AlignablePositions::const_iterator ipos = newpos.begin(); ipos != newpos.end(); ++ipos) {
      if (id == ipos->id() && typeId == ipos->objId()) {
	if (found) {
	  edm::LogError("DuplicatePosition")
	    << "New positions for alignable found more than once!";
	} else {
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
applyAlignableRelativePositions( const align::Alignables& alivec, const AlignableShifts& shifts, int& ierr )
{

  ierr=0;
  unsigned int nappl=0;
  unsigned int nAlignables = alivec.size();

  for (unsigned int i = 0; i < nAlignables; ++i) {
    Alignable* ali = alivec[i];

    align::ID id = ali->id();
    align::StructureType typeId = ali->alignableObjectId();

    // Find corresponding entry in AlignableShifts
    bool found = false;
    for (AlignableShifts::const_iterator ipos = shifts.begin(); ipos != shifts.end(); ++ipos) {
      if (id == ipos->id() && typeId == ipos->objId()) {
	if (found) {
	  edm::LogError("DuplicatePosition")
	    << "New positions for alignable found more than once!";
	} else {
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
void AlignmentParameterStore::attachAlignmentParameters( const align::Alignables& alivec, 
                                                         const Parameters& parvec, int& ierr )
{
  int ipass = 0;
  int ifail = 0;
  ierr = 0;

  // Iterate over alignables
  for ( align::Alignables::const_iterator iali = alivec.begin(); iali != alivec.end(); ++iali ) 
  {
    // Iterate over Parameters
    bool found=false;
    for ( Parameters::const_iterator ipar = parvec.begin(); ipar != parvec.end(); ++ipar) 
    {
      // Get new alignment parameters
      AlignmentParameters* ap = *ipar; 

      // Check if parameters belong to alignable 
      if ( ap->alignable() == (*iali) )
      {
	if (!found) 
	{
          (*iali)->setAlignmentParameters(ap);
          ++ipass;
          found=true;
        } 
        else edm::LogError("Alignment") << "@SUB=AlignmentParameterStore::attachAlignmentParameters" 
					<< "More than one parameters for Alignable.";
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
void AlignmentParameterStore::attachCorrelations( const align::Alignables& alivec, 
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
attachUserVariables( const align::Alignables& alivec,
                     const std::vector<AlignmentUserVariables*>& uvarvec, int& ierr )
{
  ierr=0;

  LogDebug("DumpArguments") << "size of alivec:   "  << alivec.size()
                            << "\nsize of uvarvec: " << uvarvec.size();

  std::vector<AlignmentUserVariables*>::const_iterator iuvar=uvarvec.begin();

  for ( align::Alignables::const_iterator iali=alivec.begin(); iali!=alivec.end(); ++iali, ++iuvar ) 
  {
    AlignmentParameters* ap = (*iali)->alignmentParameters();
    AlignmentUserVariables* uvarnew = (*iuvar);
    ap->setUserVariables(uvarnew);
  }
}


//__________________________________________________________________________________________________
void AlignmentParameterStore::setAlignmentPositionError( const align::Alignables& alivec, 
                                                         double valshift, double valrot )
{
  unsigned int nAlignables = alivec.size();

  for (unsigned int i = 0; i < nAlignables; ++i)
  {
    Alignable* ali = alivec[i];

    // First reset APE	 
    AlignmentPositionError nulApe(0,0,0);	 
    ali->setAlignmentPositionError(nulApe, true);

    // Set APE from displacement
    AlignmentPositionError ape(valshift,valshift,valshift);
    if ( valshift > 0. ) ali->addAlignmentPositionError(ape, true);
    else ali->setAlignmentPositionError(ape, true);
    // GF: Resetting and setting as above does not really make sense to me, 
    //     and adding to zero or setting is the same! I'd just do 
    //ali->setAlignmentPositionError(AlignmentPositionError ape(valshift,valshift,valshift),true);

    // Set APE from rotation
    align::EulerAngles r(3);
    r(1)=valrot; r(2)=valrot; r(3)=valrot;
    ali->addAlignmentPositionErrorFromRotation(align::toMatrix(r), true);
  }

  LogDebug("StoreAPE") << "Store APE from shift: " << valshift
		       << "\nStore APE from rotation: " << valrot;
}

//__________________________________________________________________________________________________
bool AlignmentParameterStore
::hierarchyConstraints(const Alignable *ali, const align::Alignables &aliComps,
		       std::vector<std::vector<ParameterId> > &paramIdsVecOut,
		       std::vector<std::vector<double> > &factorsVecOut,
		       bool all, double epsilon) const
{
  // Weak point if all = false:
  // Ignores constraints between non-subsequent levels in case the parameter is not considered in
  // the intermediate level, e.g. global z for dets and layers is aligned, but not for rods!
  if (!ali || !ali->alignmentParameters()) return false;

  const std::vector<bool> &aliSel= ali->alignmentParameters()->selector();
  paramIdsVecOut.clear();
  factorsVecOut.clear();

  bool firstComp = true;
  for (align::Alignables::const_iterator iComp = aliComps.begin(), iCompE = aliComps.end();
       iComp != iCompE; ++iComp) {

    const ParametersToParametersDerivatives p2pDerivs(**iComp, *ali);
    if (!p2pDerivs.isOK()) {
      throw cms::Exception("BadConfig")
	<< "AlignmentParameterStore::hierarchyConstraints"
	<< " Bad match of types of AlignmentParameters classes.\n";
      return false;
    }
    const std::vector<bool> &aliCompSel = (*iComp)->alignmentParameters()->selector();
    for (unsigned int iParMast = 0, iParMastUsed = 0; iParMast < aliSel.size(); ++iParMast) {
      if (!all && !aliSel[iParMast]) continue;// no higher level parameter & constraint deselected
      if (firstComp) { // fill output with empty arrays 
	paramIdsVecOut.push_back(std::vector<ParameterId>());
	factorsVecOut.push_back(std::vector<double>());
      }
      for (unsigned int iParComp = 0; iParComp < aliCompSel.size(); ++iParComp) {
	if (aliCompSel[iParComp]) {
	  double factor = 0.;
	  if( theTypeOfConstraints == HIERARCHY_CONSTRAINTS ) {
	    // hierachy constraints
	    factor = p2pDerivs(iParMast, iParComp);
	  } else if( theTypeOfConstraints == APPROX_AVERAGING_CONSTRAINTS ) {
	    // CHK poor mans averaging constraints
	    factor = p2pDerivs(iParMast, iParComp);
	    if (iParMast < 3 && (iParComp % 9) >= 3) factor = 0.;
	  }
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
