#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"

#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"


//__________________________________________________________________________________________________
AlignmentParameters::AlignmentParameters() :
  theAlignable( 0),
  theUserVariables( 0),
  bValid(true)
{}


//__________________________________________________________________________________________________
AlignmentParameters::AlignmentParameters(Alignable* object, const AlgebraicVector& par, 
					 const AlgebraicSymMatrix& cov) :
  theAlignable(object),
  theData( DataContainer( new AlignmentParametersData(par,cov) ) ),
  theUserVariables(0),
  bValid(true)
{
  // is the data consistent?
  theData->checkConsistency();
}


//__________________________________________________________________________________________________
AlignmentParameters::AlignmentParameters(Alignable* object, const AlgebraicVector& par, 
                                         const AlgebraicSymMatrix& cov, 
                                         const std::vector<bool>& sel) :
  theAlignable(object),
  theData( DataContainer( new AlignmentParametersData(par,cov,sel) ) ),
  theUserVariables(0),
  bValid(true)
{
  // is the data consistent?
  theData->checkConsistency();
}


//__________________________________________________________________________________________________
AlignmentParameters::AlignmentParameters(Alignable* object,
					 const AlignmentParametersData::DataContainer& data ) :
  theAlignable(object),
  theData(data),
  theUserVariables(0),
  bValid(true)
{
  // is the data consistent?
  theData->checkConsistency();
}


//__________________________________________________________________________________________________
AlignmentParameters::~AlignmentParameters()
{ 
  if ( theUserVariables ) delete theUserVariables;
}


//__________________________________________________________________________________________________
const std::vector<bool>& AlignmentParameters::selector(void) const
{ 
  return theData->selector();
}

//__________________________________________________________________________________________________
int AlignmentParameters::numSelected(void) const
{
  return theData->numSelected();
}


//__________________________________________________________________________________________________
AlgebraicVector AlignmentParameters::selectedParameters(void) const
{ 
  return collapseVector( theData->parameters(), theData->selector() );
}


//__________________________________________________________________________________________________
AlgebraicSymMatrix AlignmentParameters::selectedCovariance(void) const
{ 
  return collapseSymMatrix( theData->covariance(), theData->selector() );
}


//__________________________________________________________________________________________________
const AlgebraicVector& AlignmentParameters::parameters(void) const
{ 
  return theData->parameters();
}


//__________________________________________________________________________________________________
const AlgebraicSymMatrix& AlignmentParameters::covariance(void) const
{ 
  return theData->covariance();
}

//__________________________________________________________________________________________________
AlgebraicMatrix
AlignmentParameters::selectedDerivatives(const TrajectoryStateOnSurface& tsos,
					 const AlignableDetOrUnitPtr &alignableDet) const
{
  const AlgebraicMatrix dev(this->derivatives(tsos, alignableDet));

  const int ncols  = dev.num_col();
  const int nrows  = dev.num_row();
  const int nsel   = numSelected();

  AlgebraicMatrix seldev(nsel, ncols);

  int ir2 = 0;
  for (int irow = 0; irow < nrows; ++irow) {
    if (this->selector()[irow]) {
      for (int icol = 0; icol < ncols; ++icol) {
	seldev[ir2][icol] = dev[irow][icol];
      }
      ++ir2;
    }
  }

  return seldev;
}

//__________________________________________________________________________________________________
void  AlignmentParameters::setUserVariables(AlignmentUserVariables* auv)
{ 
  if ( theUserVariables ) delete theUserVariables;
  theUserVariables = auv;
}


//__________________________________________________________________________________________________
AlignmentUserVariables*  AlignmentParameters::userVariables(void) const
{ 
  return theUserVariables;
}


//__________________________________________________________________________________________________
Alignable* AlignmentParameters::alignable(void) const
{ 
  return theAlignable;
}

//__________________________________________________________________________________________________
unsigned int AlignmentParameters::hierarchyLevel() const
{
  if (!theAlignable) {
    edm::LogError("Alignment") << "@SUB=AlignmentParameters::hierarchyLevel"
			       << "Called for AlignmentParameters without pointer to Alignable";
    return 0;
  }

  std::vector<Alignable*> comps;
  theAlignable->firstCompsWithParams(comps);
  if (comps.empty()) return 0;

  unsigned int maxLevelOfComp = 0;
  for (std::vector<Alignable*>::const_iterator iAli = comps.begin(), iAliEnd = comps.end();
       iAli != iAliEnd; ++iAli) {// firstCompsWithParams guaranties that alignmentParameters() != 0:
    const unsigned int compResult = (*iAli)->alignmentParameters()->hierarchyLevel();
    // levels might be different for components, get largest:
    if (maxLevelOfComp < compResult) maxLevelOfComp = compResult;
  }

  return maxLevelOfComp + 1;
}


//__________________________________________________________________________________________________
int AlignmentParameters::size(void) const
{ 
  return theData->parameters().num_row();
}


//__________________________________________________________________________________________________
bool AlignmentParameters::isValid(void) const
{ 
  return bValid;
}


//__________________________________________________________________________________________________
void AlignmentParameters::setValid(bool v)
{ 
  bValid=v;
}


//__________________________________________________________________________________________________
AlgebraicSymMatrix 
AlignmentParameters::collapseSymMatrix(const AlgebraicSymMatrix& m,
                                       const std::vector<bool>& sel ) const
{

  int nRows = m.num_row();
  int size  = sel.size();

  // Check size matching
  if ( nRows != size ) 
    throw cms::Exception("LogicError") << "Size mismatch in parameters";

  // If OK, continue
  std::vector<int> rowvec;
  for ( int i=0; i<nRows; i++ ) 
    if ( sel[i] ) rowvec.push_back(i);
 
  int nSelectedRows = rowvec.size();
  AlgebraicSymMatrix result( nSelectedRows, 0 );
  for (int i=0; i<nSelectedRows; i++) 
    for (int j=0; j<nSelectedRows; j++)
      result[i][j] = m[ rowvec[i] ][ rowvec[j] ];

  return result;

}


//__________________________________________________________________________________________________
AlgebraicVector AlignmentParameters::collapseVector(const AlgebraicVector& m, 
                                                    const std::vector<bool>& sel ) const
{

  int nRows = m.num_row();
  int size  = sel.size();

  // Check size matching
  if ( nRows != size ) 
    throw cms::Exception("LogicError") << "Size mismatch in parameters";

  // If OK, continue
  std::vector<int> rowvec;
  for ( int i=0; i<nRows; i++ ) 
    if ( sel[i] ) rowvec.push_back(i);

  int nSelectedRows=rowvec.size();
  AlgebraicVector result( nSelectedRows, 0 );
  for ( int i=0; i<nSelectedRows; i++ )
    result[i] = m[ (int)rowvec[i] ];

  return result;

}


//__________________________________________________________________________________________________
AlgebraicSymMatrix AlignmentParameters::expandSymMatrix(const AlgebraicSymMatrix& m, 
                                                        const std::vector<bool>& sel) const
{

  int nRows = m.num_row();
  int size  = sel.size();

  std::vector<int> rowvec;
  for ( int i=0; i<size; i++ ) 
    if ( sel[i] ) rowvec.push_back(i);

  // Check size matching
  if( nRows != static_cast<int>(rowvec.size()) ) 
    throw cms::Exception("LogicError") << "Size mismatch in parameters";

  // If OK, continue
  AlgebraicSymMatrix result(size,0);
  for ( int i=0; i<nRows; i++ )
    for (int j=0; j<nRows; j++)
      result[ rowvec[i] ][ rowvec[j] ] = m[i][j];

  return result;
}


//__________________________________________________________________________________________________
AlgebraicVector AlignmentParameters::expandVector(const AlgebraicVector& m, 
                                                  const std::vector<bool>& sel) const
{

  int nRows = m.num_row();
  int size  = sel.size();

  std::vector<int> rowvec;
  for ( int i=0; i<size; i++ ) 
    if (sel[i]==true) rowvec.push_back(i);

  // Check size matching
  if( nRows != static_cast<int>(rowvec.size()) ) 
    throw cms::Exception("LogicError") << "Size mismatch in parameters";

  // If OK, continue
  AlgebraicVector result(size,0);
  for (int i=0; i<nRows; i++) result[ rowvec[i] ] = m[i];
  return result;

}
