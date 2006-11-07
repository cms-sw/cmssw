#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"

#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"


//__________________________________________________________________________________________________
AlignmentParameters::AlignmentParameters() :
  theAlignable(0),
  theUserVariables(0),
  bValid(true)
{}


//__________________________________________________________________________________________________
AlignmentParameters::AlignmentParameters(Alignable* object, const AlgebraicVector& par, 
					 const AlgebraicSymMatrix& cov) :
  theAlignable(object),
  theParameters(par),
  theCovariance(cov),
  theUserVariables(0),
  theSelector( size(), true ),
  bValid(true)
{

  if ( par.num_row() != cov.num_row() )
    throw cms::Exception("LogicError") << "@SUB=AlignmentParameters::AlignmentParameters "
                                       << "Size mismatch: parameter size " << par.num_row() 
                                       << ", covariance size " << cov.num_row() << ".";
}


//__________________________________________________________________________________________________
AlignmentParameters::AlignmentParameters(Alignable* object, const AlgebraicVector& par, 
                                         const AlgebraicSymMatrix& cov, 
                                         const std::vector<bool>& sel) :
  theAlignable(object),
  theParameters(par),
  theCovariance(cov),
  theUserVariables(0),
  theSelector(sel)
{  

  if ( (par.num_row() != cov.num_row()) || (par.num_row() != static_cast<int>(sel.size())) )
    throw cms::Exception("LogicError") << "@SUB=AlignmentParameters::AlignmentParameters "
                                       << "Size mismatch: parameter size " << par.num_row() 
                                       << ", covariance size " << cov.num_row()
                                       << ", selection size " << sel.size() << ".";
}


//__________________________________________________________________________________________________
AlignmentParameters::~AlignmentParameters()
{ 

  delete theUserVariables;

}


//__________________________________________________________________________________________________
const std::vector<bool>& AlignmentParameters::selector(void) const
{ 
  return theSelector;
}

//__________________________________________________________________________________________________
const int AlignmentParameters::numSelected(void) const
{

  int nsel=0;
  for ( int i=0; i<size(); i++ ) if ( theSelector[i] ) nsel++;
  return nsel;

}


//__________________________________________________________________________________________________
AlgebraicVector AlignmentParameters::selectedParameters(void) const
{ 

  AlgebraicVector selpar=collapseVector(theParameters,theSelector);
  return selpar;

}


//__________________________________________________________________________________________________
AlgebraicSymMatrix AlignmentParameters::selectedCovariance(void) const
{ 
  AlgebraicSymMatrix selcov=collapseSymMatrix( theCovariance, theSelector );
  return selcov;
}


//__________________________________________________________________________________________________
const AlgebraicVector& AlignmentParameters::parameters(void) const
{ 
  return theParameters;
}


//__________________________________________________________________________________________________
const AlgebraicSymMatrix& AlignmentParameters::covariance(void) const
{ 
  return theCovariance;
}


//__________________________________________________________________________________________________
void  AlignmentParameters::setUserVariables(AlignmentUserVariables* auv)
{ 
  delete theUserVariables;
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
const int AlignmentParameters::size(void) const
{ 
  return theParameters.num_row();
}


//__________________________________________________________________________________________________
const bool AlignmentParameters::isValid(void) const
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
