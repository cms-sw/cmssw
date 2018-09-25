#include "Alignment/CommonAlignment/interface/AlignmentParametersData.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
#include <functional>


AlignmentParametersData::AlignmentParametersData( void ) :
  theParameters( new AlgebraicVector() ),
  theCovariance( new AlgebraicSymMatrix() ),
  theSelector( new std::vector<bool>() ),
  theNumSelected( 0 )
{}


AlignmentParametersData::AlignmentParametersData( AlgebraicVector* param,
						  AlgebraicSymMatrix* cov,
						  std::vector<bool>* sel ) :
  theParameters( param ),
  theCovariance( cov ),
  theSelector( sel )
{
  theNumSelected = std::count_if( theSelector->begin(),
				  theSelector->end(),
				  [](auto const &c){return c == true;});
}


AlignmentParametersData::AlignmentParametersData( const AlgebraicVector& param,
						  const AlgebraicSymMatrix& cov,
						  const std::vector<bool>& sel ) :
  theParameters( new AlgebraicVector( param ) ),
  theCovariance( new AlgebraicSymMatrix( cov ) ),
  theSelector( new std::vector<bool>( sel ) )
{
  theNumSelected = std::count_if( theSelector->begin(),
				  theSelector->end(),
				  [](auto const &c){return c == true;});
}


AlignmentParametersData::AlignmentParametersData( AlgebraicVector* param,
						  AlgebraicSymMatrix* cov ) :
  theParameters( param ),
  theCovariance( cov ),
  theSelector( new std::vector<bool>( param->num_row(), true ) ),
  theNumSelected( param->num_row() )
{}


AlignmentParametersData::AlignmentParametersData( const AlgebraicVector& param,
						  const AlgebraicSymMatrix& cov ) :
  theParameters( new AlgebraicVector( param ) ),
  theCovariance( new AlgebraicSymMatrix( cov ) ),
  theSelector( new std::vector<bool>( param.num_row(), true ) ),
  theNumSelected( param.num_row() )
{}


AlignmentParametersData::~AlignmentParametersData( void )
{
  delete theParameters;
  delete theCovariance;
  delete theSelector;
}


void AlignmentParametersData::checkConsistency( void ) const
{
  int selectorSize = static_cast<int>( theSelector->size() );
  int paramSize = theParameters->num_row();
  int covSize = theCovariance->num_row();

  if ( ( paramSize != covSize ) || ( paramSize != selectorSize ) )
      throw cms::Exception("LogicError") << "@SUB=AlignmentParametersData::checkConsistency "
					 << "\nSize mismatch: parameter size = " << paramSize
					 << ", covariance size = " << covSize
					 << ", selector size = " << selectorSize << ".";
}
