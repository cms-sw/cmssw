
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUserVariables.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/DataCollector.h"

#include "Alignment/CommonAlignmentParametrization/interface/AlignmentTransformations.h"

using namespace alignmentservices;


KalmanAlignmentUserVariables::KalmanAlignmentUserVariables( Alignable* parent,
							    TrackerAlignableId* alignableId,
							    int frequency ) :
    theParentAlignable( parent ),
    theNumberOfHits( 0 ),
    theNumberOfUpdates( 0 ),
    theUpdateFrequency( frequency ),
    theFirstUpdate( true )
{
  if ( parent && alignableId )
  {
    pair< int, int > typeAndLayer = alignableId->typeAndLayerFromAlignable( parent );

    int iType = typeAndLayer.first;
    int iLayer = typeAndLayer.second;
    int iId = alignableId->alignableId( parent );

    string strName = alignableId->alignableTypeName( parent ) + string( "_" );
    string strType = string( "Type" ) + toString( iType ) + string( "_" );
    string strLayer = string( "Layer" ) + toString( iLayer ) + string( "_" );
    string strId =  string( "Id" ) + toString( iId );

    theIdentifier = strName + strType + strLayer + strId;
  }
  else if ( parent )
  {
    theIdentifier = string( "Alignable_Id" ) + toString( parent->geomDetId().rawId() );
  }
  else theIdentifier = string( "NoAlignable" );
}


void KalmanAlignmentUserVariables::update( void )
{
  if ( theParentAlignable )
  {
    ++theNumberOfUpdates;

    if ( ( ( theNumberOfUpdates % theUpdateFrequency == 0  ) || theFirstUpdate ) && DataCollector::isAvailable() )
    {
      AlgebraicVector parameters = theParentAlignable->alignmentParameters()->selectedParameters();
      AlgebraicSymMatrix covariance = theParentAlignable->alignmentParameters()->selectedCovariance();
      vector< bool > selector = theParentAlignable->alignmentParameters()->selector();

      AlgebraicVector trueParameters = extractTrueParameters();

      const int nParameter = 6;
      int selected = 0;
      
      for ( int i = 0; i < nParameter; ++i )
      {
	if ( selector[i] )
	{
	  string parameterId = selectedParameter( i ) + theIdentifier;

	  if ( theFirstUpdate )
	  {
	    DataCollector::fillGraph( string("Delta") + parameterId, 0, -trueParameters[i]/selectedScaling(i) );
	    DataCollector::fillGraph( string("Sigma") + parameterId, 0, sqrt(covariance[selected][selected])/selectedScaling(i) );
	  }
	  else
	  {
	    DataCollector::fillGraph( string("Delta") + parameterId, theNumberOfUpdates, (parameters[selected]-trueParameters[i])/selectedScaling(i) );
	    DataCollector::fillGraph( string("Sigma") + parameterId, theNumberOfUpdates, sqrt(covariance[selected][selected])/selectedScaling(i) );
	  }
	  selected++;
	}
      }

      if ( theFirstUpdate ) theFirstUpdate = false;
    }
  }
}

const AlgebraicVector KalmanAlignmentUserVariables::extractTrueParameters( void ) const
{
  // get surface of alignable
  const AlignableSurface surface = theParentAlignable->surface();

  AlignmentTransformations TkAT;
  // get euler angles (global frame)
  AlgebraicVector globalEulerAngles = TkAT.eulerAngles( theParentAlignable->rotation(), 0 );
  // get euler angles (local frame)
  AlgebraicVector localEulerAngles = TkAT.globalToLocalEulerAngles( globalEulerAngles, surface.rotation() );

  // get local displacement
  LocalVector localDisplacement = surface.toLocal( theParentAlignable->displacement() );
  // get shifts (take local rotation into account)
  Surface::RotationType localRotation = TkAT.rotationType( TkAT.rotMatrix3( localEulerAngles ) );
  LocalVector localShifts( localRotation.multiplyInverse( localDisplacement.basicVector() ) );

  AlgebraicVector trueParameters( 6 );
  trueParameters[0] = localShifts.x();
  trueParameters[1] = localShifts.y();
  trueParameters[2] = localShifts.z();
  trueParameters[3] = localEulerAngles[0];
  trueParameters[4] = localEulerAngles[1];
  trueParameters[5] = localEulerAngles[2];

  return trueParameters;
}


const string KalmanAlignmentUserVariables::selectedParameter( const int& selected ) const
{
  switch ( selected )
  {
  case 0:
    return string( "X_" );
    break;
  case 1:
    return string( "Y_" );
    break;
  case 2:
    return string( "Z_" );
    break;
  case 3:
    return string( "Alpha_" );
    break;
  case 4:
    return string( "Beta_" );
    break;
  case 5:
    return string( "Gamma_" );
    break;
  default:
    cout << "[KalmanAlignmentUserVariables::selectedParameter] Index out of range (selector = " << selected << ")" << endl;
    return toString( selected ) + string( "_" );
    break;
  }

  return string( "YouShouldNeverEverSeeThis_" );
}


const float KalmanAlignmentUserVariables::selectedScaling( const int& selected ) const
{
  const float micron = 1e-4;
  const float millirad = 1e-3;

  switch ( selected )
  {
  case 0:
  case 1:
  case 2:
    return micron;
    break;
  case 3:
  case 4:
  case 5:
    return millirad;
    break;
  default:
    cout << "[KalmanAlignmentUserVariables::selectedScaling] Index out of range (selector = " << selected << ")" << endl;
    return 1.;
    break;
  }

  return 1.;
}


const string KalmanAlignmentUserVariables::toString( const int& i ) const
{
  char temp[10];
  sprintf( temp, "%u", i );

  return string( temp );
}
