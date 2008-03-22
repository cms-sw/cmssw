
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUserVariables.h"
#include <boost/cstdint.hpp> 
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentDataCollector.h"

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"

using namespace std;

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
    static AlignableObjectId idMap;

    pair< int, int > typeAndLayer = alignableId->typeAndLayerFromDetId( parent->id() );

    int iType = typeAndLayer.first;
    int iLayer = typeAndLayer.second;
    align::ID iId = parent->id();

    string strName = idMap.typeToName( parent->alignableObjectId() ) + string( "_" );
    string strType = string( "Type" ) + toString( iType ) + string( "_" );
    string strLayer = string( "Layer" ) + toString( iLayer ) + string( "_" );
    string strId =  string( "Id" ) + toString( iId );

    theIdentifier = strType + strLayer + strName + strId;
  }
  else if ( parent )
  {
    theIdentifier = string( "Alignable_Id" ) + toString( parent->geomDetId().rawId() );
  }
  else theIdentifier = string( "NoAlignable" );
}


void KalmanAlignmentUserVariables::update( bool enforceUpdate )
{
  if ( theParentAlignable )
  {
    ++theNumberOfUpdates;

    if ( theNumberOfUpdates%theUpdateFrequency == 0 || theFirstUpdate || enforceUpdate )
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
	    KalmanAlignmentDataCollector::fillGraph( string("Delta") + parameterId, 0,
						     -trueParameters[i]/selectedScaling(i) );

	    KalmanAlignmentDataCollector::fillGraph( string("Sigma") + parameterId, 0,
						     sqrt(covariance[selected][selected])/selectedScaling(i) );
	  }
	  else
	  {
	    KalmanAlignmentDataCollector::fillGraph( string("Delta") + parameterId, theNumberOfUpdates,
						     (parameters[selected]-trueParameters[i])/selectedScaling(i) );

	    KalmanAlignmentDataCollector::fillGraph( string("Sigma") + parameterId, theNumberOfUpdates,
						     sqrt(covariance[selected][selected])/selectedScaling(i) );

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
  const AlignableSurface& surface = theParentAlignable->surface();

  // get global rotation
  const align::RotationType& globalRotation = theParentAlignable->rotation();
  // get local rotation
  align::RotationType localRotation = surface.toLocal( globalRotation );
  // get euler angles (local frame)
  align::EulerAngles localEulerAngles = align::toAngles( localRotation );

  // get global shifts
  align::GlobalVector globalShifts( globalRotation.multiplyInverse( theParentAlignable->displacement().basicVector() ) );
  // get local shifts
  align::LocalVector localShifts = surface.toLocal( globalShifts );

  AlgebraicVector trueParameters( 6 );
  trueParameters[0] = -localShifts.x();
  trueParameters[1] = -localShifts.y();
  trueParameters[2] = -localShifts.z();
  trueParameters[3] = -localEulerAngles[0];
  trueParameters[4] = -localEulerAngles[1];
  trueParameters[5] = -localEulerAngles[2];

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
