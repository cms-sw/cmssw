
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUserVariables.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentDataCollector.h"

#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

// Uncomment to plot the evolution of the alignment
// parameters in the local rather than the global frame.
//#define USE_LOCAL_PARAMETERS


const TrackerAlignableId* KalmanAlignmentUserVariables::theAlignableId = new TrackerAlignableId;

KalmanAlignmentUserVariables::KalmanAlignmentUserVariables( Alignable* parent,
                                                            const TrackerTopology* tTopo,
							    int frequency ) :
    theParentAlignable( parent ),
    theNumberOfHits( 0 ),
    theNumberOfUpdates( 0 ),
    theUpdateFrequency( frequency ),
    theFirstUpdate( true ),
    theAlignmentFlag( false )
{
  if ( parent )
  {
    pair< int, int > typeAndLayer = theAlignableId->typeAndLayerFromDetId( parent->geomDetId(), tTopo );

    int iType = typeAndLayer.first;
    int iLayer = typeAndLayer.second;
    int iId = parent->id();

    string strName = AlignableObjectId::idToString( parent->alignableObjectId() ) + string( "_" );
    string strType = string( "Type" ) + toString( iType ) + string( "_" );
    string strLayer = string( "Layer" ) + toString( iLayer ) + string( "_" );
    string strId =  string( "Id" ) + toString( iId );

    theTypeAndLayer = strType + strLayer;
    theIdentifier = theTypeAndLayer + strName + strId;
    
  }
  else theIdentifier = string( "NoAlignable" );
}


//KalmanAlignmentUserVariables::~KalmanAlignmentUserVariables( void ) {}


void KalmanAlignmentUserVariables::update( bool enforceUpdate )
{
  if ( theParentAlignable )
  {
    ++theNumberOfUpdates;

    if ( ( ( theNumberOfUpdates % theUpdateFrequency == 0  ) || enforceUpdate ) )
    {

#ifdef USE_LOCAL_PARAMETERS

      const AlgebraicVector parameters = theParentAlignable->alignmentParameters()->selectedParameters();
      const AlgebraicSymMatrix covariance = theParentAlignable->alignmentParameters()->selectedCovariance();
      const vector< bool >& selector = theParentAlignable->alignmentParameters()->selector();

      AlgebraicVector trueParameters( extractTrueParameters() );

      const int nParameter = 6;
      int selected = 0;
      
      for ( int i = 0; i < nParameter; ++i )
      {
	if ( selector[i] )
	{
	  string parameterId = selectedParameter( i ) + string( "_" ) + theIdentifier;

	  if ( theFirstUpdate )
	  {
	    KalmanAlignmentDataCollector::fillGraph( string("LocalDelta") + parameterId, 0, -trueParameters[i]/selectedScaling(i) );
	    KalmanAlignmentDataCollector::fillGraph( string("LocalSigma") + parameterId, 0, sqrt(covariance[selected][selected])/selectedScaling(i) );
	  }

	  KalmanAlignmentDataCollector::fillGraph( string("LocalDelta") + parameterId, theNumberOfUpdates/theUpdateFrequency, (parameters[selected]-trueParameters[i])/selectedScaling(i) );
	  KalmanAlignmentDataCollector::fillGraph( string("LocalSigma") + parameterId, theNumberOfUpdates/theUpdateFrequency, sqrt(covariance[selected][selected])/selectedScaling(i) );

	  selected++;
	}
      }

      if ( theFirstUpdate ) theFirstUpdate = false;

#else

      const AlgebraicVector& parameters = theParentAlignable->alignmentParameters()->parameters();
      const AlgebraicSymMatrix& covariance = theParentAlignable->alignmentParameters()->covariance();

      const AlignableSurface& surface = theParentAlignable->surface();

      // Get global euler angles.
      align::EulerAngles localEulerAngles( parameters.sub( 4, 6 ) );
      const align::RotationType localRotation = align::toMatrix( localEulerAngles );
      const align::RotationType globalRotation = surface.toGlobal( localRotation );
      align::EulerAngles globalEulerAngles = align::toAngles( globalRotation );

      // Get global shifts.
      align::LocalVector localShifts( parameters[0], parameters[1], parameters[2] );
      align::GlobalVector globalShifts( surface.toGlobal( localShifts ) );

      const int nParameter = 6;
      AlgebraicVector globalParameters( nParameter );
      globalParameters[0] = globalShifts.x();
      globalParameters[1] = globalShifts.y();
      globalParameters[2] = globalShifts.z();
      globalParameters[3] = globalEulerAngles[0];
      globalParameters[4] = globalEulerAngles[1];
      globalParameters[5] = globalEulerAngles[2];

      AlgebraicVector trueParameters( extractTrueParameters() );
      
      for ( int i = 0; i < nParameter; ++i )
      {
	string parameterId = selectedParameter( i ) + string( "_" ) + theIdentifier;

	if ( theFirstUpdate )
	{
	  KalmanAlignmentDataCollector::fillGraph( string("GlobalDelta") + parameterId, 0, -trueParameters[i]/selectedScaling(i) );
	  KalmanAlignmentDataCollector::fillGraph( string("LocalSigma") + parameterId, 0, sqrt(covariance[i][i])/selectedScaling(i) );
	}

	KalmanAlignmentDataCollector::fillGraph( string("GlobalDelta") + parameterId, theNumberOfUpdates/theUpdateFrequency, (globalParameters[i]-trueParameters[i])/selectedScaling(i) );
	KalmanAlignmentDataCollector::fillGraph( string("LocalSigma") + parameterId, theNumberOfUpdates/theUpdateFrequency, sqrt(covariance[i][i])/selectedScaling(i) );
      }

      if ( theFirstUpdate ) theFirstUpdate = false;

#endif

    }
  }
}


void KalmanAlignmentUserVariables::update( const AlignmentParameters* param )
{
  if ( theParentAlignable )
  {
    ++theNumberOfUpdates;

    const AlgebraicVector& parameters = param->selectedParameters();
    const AlgebraicSymMatrix& covariance = param->selectedCovariance();
    const vector< bool >& selector = param->selector();

    AlgebraicVector trueParameters( extractTrueParameters() );

    const int nParameter = 6;
    int selected = 0;
      
    for ( int i = 0; i < nParameter; ++i )
    {
      if ( selector[i] )
      {
	string parameterId = selectedParameter( i ) + string( "_" ) + theIdentifier;

	KalmanAlignmentDataCollector::fillGraph( string("Delta") + parameterId, theNumberOfUpdates/theUpdateFrequency, (parameters[selected]-trueParameters[i])/selectedScaling(i) );
	KalmanAlignmentDataCollector::fillGraph( string("Sigma") + parameterId, theNumberOfUpdates/theUpdateFrequency, sqrt(covariance[selected][selected])/selectedScaling(i) );

	selected++;
      }
    }

    if ( theFirstUpdate ) theFirstUpdate = false;
  }
}


void KalmanAlignmentUserVariables::histogramParameters( string histoNamePrefix )
{
  if ( theParentAlignable )
  {

#ifdef USE_LOCAL_PARAMETERS

    AlgebraicVector parameters = theParentAlignable->alignmentParameters()->selectedParameters();
    AlgebraicSymMatrix covariance = theParentAlignable->alignmentParameters()->selectedCovariance();
    vector< bool > selector = theParentAlignable->alignmentParameters()->selector();

    AlgebraicVector trueParameters = extractTrueParameters();

    const int nParameter = 6;
    int selected = 0;

    //histoNamePrefix += theTypeAndLayer;
      
    for ( int i = 0; i < nParameter; ++i )
    {
      if ( selector[i] )
      {
	string startHistoName = histoNamePrefix + theTypeAndLayer + string( "_Start" ) + selectedParameter( i );
	KalmanAlignmentDataCollector::fillHistogram( startHistoName, -trueParameters[i]/selectedScaling(i) );

	string deltaHistoName = histoNamePrefix + theTypeAndLayer + string( "_Delta" ) + selectedParameter( i );
	KalmanAlignmentDataCollector::fillHistogram( deltaHistoName, (parameters[selected]-trueParameters[i])/selectedScaling(i) );

	string pullsHistoName = histoNamePrefix + theTypeAndLayer + string( "_Pulls" ) + selectedParameter( i );
	KalmanAlignmentDataCollector::fillHistogram( pullsHistoName, (parameters[selected]-trueParameters[i])/sqrt(covariance[selected][selected]) );

	startHistoName = histoNamePrefix + string( "_Start" ) + selectedParameter( i );
	KalmanAlignmentDataCollector::fillHistogram( startHistoName, -trueParameters[i]/selectedScaling(i) );

	deltaHistoName = histoNamePrefix + string( "_Delta" ) + selectedParameter( i );
	KalmanAlignmentDataCollector::fillHistogram( deltaHistoName, (parameters[selected]-trueParameters[i])/selectedScaling(i) );

	pullsHistoName = histoNamePrefix + string( "_Pulls" ) + selectedParameter( i );
	KalmanAlignmentDataCollector::fillHistogram( pullsHistoName, (parameters[selected]-trueParameters[i])/sqrt(covariance[selected][selected]) );

	selected++;
      }
    }

#else

    const AlgebraicVector& parameters = theParentAlignable->alignmentParameters()->parameters();

    const AlignableSurface& surface = theParentAlignable->surface();

    // Get global euler angles.
    align::EulerAngles localEulerAngles( parameters.sub( 4, 6 ) );
    const align::RotationType localRotation = align::toMatrix( localEulerAngles );
    const align::RotationType globalRotation = surface.toGlobal( localRotation );
    align::EulerAngles globalEulerAngles = align::toAngles( globalRotation );

    // Get global shifts.
    align::LocalVector localShifts( parameters[0], parameters[1], parameters[2] );
    align::GlobalVector globalShifts( surface.toGlobal( localShifts ) );

    const int nParameter = 6;
    AlgebraicVector globalParameters( nParameter );
    globalParameters[0] = globalShifts.x();
    globalParameters[1] = globalShifts.y();
    globalParameters[2] = globalShifts.z();
    globalParameters[3] = globalEulerAngles[0];
    globalParameters[4] = globalEulerAngles[1];
    globalParameters[5] = globalEulerAngles[2];

    AlgebraicVector trueParameters( extractTrueParameters() );

    KalmanAlignmentDataCollector::fillGraph( "y_vs_dx", theParentAlignable->globalPosition().y(), trueParameters[0]-globalParameters[0] );
    KalmanAlignmentDataCollector::fillGraph( "r_vs_dx", theParentAlignable->globalPosition().perp(), trueParameters[0]-globalParameters[0] );
    KalmanAlignmentDataCollector::fillGraph( "y_vs_dx_true", theParentAlignable->globalPosition().y(), trueParameters[0] );
      
    for ( int i = 0; i < nParameter; ++i )
    {
      string startHistoName = histoNamePrefix + string( "_Start" ) + selectedParameter( i );
      KalmanAlignmentDataCollector::fillHistogram( startHistoName, -trueParameters[i]/selectedScaling(i) );

      string deltaHistoName = histoNamePrefix + string( "_Delta" ) + selectedParameter( i );
      KalmanAlignmentDataCollector::fillHistogram( deltaHistoName, (globalParameters[i]-trueParameters[i])/selectedScaling(i) );

      string valueHistoName = histoNamePrefix + string( "_Value" ) + selectedParameter( i );
      KalmanAlignmentDataCollector::fillHistogram( valueHistoName, globalParameters[i]/selectedScaling(i) );

      startHistoName = histoNamePrefix + theTypeAndLayer + string( "_Start" ) + selectedParameter( i );
      KalmanAlignmentDataCollector::fillHistogram( startHistoName, -trueParameters[i]/selectedScaling(i) );

      deltaHistoName = histoNamePrefix + theTypeAndLayer + string( "_Delta" ) + selectedParameter( i );
      KalmanAlignmentDataCollector::fillHistogram( deltaHistoName, (globalParameters[i]-trueParameters[i])/selectedScaling(i) );

      valueHistoName = histoNamePrefix + theTypeAndLayer + string( "_Value" ) + selectedParameter( i );
      KalmanAlignmentDataCollector::fillHistogram( valueHistoName, globalParameters[i]/selectedScaling(i) );
    }

#endif

  }
}


void KalmanAlignmentUserVariables::fixAlignable( void )
{
  AlignmentParameters* oldParameters = theParentAlignable->alignmentParameters();
  AlgebraicSymMatrix fixedCovariance = 1e-6*oldParameters->covariance();
  AlignmentParameters* newParameters = oldParameters->clone( oldParameters->parameters(), fixedCovariance );
  theParentAlignable->setAlignmentParameters( newParameters );
}


void KalmanAlignmentUserVariables::unfixAlignable( void )
{
  AlignmentParameters* oldParameters = theParentAlignable->alignmentParameters();
  AlgebraicSymMatrix fixedCovariance = 1e6*oldParameters->covariance();
  AlignmentParameters* newParameters = oldParameters->clone( oldParameters->parameters(), fixedCovariance );
  theParentAlignable->setAlignmentParameters( newParameters );
}


const AlgebraicVector KalmanAlignmentUserVariables::extractTrueParameters( void ) const
{

#ifdef USE_LOCAL_PARAMETERS

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

#else

  // get global rotation
  const align::RotationType& globalRotation = theParentAlignable->rotation();
  // get euler angles (global frame)
  align::EulerAngles globalEulerAngles = align::toAngles( globalRotation );

  // get global shifts
  align::GlobalVector globalShifts( globalRotation.multiplyInverse( theParentAlignable->displacement().basicVector() ) );

  AlgebraicVector trueParameters( 6 );
  trueParameters[0] = -globalShifts.x();
  trueParameters[1] = -globalShifts.y();
  trueParameters[2] = -globalShifts.z();
  trueParameters[3] = -globalEulerAngles[0];
  trueParameters[4] = -globalEulerAngles[1];
  trueParameters[5] = -globalEulerAngles[2];

#endif

  return trueParameters;
}


const string KalmanAlignmentUserVariables::selectedParameter( const int& selected ) const
{
  switch ( selected )
  {
  case 0:
    return string( "X" );
    break;
  case 1:
    return string( "Y" );
    break;
  case 2:
    return string( "Z" );
    break;
  case 3:
    return string( "Alpha" );
    break;
  case 4:
    return string( "Beta" );
    break;
  case 5:
    return string( "Gamma" );
    break;
  default:
    throw cms::Exception( "OutOfRange" ) << "[KalmanAlignmentUserVariables::selectedParameter] "
					 << "Index out of range (selector = " << selected << ")";
  }
}


float KalmanAlignmentUserVariables::selectedScaling( const int& selected ) const
{
  const float micron = 1e-4;
  const float millirad = 1e-3;
  //const float murad = 1e-6;

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
    //return murad;
    break;
  default:
    throw cms::Exception( "LogicError" ) << "@SUB=KalmanAlignmentUserVariables::selectedScaling"
					 << "Index out of range (selector = " << selected << ")\n";
  }
}


const string KalmanAlignmentUserVariables::toString( const int& i ) const
{
  char temp[10];
  snprintf( temp, sizeof(temp), "%u", i );

  return string( temp );
}
