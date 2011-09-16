

#include "Alignment/LaserAlignment/interface/LASConstants.h"
#include "FWCore/Utilities/interface/Exception.h"

///
///
///
LASConstants::LASConstants() :
  atRadius(0.), tecBsZPosition(0.), atZPosition(0.)
{
}





///
///
///
LASConstants::LASConstants( std::vector<edm::ParameterSet> const& theConstConf ) {

  InitContainers();

  for( std::vector<edm::ParameterSet>::const_iterator iter = theConstConf.begin(); iter < theConstConf.end(); ++iter ) {

    if( iter->getParameter<std::string>( "PSetName" ) == "BeamsplitterKinks" ) FillBsKinks( *iter );
    else if( iter->getParameter<std::string>( "PSetName" ) == "Radii" ) FillRadii( *iter );
    else if( iter->getParameter<std::string>( "PSetName" ) == "ZPositions" ) FillZPositions( *iter );
    else {
      std::cerr << " [] ** WARNING: Cannot process unknown parameter set named: " << iter->getParameter<std::string>( "PSetName" ) << "." << std::endl;
    }

  }

}





///
///
///
LASConstants::~LASConstants() {
}





///
/// Returns one beamsplitter kink, parameters are:
/// det  (0=TEC+/1=TEC-)
/// ring (0=R4/1=R6)
/// beam (0..7)
///
double LASConstants::GetEndcapBsKink( unsigned int det, unsigned int ring , unsigned int beam ) const {
  
  if( ! ( ( det == 0 || det == 1 ) && ( ring == 0 || ring == 1 ) && ( beam < 8U ) ) ) { // beam >= 0, since beam is unsigned
    throw cms::Exception( " [LASConstants::GetEndcapBsKink]" ) << " ** ERROR: no such element: det " << det << ", ring " << ring << ", beam " << beam << "." << std::endl;
  }

  return endcapBsKinks.at( det ).at( ring ).at( beam );
  
}





///
/// Returns beamplitter kink for alignment tube beam <beam> (0..7)
///
double LASConstants::GetAlignmentTubeBsKink( unsigned int beam ) const {
  
  if( beam >= 8U ) { // beam >= 0, since beam is unsigned
    throw cms::Exception( " [LASConstants::GetAlignmentTubeBsKink]" ) << " ** ERROR: no such beam: " << beam << "." << std::endl;
  }

  return alignmentTubeBsKinks.at( beam );

}





///
///
///
double LASConstants::GetTecRadius( unsigned int ring ) const {

  if( ring > 1U ) { // ring >= 0, since ring is unsigned
    throw cms::Exception( " [LASConstants::GetTecRadius]" ) << " ** ERROR: no such ring: " << ring << "." << std::endl;
  }

  return tecRadii.at( ring );

}





///
///
///
double LASConstants::GetAtRadius( void ) const {

  return atRadius;

}





///
///
///
double LASConstants::GetTecZPosition( unsigned int det, unsigned int disk ) const {

  if( ( det > 1 ) || ( disk > 8 ) ) {
    throw cms::Exception( " [LASConstants::GetTecZPosition]" ) << " ** ERROR: no such element: det " << det << ", disk " << disk << "." << std::endl;
  }

  if( det == 0 ) return tecZPositions.at( disk ); // tec+
  else return -1. * tecZPositions.at( disk ); // tec-

}





///
///
///
double LASConstants::GetTibZPosition( unsigned int pos ) const {

  if( pos > 5 ) {
    throw cms::Exception( " [LASConstants::GetTibZPosition]" ) << " ** ERROR: no such position: " << pos << "." << std::endl;
  }

  return tibZPositions.at( pos );

}





///
///
///
double LASConstants::GetTobZPosition( unsigned int pos ) const {

  if( pos > 5 ) {
    throw cms::Exception( " [LASConstants::GetTobZPosition]" ) << " ** ERROR: no such position: " << pos << "." << std::endl;
  }

  return tobZPositions.at( pos );

}





///
///
///
double LASConstants::GetTecBsZPosition( unsigned int det ) const {

  return tecBsZPosition;

}





///
///
///
double LASConstants::GetAtBsZPosition( void ) const {

  return atZPosition;

}





///
///
///
void LASConstants::InitContainers( void ) {

  // beam splitter kinks

  endcapBsKinks.resize( 2 ); // create two dets
  for( int det = 0; det < 2; ++det ) {
    endcapBsKinks.at( det ).resize( 2 ); // create two rings per det
    for( int ring = 0; ring < 2; ++ring ) {
      endcapBsKinks.at( det ).at( ring ).resize( 8 ); // 8 beams per ring
    }
  }

  alignmentTubeBsKinks.resize( 8 ); // 8 beams


  // radii
  tecRadii.resize( 2 );

  // z positions
  tecZPositions.resize( 9 );
  tibZPositions.resize( 6 );
  tobZPositions.resize( 6 );

}





///
/// fill the beamplitter-kink related containers
///
void LASConstants::FillBsKinks( edm::ParameterSet const&  theBsKinkConf ) {

  // tec+
  endcapBsKinks.at( 0 ).at( 0 ) = theBsKinkConf.getParameter<std::vector<double> >( "LASTecPlusRing4BsKinks" );
  endcapBsKinks.at( 0 ).at( 1 ) = theBsKinkConf.getParameter<std::vector<double> >( "LASTecPlusRing6BsKinks" );

  // apply global offsets
  for( unsigned int ring = 0; ring < 2; ++ring ) {
    for( unsigned int beam = 0; beam < 8; ++beam ) {
      endcapBsKinks.at( 0 ).at( ring ).at( beam ) += theBsKinkConf.getParameter<double>( "TecPlusGlobalOffset" );
    }
  }

  // tec-
  endcapBsKinks.at( 1 ).at( 0 ) = theBsKinkConf.getParameter<std::vector<double> >( "LASTecMinusRing4BsKinks" );
  endcapBsKinks.at( 1 ).at( 1 ) = theBsKinkConf.getParameter<std::vector<double> >( "LASTecMinusRing6BsKinks" );

  // apply global offsets
  for( unsigned int ring = 0; ring < 2; ++ring ) {
    for( unsigned int beam = 0; beam < 8; ++beam ) {
      endcapBsKinks.at( 1 ).at( ring ).at( beam ) += theBsKinkConf.getParameter<double>( "TecMinusGlobalOffset" );
    }
  }

  // at
  alignmentTubeBsKinks = theBsKinkConf.getParameter<std::vector<double> >( "LASAlignmentTubeBsKinks" );

}





///
/// fill the beam radii
///
void LASConstants::FillRadii( edm::ParameterSet const&  theRadiiConf ) {
  
  tecRadii = theRadiiConf.getParameter<std::vector<double> >( "LASTecRadius" );
  atRadius = theRadiiConf.getParameter<double>( "LASAtRadius" );

}





///
///
///
void LASConstants::FillZPositions( edm::ParameterSet const& theZPosConf ) {

  tecZPositions  = theZPosConf.getParameter<std::vector<double> >( "LASTecZPositions" );
  tibZPositions  = theZPosConf.getParameter<std::vector<double> >( "LASTibZPositions" );
  tobZPositions  = theZPosConf.getParameter<std::vector<double> >( "LASTobZPositions" );
  tecBsZPosition = theZPosConf.getParameter<double>( "LASTecBeamSplitterZPosition" );
  atZPosition    = theZPosConf.getParameter<double>( "LASAtBeamsplitterZPosition" );

}
