
#include "Alignment/LaserAlignment/src/LASEndcapAlignmentParameterSet.h"




///
///
///
LASEndcapAlignmentParameterSet::LASEndcapAlignmentParameterSet() {

  Init();
  
}





///
/// whatever initialization is needed
///
void LASEndcapAlignmentParameterSet::Init( void ) {

  // could use a single vector<vector<vector<pair<> > > >
  // but better split it in 2 parts

  for( int i = 0; i < 9; ++i ) { // nine times; once for each disk
    tecPlusParameters.push_back ( std::vector<std::pair<double,double> >( 3 ) );
    tecMinusParameters.push_back( std::vector<std::pair<double,double> >( 3 ) );
  }

}





///
/// function for accessing a single parameter (pair<>);
/// indices are:
///  * aSubdetector = 0 (TEC+), 1 (TEC-)
///  * aDisk = 0..8 (from inner to outer)
///  * aParameter: 0 (rotation angle), 1 (x displacement), 2 (y displacement)
///
std::pair<double,double>& LASEndcapAlignmentParameterSet::GetParameter( int aSubdetector, int aDisk, int aParameter ) {

  if( aSubdetector < 0 || aSubdetector > 1 ) {
    throw cms::Exception( "Laser Alignment" ) << " [LASEndcapAlignmentParameterSet::GetParameter] ERROR ** Illegal subdetector index: " << aSubdetector << "." << std::endl;
  }

  if( aDisk < 0 || aDisk > 8 ) {
    throw cms::Exception( "Laser Alignment" ) << " [LASEndcapAlignmentParameterSet::GetParameter] ERROR ** Illegal endface index: " << aDisk << "." << std::endl;
  }
  
  if( aParameter < 0 || aParameter > 2 ) {
    throw cms::Exception( "Laser Alignment" ) << " [LASEndcapAlignmentParameterSet::GetParameter] ERROR ** Illegal parameter index: " << aParameter << "." << std::endl;
  }


  if( aSubdetector == 0 ) return tecPlusParameters.at( aDisk ).at( aParameter );
  return tecMinusParameters.at( aDisk ).at( aParameter );

}





///
/// printout for debugging
///
void LASEndcapAlignmentParameterSet::Dump( void ) {
  
  std::cout << " [LASEndcapAlignmentParameterSet::Dump] -- Dumping parameters in format: deltaPhi±e, deltaX±e, deltaY±e" << std::endl;
  for( int det = 0; det < 2; ++det ) {
    std::cout << "   " << (det==0 ? "TEC+" : "TEC-") << ": " << std::endl;
    for( int disk = 0; disk < 9; ++disk ) {
      std::cout << "     disk " << disk << ": ";
      for( int par = 0; par < 3; ++par ) std::cout << std::setw( 5 ) << GetParameter( det, disk, par ).first 
						   << " ± " << std::setw( 5 ) << std::left << GetParameter( det, disk, par ).second;
      std::cout << std::endl;
    }
  }
  std::cout << " [LASEndcapAlignmentParameterSet::Dump] -- End of dump." << std::endl << std::endl;

}

