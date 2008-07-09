
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
    tecPlusDiskParameters.push_back ( std::vector<std::pair<double,double> >( 3 ) );
    tecMinusDiskParameters.push_back( std::vector<std::pair<double,double> >( 3 ) );
  }

  // once for each parameter
  for( int i = 0; i < 6; ++i ) {
    tecPlusGlobalParameters.push_back( std::pair<double,double>( 0., 0. ) );
    tecMinusGlobalParameters.push_back( std::pair<double,double>( 0., 0. ) );
  }

  // once for each beam
  for( int i = 0; i < 8; ++i ) {
    tecPlusBeamParameters.push_back( std::vector<std::pair<double,double> >( 2 ) );
    tecMinusBeamParameters.push_back( std::vector<std::pair<double,double> >( 2 ) );
  }

}





///
/// function for accessing a single disk parameter (pair<>);
/// indices are:
///  * aSubdetector = 0 (TEC+), 1 (TEC-)
///  * aDisk = 0..8 (from inner to outer)
///  * aParameter: 0 (rotation angle), 1 (x displacement), 2 (y displacement)
///
std::pair<double,double>& LASEndcapAlignmentParameterSet::GetDiskParameter( int aSubdetector, int aDisk, int aParameter ) {

  if( aSubdetector < 0 || aSubdetector > 1 ) {
    throw cms::Exception( "Laser Alignment" ) << " [LASEndcapAlignmentParameterSet::GetDiskParameter] ERROR ** Illegal subdetector index: " << aSubdetector << "." << std::endl;
  }

  if( aDisk < 0 || aDisk > 8 ) {
    throw cms::Exception( "Laser Alignment" ) << " [LASEndcapAlignmentParameterSet::GetDiskParameter] ERROR ** Illegal disk index: " << aDisk << "." << std::endl;
  }
  
  if( aParameter < 0 || aParameter > 2 ) {
    throw cms::Exception( "Laser Alignment" ) << " [LASEndcapAlignmentParameterSet::GetDiskParameter] ERROR ** Illegal parameter index: " << aParameter << "." << std::endl;
  }


  if( aSubdetector == 0 ) return tecPlusDiskParameters.at( aDisk ).at( aParameter );
  return tecMinusDiskParameters.at( aDisk ).at( aParameter );

}





///
/// function for accessing a single disk parameter (pair<>);
/// indices are:
///  * aSubdetector = 0 (TEC+), 1 (TEC-)
///  * aParameter: 0 (global rotation), 1 (global torsion), 
///                2 (global x shift),  3 (global x shear),
///                4 (global y shift),  5 (global y shear)
///
std::pair<double,double>& LASEndcapAlignmentParameterSet::GetGlobalParameter( int aSubdetector, int aParameter ) {
  
  if( aSubdetector < 0 || aSubdetector > 1 ) {
    throw cms::Exception( "Laser Alignment" ) << " [LASEndcapAlignmentParameterSet::GetGlobalParameter] ERROR ** Illegal subdetector index: " << aSubdetector << "." << std::endl;
  }

  if( aParameter < 0 || aParameter > 5 ) {
    throw cms::Exception( "Laser Alignment" ) << " [LASEndcapAlignmentParameterSet::GetGlobalParameter] ERROR ** Illegal parameter index: " << aParameter << "." << std::endl;
  }

  if( aSubdetector == 0 ) return tecPlusGlobalParameters.at( aParameter );
  return tecMinusGlobalParameters.at( aParameter );

}





///
/// function for accessing a single disk parameter (pair<>);
/// indices are:
///  * aSubdetector = 0 (TEC+), 1 (TEC-)
///  * aBeam = 0..7
///  * aParameter: 0 (deltaPhi on disk0), 1 (deltaPhi on disk8), 
///
std::pair<double,double>& LASEndcapAlignmentParameterSet::GetBeamParameter( int aSubdetector, int aBeam, int aParameter ) {
  
  if( aSubdetector < 0 || aSubdetector > 1 ) {
    throw cms::Exception( "Laser Alignment" ) << " [LASEndcapAlignmentParameterSet::GetBeamParameter] ERROR ** Illegal subdetector index: " << aSubdetector << "." << std::endl;
  }

  if( aBeam < 0 || aBeam > 7 ) {
    throw cms::Exception( "Laser Alignment" ) << " [LASEndcapAlignmentParameterSet::GetBeamParameter] ERROR ** Illegal beam index: " << aBeam << "." << std::endl;
  }

  if( aParameter < 0 || aParameter > 5 ) {
    throw cms::Exception( "Laser Alignment" ) << " [LASEndcapAlignmentParameterSet::GetBeamParameter] ERROR ** Illegal parameter index: " << aParameter << "." << std::endl;
  }

  if( aSubdetector == 0 ) return tecPlusBeamParameters.at( aBeam ).at( aParameter );
  return tecMinusBeamParameters.at( aBeam ).at( aParameter );

}





///
/// printout for debugging
///
void LASEndcapAlignmentParameterSet::Dump( void ) {
  
  std::cout << " [LASEndcapAlignmentParameterSet::Dump] -- Dumping parameters:" << std::endl;
  for( int det = 0; det < 2; ++det ) {
    std::cout << "   " << (det==0 ? "TEC+" : "TEC-") << " disk parameters in format: dPhi±e dX±e dY±e (rad/mm): " << std::endl;
    for( int disk = 0; disk < 9; ++disk ) {
      std::cout << "    disk " << disk << ": ";
      for( int par = 0; par < 3; ++par ) std::cout << "  " << std::setw( 9 ) << std::setprecision( 6 ) << std::right << GetDiskParameter( det, disk, par ).first 
						   << " ± " << std::setw( 9 ) << std::setprecision( 6 ) << std::left << GetDiskParameter( det, disk, par ).second;
      std::cout << std::endl;
    }
  }

  for( int det = 0; det < 2; ++det ) {
    std::cout << "   " << (det==0 ? "TEC+" : "TEC-") << " global parameters in format: dPhi0±e dPhiT±e dX0±e dXT±e dY0±e dYT±e (rad/mm): " << std::endl;
    for( int par = 0; par < 6; ++par ) std::cout << "   " << std::setw( 9 ) << std::setprecision( 6 ) << std::right << GetGlobalParameter( det, par ).first 
						 << " ± " << std::setw( 9 ) << std::setprecision( 6 ) << std::left << GetGlobalParameter( det, par ).second;
    std::cout << std::endl;
  }

  for( int det = 0; det < 2; ++det ) {
    std::cout << "   " << (det==0 ? "TEC+" : "TEC-") << " beam parameters in format: dPhi1±e dPhi2±e (rad): " << std::endl;
    for( int beam = 0; beam < 8; ++beam ) {
      std::cout << "    beam " << beam << ": ";
      for( int par = 0; par < 2; ++par ) std::cout << std::setw( 9 ) << std::setprecision( 6 ) << std::right << GetBeamParameter( det, beam, par ).first 
						   << " ± " << std::setw( 9 ) << std::setprecision( 6 ) << std::left << GetBeamParameter( det, beam, par ).second;
      std::cout << std::endl;
    }
  }

  std::cout << " [LASEndcapAlignmentParameterSet::Dump] -- End of dump." << std::endl << std::endl;

}

