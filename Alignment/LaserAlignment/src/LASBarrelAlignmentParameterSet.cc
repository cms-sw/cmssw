
#include "Alignment/LaserAlignment/src/LASBarrelAlignmentParameterSet.h"



///
///
///
LASBarrelAlignmentParameterSet::LASBarrelAlignmentParameterSet(){

  Init();
  
}





///
/// whatever initialization is needed
///
void LASBarrelAlignmentParameterSet::Init( void ) {

  // could use a single vector<vector<vector<pair<> > > >
  // but better split it in 6 parts

  for( int i = 0; i < 2; ++i ) { // twice; once for each endface
    tecPlusParameters.push_back ( std::vector<std::pair<double,double> >( 3 ) );
    tecMinusParameters.push_back( std::vector<std::pair<double,double> >( 3 ) );
    tibPlusParameters.push_back ( std::vector<std::pair<double,double> >( 3 ) );
    tibMinusParameters.push_back( std::vector<std::pair<double,double> >( 3 ) );
    tobPlusParameters.push_back ( std::vector<std::pair<double,double> >( 3 ) );
    tobMinusParameters.push_back( std::vector<std::pair<double,double> >( 3 ) );
  }

}





///
/// function for accessing a single parameter (pair<>);
/// indices are:
///  * aSubdetector = 0 (TEC+), 1 (TEC-), 2 (TIB+), 3 (TIB-), 4 (TOB+), 5 (TOB-)
///  * aDisk = 0 (lower z), 1 (upper z)
///  * aParameter: 0 (rotation angle), 1 (x displacement), 2 (y displacement)
///
std::pair<double,double>& LASBarrelAlignmentParameterSet::GetParameter( int aSubdetector, int aDisk, int aParameter ) {

  if( aSubdetector < 0 || aSubdetector > 5 ) {
    throw cms::Exception( "Laser Alignment" ) << " [LASBarrelAlignmentParameterSet::GetParameter] ERROR ** Illegal subdetector index: " << aSubdetector << "." << std::endl;
  }

  if( aDisk < 0 || aDisk > 1 ) {
    throw cms::Exception( "Laser Alignment" ) << " [LASBarrelAlignmentParameterSet::GetParameter] ERROR ** Illegal endface index: " << aDisk << "." << std::endl;
  }
  
  if( aParameter < 0 || aParameter > 2 ) {
    throw cms::Exception( "Laser Alignment" ) << " [LASBarrelAlignmentParameterSet::GetParameter] ERROR ** Illegal parameter index: " << aParameter << "." << std::endl;
  }


  // would use a switch here, but this creates problems..
  if( aSubdetector == 0 ) return tecPlusParameters.at( aDisk ).at( aParameter );
  else if( aSubdetector == 1 ) return tecMinusParameters.at( aDisk ).at( aParameter );
  else if( aSubdetector == 2 ) return tibPlusParameters.at( aDisk ).at( aParameter );
  else if( aSubdetector == 3 ) return tibMinusParameters.at( aDisk ).at( aParameter );
  else if( aSubdetector == 4 ) return tobPlusParameters.at( aDisk ).at( aParameter );
  else return tobMinusParameters.at( aDisk ).at( aParameter );


}
