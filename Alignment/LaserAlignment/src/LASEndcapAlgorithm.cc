
#include "Alignment/LaserAlignment/src/LASEndcapAlgorithm.h"



///
///
///
LASEndcapAlgorithm::LASEndcapAlgorithm() {
}





///
/// implementation of the analytical solution for the endcap;
/// described in bruno's thesis:
/// http://darwin.bth.rwth-aachen.de/opus3/volltexte/2002/348
/// but extended with the beams' phi positions
///
/// returns a vector: [0] for TEC+ and [1] for TEC-
///
LASEndcapAlignmentParameterSet LASEndcapAlgorithm::CalculateParameters( LASGlobalData<LASCoordinateSet>& measuredCoordinates, 
									LASGlobalData<LASCoordinateSet>& nominalCoordinates ) {
  
  std::cout << " [LASEndcapAlgorithm::CalculateParameters] -- Starting." << std::endl;

  // loop object
  LASGlobalLoop globalLoop;
  int det, ring, beam, disk;

  // vector containing the z positions of the disks in mm;
  // outer vector: TEC+/-, inner vector: 9 disks
  double zPositions[9] = { 1250, 1390, 1530, 1670, 1810, 1985, 2175, 2380, 2595 };
  std::vector<std::vector<double> > diskZPositions( 2, std::vector<double>( 9, 0. ) );
  for( int aDet = 0; aDet < 2; ++ aDet ) {
    for( int aDisk = 0; aDisk < 9; ++aDisk ) {
      diskZPositions.at( aDet ).at( aDisk ) = ( aDet==0 ? zPositions[aDisk] : -1. * zPositions[aDisk] );
    }
  }
  
  // constants
  const double endcapLength = 1345.; // mm

  // define some sums...

  // sum over phi for each endcap and for each disk (both rings)
  // outer vector: TEC+/-, inner vector: 9 disks
  std::vector<std::vector<double> > sumOverPhi( 2, std::vector<double>( 9, 0. ) );

  // sum over sin(phi_nominal)*R*phi for each endcap and for each disk (both rings)
  std::vector<std::vector<double> > sumOverSinPhiY( 2, std::vector<double>( 9, 0. ) );

  // sum over cos(phi_nominal)*R*phi for each endcap and for each disk (both rings)
  std::vector<std::vector<double> > sumOverCosPhiY( 2, std::vector<double>( 9, 0. ) );
  
  // ...and calculate them
  det = 0; ring = 0; beam = 0; disk = 0;
  do {

    sumOverPhi.at( det ).at( disk )    += measuredCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi();

    sumOverSinPhiY.at( det ).at( disk ) += sin( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() )
      * ( measuredCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() - nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) // residual in phi
      * nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetR(); // now residual in mm (could also take measured instead of nominal here)
    
    sumOverCosPhiY.at( det ).at( disk ) += cos( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() )
      * ( measuredCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() - nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) // residual in phi
      * nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetR(); // now residual in mm (could also take measured instead of nominal here)

  } while( globalLoop.TECLoop( det, ring, beam, disk ) );



  // now we can calculate the parameters for both TECs simultaneously,
  // so they're all vectors( 2 ) for TEC+/- (global parameters), or dim 2*9 (disk parameters)

  // @@@ for the time being, the global parameters are set to zero @@@

  // deltaPhi_0
  std::vector<double> deltaPhi0( 2, 0. ); // yet not implemented
  
  // deltaPhi_t
  std::vector<double> deltaPhiT( 2, 0. ); // yet not implemented
  
  // deltaPhi_k (k=0..8)
  std::vector<std::vector<double> > deltaPhiK( 2, std::vector<double>( 9, 0. ) );

  // deltaX_0
  std::vector<double> deltaX0( 2, 0. ); // yet not implemented
  
  // deltaX_t
  std::vector<double> deltaXT( 2, 0. ); // yet not implemented
  
  // deltaX_k (k=0..8)
  std::vector<std::vector<double> > deltaXK( 2, std::vector<double>( 9, 0. ) );

  // deltaY_0
  std::vector<double> deltaY0( 2, 0. ); // yet not implemented
  
  // deltaY_t
  std::vector<double> deltaYT( 2, 0. ); // yet not implemented
  
  // deltaY_k (k=0..8)
  std::vector<std::vector<double> > deltaYK( 2, std::vector<double>( 9, 0. ) );


  // fill the non-const vectors
  for( int aDet = 0; aDet < 2; ++aDet ) { // TEC+/- loop
    
    // deltaPhi_k (k=0..8)
    for( int aDisk = 0; aDisk < 9; ++aDisk ) {
      deltaPhiK.at( aDet ).at( aDisk ) = ( -1. * diskZPositions.at( aDet ).at( aDisk ) * deltaPhiT.at( aDet ) / endcapLength )
	-  ( deltaPhi0.at( aDet ) )  -  sumOverPhi.at( aDet ).at( aDisk ) / 8. + 2. * M_PI; // the +2*M_PI has been determined empirically...
    }
    
    // deltaX_k (k=0..8)
    for( int aDisk = 0; aDisk < 9; ++aDisk ) {
      deltaXK.at( aDet ).at( aDisk ) =  ( -1. * diskZPositions.at( aDet ).at( aDisk ) * deltaXT.at( aDet ) / endcapLength )
 	-  ( deltaX0.at( aDet ) )  +  2. * sumOverSinPhiY.at( aDet ).at( aDisk ) / 8.;
    }

    // deltaY_k (k=0..8)
    for( int aDisk = 0; aDisk < 9; ++aDisk ) {
      deltaYK.at( aDet ).at( aDisk ) =  ( -1. * diskZPositions.at( aDet ).at( aDisk ) * deltaYT.at( aDet ) / endcapLength )
 	-  ( deltaY0.at( aDet ) )  -  2. * sumOverCosPhiY.at( aDet ).at( aDisk ) / 8.;
    }
  
  }


  // fill the result
  LASEndcapAlignmentParameterSet theResult;

  // for the moment we fill only the values, not the errors

  for( int aDet = 0; aDet < 2; ++aDet ) {
    for( int aDisk = 0; aDisk < 9; ++aDisk ) {
      
      // the rotation parameters: deltaPhi_k
      theResult.GetParameter( aDet, aDisk, 0 ).first = deltaPhiK.at( aDet ).at( aDisk );

      // the x offsets: deltaX_k
      theResult.GetParameter( aDet, aDisk, 1 ).first = deltaXK.at( aDet ).at( aDisk );
      
      // the y offsets: deltaY_k
      theResult.GetParameter( aDet, aDisk, 2 ).first = deltaYK.at( aDet ).at( aDisk );

    }
  }

  std::cout << " [LASEndcapAlgorithm::CalculateParameters] -- Done." << std::endl;

  return( theResult );

}
