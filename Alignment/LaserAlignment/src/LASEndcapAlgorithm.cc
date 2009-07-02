
#include "Alignment/LaserAlignment/src/LASEndcapAlgorithm.h"



///
///
///
LASEndcapAlgorithm::LASEndcapAlgorithm() {
}





///
/// implementation of the analytical solution for the endcap;
/// described in bruno's thesis (Appendix B):
/// http://darwin.bth.rwth-aachen.de/opus3/volltexte/2002/348
/// but extended with the beams' phi positions
///
LASEndcapAlignmentParameterSet LASEndcapAlgorithm::CalculateParameters( LASGlobalData<LASCoordinateSet>& measuredCoordinates, 
									LASGlobalData<LASCoordinateSet>& nominalCoordinates ) {
  
  std::cout << " [LASEndcapAlgorithm::CalculateParameters] -- Starting." << std::endl;

  // loop object
  LASGlobalLoop globalLoop;
  int det, ring, beam, disk;

  // vector containing the z positions of the disks in mm;
  // outer vector: TEC+/-, inner vector: 9 disks
  double zPositions[9] = { 1250., 1390., 1530., 1670., 1810., 1985., 2175., 2380., 2595. };
  std::vector<std::vector<double> > diskZPositions( 2, std::vector<double>( 9, 0. ) );
  for( det = 0; det < 2; ++ det ) {
    for( disk = 0; disk < 9; ++disk ) {
      diskZPositions.at( det ).at( disk ) = ( det==0 ? zPositions[disk] : -1. * zPositions[disk] );
    }
  }
  
  // constants
  const double endcapLength = 1345.; // mm

  // now come some sums which are later used in the formulas for the parameters.
  // the rings are implicitly summed over, however, this brings a little complication:
  // to make the calculation of the parameters independent of the ring (=radius),
  // we define some of the sums twice, once for phi and once for y=r*phi

  // sum over r*phi or phi for each endcap and for each disk (both rings)
  // outer vector: TEC+/-, inner vector: 9 disks
  std::vector<std::vector<double> > sumOverY( 2, std::vector<double>( 9, 0. ) );
  std::vector<std::vector<double> > sumOverPhi( 2, std::vector<double>( 9, 0. ) );

  // double sum over r*phi or phi, for each endcap (both rings)
  // outer vector: TEC+/-
  std::vector<double> doubleSumOverY( 2, 0. );
  std::vector<double> doubleSumOverPhi( 2, 0. );

  // sum over r*phi*z or phi*z, for each endcap (both rings)
  // outer vector: TEC+/-
  std::vector<double> sumOverYZ( 2, 0. );
  std::vector<double> sumOverPhiZ( 2, 0. );

  // sum over sin(phi_nominal)*R*phi for each endcap and for each disk (both rings)
  std::vector<std::vector<double> > sumOverSinThetaY( 2, std::vector<double>( 9, 0. ) );

  // sum over cos(phi_nominal)*R*phi for each endcap and for each disk (both rings)
  std::vector<std::vector<double> > sumOverCosThetaY( 2, std::vector<double>( 9, 0. ) );

  // double sum over sin or cos(phi_nominal)*phi, for each endcap
  std::vector<double> doubleSumOverSinThetaY( 2, 0. );
  std::vector<double> doubleSumOverCosThetaY( 2, 0. );

  // double sum over sin or cos(phi_nominal)*phi*z, for each endcap
  std::vector<double> doubleSumOverSinThetaYZ( 2, 0. );
  std::vector<double> doubleSumOverCosThetaYZ( 2, 0. );

  // sum over z values / sum over z^2 values
  std::vector<double> sumOverZ( 2, 0. );
  std::vector<double> sumOverZSquared( 2, 0. );


  // now calculate them
  det = 0; ring = 0; beam = 0; disk = 0;
  do {

    // current radius, depends on the ring
    const double radius = nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetR();

    // residual in r*phi (in the formulas this corresponds to y_ik)
    const double residual = ( measuredCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() - nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) * radius;
    
    sumOverY.at( det ).at( disk ) += residual;
    sumOverPhi.at( det ).at( disk ) += residual / radius;
    
    doubleSumOverY.at( det ) += residual;
    doubleSumOverPhi.at( det ) += residual / radius;
    
    sumOverYZ.at( det ) += diskZPositions.at( det ).at( disk ) * residual;
    sumOverPhiZ.at( det ) +=  diskZPositions.at( det ).at( disk ) * residual / radius;

    sumOverSinThetaY.at( det ).at( disk ) += sin( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) * residual;
    sumOverCosThetaY.at( det ).at( disk ) += cos( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) * residual;

    doubleSumOverSinThetaY.at( det ) += sin( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) * residual;
    doubleSumOverCosThetaY.at( det ) += cos( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) * residual;

    doubleSumOverSinThetaYZ.at( det ) += sin( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) * diskZPositions.at( det ).at( disk ) * residual;
    doubleSumOverCosThetaYZ.at( det ) += cos( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) * diskZPositions.at( det ).at( disk ) * residual;

  } while( globalLoop.TECLoop( det, ring, beam, disk ) );


  // disk-wise sums
  for( disk = 0; disk < 9; ++disk ) {
    sumOverZ.at( 0 ) += diskZPositions.at( 0 ).at( disk ); sumOverZ.at( 1 ) += diskZPositions.at( 0 ).at( disk );
    sumOverZSquared.at( 0 ) += pow( diskZPositions.at( 0 ).at( disk ), 2 ); sumOverZSquared.at( 1 ) += pow( diskZPositions.at( 0 ).at( disk ), 2 );
  }
  
  

  // now we can calculate the parameters for both TECs simultaneously,
  // so they're all vectors( 2 ) for TEC+/- (global parameters), or dim 2*9 (disk parameters)

  // define them..

  // deltaPhi_0
  std::vector<double> deltaPhi0( 2, 0. );
  
  // deltaPhi_t
  std::vector<double> deltaPhiT( 2, 0. );
  
  // deltaPhi_k (k=0..8)
  std::vector<std::vector<double> > deltaPhiK( 2, std::vector<double>( 9, 0. ) );

  // deltaX_0
  std::vector<double> deltaX0( 2, 0. );
  
  // deltaX_t
  std::vector<double> deltaXT( 2, 0. );
  
  // deltaX_k (k=0..8)
  std::vector<std::vector<double> > deltaXK( 2, std::vector<double>( 9, 0. ) );

  // deltaY_0
  std::vector<double> deltaY0( 2, 0. );
  
  // deltaY_t
  std::vector<double> deltaYT( 2, 0. );
  
  // deltaY_k (k=0..8)
  std::vector<std::vector<double> > deltaYK( 2, std::vector<double>( 9, 0. ) );


  // ..and fill them
  for( det = 0; det < 2; ++det ) { // TEC+/- loop
    
    // deltaPhi_0
    // here we use the phi-sums to eliminate the radius
    deltaPhi0.at( det ) = ( sumOverZSquared.at( det ) * doubleSumOverPhi.at( det ) - sumOverZ.at( det ) * sumOverPhiZ.at( det ) )
      / ( 8. * ( pow( sumOverZ.at( det ), 2 ) - 9. * sumOverZSquared.at( det ) ) );

    // deltaPhi_t
    // again use the phi-sums
    deltaPhiT.at( det ) = endcapLength * ( 9. * sumOverPhiZ.at( det ) - sumOverZ.at( det ) * doubleSumOverPhi.at( det ) )
      / ( 8. * ( pow( sumOverZ.at( det ), 2 ) - 9. * sumOverZSquared.at( det ) ) );

    // deltaPhi_k (k=0..8)
    // again use the phi-sums
    for( disk = 0; disk < 9; ++disk ) {
      deltaPhiK.at( det ).at( disk ) = ( -1. * diskZPositions.at( det ).at( disk ) * deltaPhiT.at( det ) / endcapLength )
	-  ( deltaPhi0.at( det ) )  -  sumOverPhi.at( det ).at( disk ) / 8.;
    }
    
    // deltaX_0
    deltaX0.at( det ) = 2. * ( sumOverZ.at( det ) * doubleSumOverSinThetaYZ.at( det ) - sumOverZSquared.at( det ) * doubleSumOverSinThetaY.at( det ) )
      / ( 8. * ( pow( sumOverZ.at( det ), 2 ) - 9. * sumOverZSquared.at( det ) ) );

    // deltaX_t
    deltaXT.at( det ) = 2. * endcapLength * ( sumOverZ.at( det ) * doubleSumOverSinThetaY.at( det ) - 9. * doubleSumOverSinThetaYZ.at( det ) )
      / ( 8. * ( pow( sumOverZ.at( det ), 2 ) - 9. * sumOverZSquared.at( det ) ) );

    // deltaX_k (k=0..8)
    for( disk = 0; disk < 9; ++disk ) {
      deltaXK.at( det ).at( disk ) =  ( -1. * diskZPositions.at( det ).at( disk ) * deltaXT.at( det ) / endcapLength )
 	-  ( deltaX0.at( det ) )  +  2. * sumOverSinThetaY.at( det ).at( disk ) / 8.;
    }

    // deltaY_0
    deltaY0.at( det ) = 2. * ( sumOverZSquared.at( det ) * doubleSumOverCosThetaY.at( det ) - sumOverZ.at( det ) * doubleSumOverCosThetaYZ.at( det ) )
      / ( 8. * ( pow( sumOverZ.at( det ), 2 ) - 9. * sumOverZSquared.at( det ) ) );

    // deltaY_t
    deltaYT.at( det ) = 2. * endcapLength * ( 9. * doubleSumOverCosThetaYZ.at( det ) - sumOverZ.at( det ) * doubleSumOverCosThetaY.at( det ) )
      / ( 8. * ( pow( sumOverZ.at( det ), 2 ) - 9. * sumOverZSquared.at( det ) ) );

    // deltaY_k (k=0..8)
    for( disk = 0; disk < 9; ++disk ) {
      deltaYK.at( det ).at( disk ) =  ( -1. * diskZPositions.at( det ).at( disk ) * deltaYT.at( det ) / endcapLength )
 	-  ( deltaY0.at( det ) )  -  2. * sumOverCosThetaY.at( det ).at( disk ) / 8.;
    }
  
  }


  // fill the result
  LASEndcapAlignmentParameterSet theResult;

  // for the moment we fill only the values, not the errors

  for( det = 0; det < 2; ++det ) {
    for( disk = 0; disk < 9; ++disk ) {
      
      // the rotation parameters: deltaPhi_k
      theResult.GetParameter( det, disk, 0 ).first = deltaPhiK.at( det ).at( disk );

      // the x offsets: deltaX_k
      theResult.GetParameter( det, disk, 1 ).first = deltaXK.at( det ).at( disk );
      
      // the y offsets: deltaY_k
      theResult.GetParameter( det, disk, 2 ).first = deltaYK.at( det ).at( disk );

    }
  }

  std::cout << " [LASEndcapAlgorithm::CalculateParameters] -- Done." << std::endl;

  return( theResult );

}
