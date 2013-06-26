
#include "Alignment/LaserAlignment/interface/LASEndcapAlgorithm.h"



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
  const double zPositions[9] = { 1322.5,  1462.5,  1602.5,  1742.5,  1882.5,  2057.5,  2247.5,  2452.5,  2667.5 };
  std::vector<std::vector<double> > diskZPositions( 2, std::vector<double>( 9, 0. ) );
  for( det = 0; det < 2; ++det ) {
    for( disk = 0; disk < 9; ++disk ) {
      // sign depends on side of course
      diskZPositions.at( det ).at( disk ) = ( det==0 ? zPositions[disk] : -1. * zPositions[disk] ); 
    }
  }

  // vector containing the phi positions of the beam in rad;
  // outer vector: TEC+/-, inner vector: 8 beams
  const double phiPositions[8] = { 0.392699, 1.178097, 1.963495, 2.748894, 3.534292, 4.319690, 5.105088, 5.890486 };
  std::vector<std::vector<double> > beamPhiPositions( 2, std::vector<double>( 8, 0. ) );
  for( det = 0; det < 2; ++ det ) {
    for( beam = 0; beam < 8; ++beam ) {
      beamPhiPositions.at( det ).at( beam ) = phiPositions[beam];
    }
  }
  
  // vector containing the radius of ring4,ring6 = (0,1)
  std::vector<double> radius( 2, 0. );
  radius.at( 0 ) = 564.; radius.at( 1 ) = 840.;

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

  // sum over phi for each endcap and for each beam (both rings)
  // outer vector: TEC+/-, inner vector: 8 beams
  std::vector<std::vector<double> > kSumOverPhi( 2, std::vector<double>( 8, 0. ) );

  // double sum over r*phi or phi, for each endcap (both rings)
  // outer vector: TEC+/-
  std::vector<double> doubleSumOverY( 2, 0. );
  std::vector<double> doubleSumOverPhi( 2, 0. );

  // sum over r*phi*z or phi*z, for each endcap and for each beam (both rings)
  // outer vector: TEC+/-, inner vector: 8 beams
  std::vector<std::vector<double> > kSumOverPhiZ( 2, std::vector<double>( 8, 0. ) );

  // sum over r*phi*z or phi*z, for each endcap (both rings)
  // outer vector: TEC+/-
  std::vector<double> doubleSumOverYZ( 2, 0. );
  std::vector<double> doubleSumOverPhiZ( 2, 0. );

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


  // now calculate the sums
  det = 0; ring = 0; beam = 0; disk = 0;
  do {
    if( ring == 1 ) continue; //################################################################################################### BOUND TO RING6
    // current radius, depends on the ring
    const double radius = nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetR();

    // residual in r*phi (in the formulas this corresponds to y_ik)
    const double residual = ( measuredCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() - nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) * radius;
    
    sumOverY.at( det ).at( disk ) += residual;
    sumOverPhi.at( det ).at( disk ) += residual / radius;
    kSumOverPhi.at( det ).at( beam ) += residual / radius;

    doubleSumOverY.at( det ) += residual;
    doubleSumOverPhi.at( det ) += residual / radius;

    kSumOverPhiZ.at( det ).at( beam ) += diskZPositions.at( det ).at( disk ) * residual / radius;

    doubleSumOverYZ.at( det ) += diskZPositions.at( det ).at( disk ) * residual;
    doubleSumOverPhiZ.at( det ) +=  diskZPositions.at( det ).at( disk ) * residual / radius;

    sumOverSinThetaY.at( det ).at( disk ) += sin( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) * residual;
    sumOverCosThetaY.at( det ).at( disk ) += cos( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) * residual;

    doubleSumOverSinThetaY.at( det ) += sin( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) * residual;
    doubleSumOverCosThetaY.at( det ) += cos( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) * residual;

    doubleSumOverSinThetaYZ.at( det ) += sin( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) * diskZPositions.at( det ).at( disk ) * residual;
    doubleSumOverCosThetaYZ.at( det ) += cos( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) * diskZPositions.at( det ).at( disk ) * residual;

  } while( globalLoop.TECLoop( det, ring, beam, disk ) );


  // disk-wise sums
  for( disk = 0; disk < 9; ++disk ) {
    sumOverZ.at( 0 ) += diskZPositions.at( 0 ).at( disk ); sumOverZ.at( 1 ) += diskZPositions.at( 1 ).at( disk );
    sumOverZSquared.at( 0 ) += pow( diskZPositions.at( 0 ).at( disk ), 2 ); sumOverZSquared.at( 1 ) += pow( diskZPositions.at( 1 ).at( disk ), 2 );
  }
  

  // now we can calculate the parameters for both TECs simultaneously,
  // so they're all vectors( 2 ) for TEC+/- (global parameters), or dim 2*9 (disk parameters),
  // or dim 2*8 (beam parameters)

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

  // beam parameters: deltaTheta_A, deltaTheta_B (i=0..7)
  std::vector<std::vector<double> > deltaTA( 2, std::vector<double>( 8, 0. ) );
  std::vector<std::vector<double> > deltaTB( 2, std::vector<double>( 8, 0. ) );
  


  // ..and fill them;
  // the additional divisors "/ 2." come from the fact that we average over both rings
  for( det = 0; det < 2; ++det ) { // TEC+/- loop
    
    // deltaPhi_0
    // here we use the phi-sums to eliminate the radius
    deltaPhi0.at( det ) = ( sumOverZSquared.at( det ) * doubleSumOverPhi.at( det ) - sumOverZ.at( det ) * doubleSumOverPhiZ.at( det ) )
      / ( 8. * ( pow( sumOverZ.at( det ), 2 ) - 9. * sumOverZSquared.at( det ) ) ); // / 2.; // @@@@@@@

    // deltaPhi_t
    // again use the phi-sums
    deltaPhiT.at( det ) = endcapLength * ( 9. * doubleSumOverPhiZ.at( det ) - sumOverZ.at( det ) * doubleSumOverPhi.at( det ) )
      / ( 8. * ( pow( sumOverZ.at( det ), 2 ) - 9. * sumOverZSquared.at( det ) ) ); // / 2.; // @@@@@@@

    // deltaPhi_k (k=0..8)
    // again use the phi-sums
    for( disk = 0; disk < 9; ++disk ) {
      deltaPhiK.at( det ).at( disk ) = ( -1. * diskZPositions.at( det ).at( disk ) * deltaPhiT.at( det ) / endcapLength )
	-  ( deltaPhi0.at( det ) )  -  sumOverPhi.at( det ).at( disk ) / 8.;// / 2.; // @@@@@@@
    }
    
    // deltaX_0
    deltaX0.at( det ) = 2. * ( sumOverZ.at( det ) * doubleSumOverSinThetaYZ.at( det ) - sumOverZSquared.at( det ) * doubleSumOverSinThetaY.at( det ) )
      / ( 8. * ( pow( sumOverZ.at( det ), 2 ) - 9. * sumOverZSquared.at( det ) ) );// / 2.; // @@@@@@@

    // deltaX_t
    deltaXT.at( det ) = 2. * endcapLength * ( sumOverZ.at( det ) * doubleSumOverSinThetaY.at( det ) - 9. * doubleSumOverSinThetaYZ.at( det ) )
      / ( 8. * ( pow( sumOverZ.at( det ), 2 ) - 9. * sumOverZSquared.at( det ) ) );// / 2.; // @@@@@@@

    // deltaX_k (k=0..8)
    for( disk = 0; disk < 9; ++disk ) {
      deltaXK.at( det ).at( disk ) =  ( -1. * diskZPositions.at( det ).at( disk ) * deltaXT.at( det ) / endcapLength )
	-  ( deltaX0.at( det ) )  +  2. * sumOverSinThetaY.at( det ).at( disk ) / 8.;// / 2.; // @@@@@@@
    }

    // deltaY_0
    deltaY0.at( det ) = 2. * ( sumOverZSquared.at( det ) * doubleSumOverCosThetaY.at( det ) - sumOverZ.at( det ) * doubleSumOverCosThetaYZ.at( det ) )
      / ( 8. * ( pow( sumOverZ.at( det ), 2 ) - 9. * sumOverZSquared.at( det ) ) );// / 2.; // @@@@@@@

    // deltaY_t
    deltaYT.at( det ) = 2. * endcapLength * ( 9. * doubleSumOverCosThetaYZ.at( det ) - sumOverZ.at( det ) * doubleSumOverCosThetaY.at( det ) )
      / ( 8. * ( pow( sumOverZ.at( det ), 2 ) - 9. * sumOverZSquared.at( det ) ) );// / 2.; // @@@@@@@

    // deltaY_k (k=0..8)
    for( disk = 0; disk < 9; ++disk ) {
      deltaYK.at( det ).at( disk ) =  ( -1. * diskZPositions.at( det ).at( disk ) * deltaYT.at( det ) / endcapLength )
 	-  ( deltaY0.at( det ) )  -  2. * sumOverCosThetaY.at( det ).at( disk ) / 8.;// / 2.; // @@@@@@@
    }


    // the beam parameters deltaTA & deltaTB
    for( beam = 0; beam < 8; ++beam ) {

      deltaTA.at( det ).at( beam ) = deltaPhi0.at( det ) 
	- ( kSumOverPhi.at( det ).at( beam ) * sumOverZSquared.at( det ) - kSumOverPhiZ.at( det ).at( beam ) * sumOverZ.at( det ) )
	/ ( pow( sumOverZ.at( det ), 2 ) - 9. * sumOverZSquared.at( det ) )
	+ ( cos( beamPhiPositions.at( det ).at( beam ) ) * deltaY0.at( det ) - sin( beamPhiPositions.at( det ).at( beam ) ) * deltaX0.at( det ) ) / radius.at( 0 ); // for ring 4..
        // + ( cos( beamPhiPositions.at( det ).at( beam ) ) * deltaY0.at( det ) - sin( beamPhiPositions.at( det ).at( beam ) ) * deltaX0.at( det ) ) / radius.at( 1 ); // ...and ring 6


      deltaTB.at( det ).at( beam ) = -1. * deltaPhiT.at( det ) - deltaPhi0.at( det )
	- ( kSumOverPhi.at( det ).at( beam ) * sumOverZ.at( det ) - 9. * kSumOverPhiZ.at( det ).at( beam ) )
	* endcapLength / ( pow( sumOverZ.at( det ), 2 ) - 9. * sumOverZSquared.at( det ) )
	- ( kSumOverPhiZ.at( det ).at( beam ) * sumOverZ.at( det ) -  kSumOverPhi.at( det ).at( beam ) * sumOverZSquared.at( det ) )
	/ ( pow( sumOverZ.at( det ), 2 ) - 9. * sumOverZSquared.at( det ) )
	+ ( ( deltaXT.at( det ) + deltaX0.at( det ) ) * sin( beamPhiPositions.at( det ).at( beam ) ) - ( deltaYT.at( det ) + deltaY0.at( det ) ) * cos( beamPhiPositions.at( det ).at( beam ) ) )
	/ radius.at( 0 ); // for ring4..
      // + ( ( deltaXT.at( det ) + deltaX0.at( det ) ) * sin( beamPhiPositions.at( det ).at( beam ) ) - ( deltaYT.at( det ) + deltaY0.at( det ) ) * cos( beamPhiPositions.at( det ).at( beam ) ) )
      // / radius.at( 1 ); // ..and ring6

    }


  }


  // fill the result
  LASEndcapAlignmentParameterSet theResult;

  // for the moment we fill only the values, not the errors

  // disk parameters
  for( det = 0; det < 2; ++det ) {
    for( disk = 0; disk < 9; ++disk ) {
      // the rotation parameters: deltaPhi_k
      theResult.GetDiskParameter( det, disk, 0 ).first = deltaPhiK.at( det ).at( disk );
      // the x offsets: deltaX_k
      theResult.GetDiskParameter( det, disk, 1 ).first = deltaXK.at( det ).at( disk );
      // the y offsets: deltaY_k
      theResult.GetDiskParameter( det, disk, 2 ).first = deltaYK.at( det ).at( disk );
    }
  }

  // global parameters
  for( int det = 0; det < 2; ++det ) {
    theResult.GetGlobalParameter( det, 0 ).first = deltaPhi0.at( det );
    theResult.GetGlobalParameter( det, 1 ).first = deltaPhiT.at( det );
    theResult.GetGlobalParameter( det, 2 ).first = deltaX0.at( det );
    theResult.GetGlobalParameter( det, 3 ).first = deltaXT.at( det );
    theResult.GetGlobalParameter( det, 4 ).first = deltaY0.at( det );
    theResult.GetGlobalParameter( det, 5 ).first = deltaYT.at( det );
  }

  // beam parameters
  for( int det = 0; det < 2; ++det ) {
    for( int beam = 0; beam < 8; ++beam ) {
      theResult.GetBeamParameter( det, 1 /*R6*/, beam, 0 ).first = deltaTA.at( det ).at( beam ); ////////////////////////////////////////////
      theResult.GetBeamParameter( det, 1 /*R6*/, beam, 1 ).first = deltaTB.at( det ).at( beam );////////////////////////////////////////////
    }
  }

  std::cout << " [LASEndcapAlgorithm::CalculateParameters] -- Done." << std::endl;

  return( theResult );

}





///
/// for a given set of endcap alignment parameters "endcapParameters",
/// this function returns the global phi offset from nominalPosition
/// for a module specified by (det,ring,beam,disk)
///
double LASEndcapAlgorithm::GetAlignmentParameterCorrection( int det, int ring, int beam, int disk, LASGlobalData<LASCoordinateSet>& nominalCoordinates, LASEndcapAlignmentParameterSet& endcapParameters ) {
  
  // ring dependent radius, to be softcoded...
  const double radius = ring==0 ? 564. : 840.;
  const double endcapLength = 1345.; // mm
  
  // the correction to phi from the endcap algorithm;
  // it is defined such that the correction is to be subtracted
  double phiCorrection = 0.;
  
  // plain disk phi
  phiCorrection += endcapParameters.GetDiskParameter( det, disk, 0 ).first;
  
  // phi component from x deviation
  phiCorrection -= sin( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) / radius * endcapParameters.GetDiskParameter( det, disk, 1 ).first;
  
  // phi component from y deviation
  phiCorrection += cos( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) / radius * endcapParameters.GetDiskParameter( det, disk, 2 ).first;
  
  // phi correction from global phi
  phiCorrection += endcapParameters.GetGlobalParameter( det, 0 ).first;
  
  // correction from global x deviation
  phiCorrection -= sin( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) / radius * endcapParameters.GetGlobalParameter( det, 2 ).first;
  
  // correction from global y deviation
  phiCorrection += cos( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) / radius * endcapParameters.GetGlobalParameter( det, 4 ).first;
  
  // correction from global torsion
  phiCorrection += nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetZ() / endcapLength * endcapParameters.GetGlobalParameter( det, 1 ).first;
  
  // correction from global x shear
  phiCorrection -= nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetZ() / endcapLength / radius *
    sin( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) * endcapParameters.GetGlobalParameter( det, 3 ).first;
  
  // correction from global y shear
  phiCorrection += nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetZ() / endcapLength / radius *
    cos( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ) * endcapParameters.GetGlobalParameter( det, 5 ).first;
  
  // correction from beam parameters
  phiCorrection += ( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetZ() / endcapLength - 1. ) * endcapParameters.GetBeamParameter( det, 1, beam, 0 ).first;
  phiCorrection += nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetZ() / endcapLength * endcapParameters.GetBeamParameter( det, 1, beam, 1 ).first;
  
  return phiCorrection;

}
