
#include "Alignment/LaserAlignment/src/LASGeometryUpdater.h"


///
///
///
LASGeometryUpdater::LASGeometryUpdater() {
}





///
/// apply the endcap alignment parameters in the LASEndcapAlignmentParameterSet
/// to the Measurements (TEC2TEC only) and the AlignableTracker object
///
void LASGeometryUpdater::EndcapUpdate( LASEndcapAlignmentParameterSet& endcapParameters,  LASGlobalData<LASCoordinateSet>& measuredCoordinates, AlignableTracker& theAlignableTracker ) {


  //
  // THIS IS A CONSTRUCTION SITE!
  //



  // edit the measurement
  
  for( int beam = 0; beam < 8; ++beam ) {
    for( int disk = 0; disk < 5; ++disk ) {
      std::cout << "BEAM " << beam << " DISK " << disk << ": " << measuredCoordinates.GetTEC2TECEntry( 0, beam, disk ).GetPhi() << std::endl;
    }
    std::cout << std::endl;
  }





  // edit the AlignableTracker object
  
  // conversion factor mm -> cm
  const double fromMmToCm = 10.;

  // access the endcaps
  const align::Alignables& theEndcaps = theAlignableTracker.endCaps();
  
  // now apply the alignment parameters
  // factors of -1. within the move() statements are determined empirically... should be understood exactly where this comes from...
  for( int det = 0; det < 2; ++det ) {
    
    // move and turn each endcap
    for( int wheel = 0; wheel < 9; ++wheel ) {
      theEndcaps.at( det )->components().at( wheel )->rotateAroundLocalZ( endcapParameters.GetDiskParameter( det, wheel, 0 ).first );
      const align::GlobalVector dXY( endcapParameters.GetDiskParameter( det, wheel, 1 ).first / fromMmToCm, endcapParameters.GetDiskParameter( det, wheel, 2 ).first / fromMmToCm, 0. );
      theEndcaps.at( det )->components().at( wheel )->move( -1. * dXY );
    }
    
  }
  
}





///
/// merge the output from endcap and barrel algorithms
/// and update the AlignableTracker object
///
/// the AlignableTracker is expected to be perfectly aligned!!
///
void LASGeometryUpdater::TrackerUpdate( LASEndcapAlignmentParameterSet& endcapParameters,
					LASBarrelAlignmentParameterSet& barrelParameters, 
					AlignableTracker& theAlignableTracker ) {
  

  // first, we access the half barrels of TIB and TOB
  const align::Alignables& theOuterHalfBarrels = theAlignableTracker.outerHalfBarrels();
  const align::Alignables& theInnerHalfBarrels = theAlignableTracker.innerHalfBarrels();

  // then the TECs and treat them also as half barrels 
  const align::Alignables& theEndcaps = theAlignableTracker.endCaps();

  // re-arrange to match the structure in LASBarrelAlignmentParameterSet and simplify the loop
  // 2 (TIB+), 3 (TIB-), 4 (TOB+), 5 (TOB-)
  std::vector<Alignable*> theHalfBarrels( 6 );
  theHalfBarrels.at( 0 ) = theEndcaps.at( 0 ); // TEC+
  theHalfBarrels.at( 1 ) = theEndcaps.at( 1 ); // TEC-
  theHalfBarrels.at( 2 ) = theInnerHalfBarrels.at( 1 ); // TIB+
  theHalfBarrels.at( 3 ) = theInnerHalfBarrels.at( 0 ); // TIB-
  theHalfBarrels.at( 4 ) = theOuterHalfBarrels.at( 1 ); // TOB+
  theHalfBarrels.at( 5 ) = theOuterHalfBarrels.at( 0 ); // TOB-

  // z difference of half barrel end faces (= hb-length) in mm
  // do this more intelligent later..
  std::vector<double> theBarrelLength( 6, 0. );
  theBarrelLength.at( 0 ) = 1348.65; // TEC
  theBarrelLength.at( 1 ) = 1348.65;
  theBarrelLength.at( 2 ) = 400.;  // TIB
  theBarrelLength.at( 3 ) = 400.;
  theBarrelLength.at( 4 ) = 790.;  // TOB
  theBarrelLength.at( 5 ) = 790.;

  const double fromMmToCm = 10.;

  // half barrel loop (no TECs -> det>1)
  for( int halfBarrel = 2; halfBarrel < 6; ++halfBarrel ) {
    
    // average x displacement = (dx1+dx2)/2
    const align::GlobalVector dxLocal( ( barrelParameters.GetParameter( halfBarrel, 0, 1 ).first + barrelParameters.GetParameter( halfBarrel, 1, 1 ).first ) / fromMmToCm / 2., 0., 0. );
    theHalfBarrels.at( halfBarrel )->move( -1. * dxLocal );

    // average y displacement = (dy1+dy2)/2
    const align::GlobalVector dyLocal( 0., ( barrelParameters.GetParameter( halfBarrel, 0, 2 ).first + barrelParameters.GetParameter( halfBarrel, 1, 2 ).first ) / fromMmToCm / 2., 0. );
    theHalfBarrels.at( halfBarrel )->move( -1. * dyLocal );

    // rotation around x axis = (dy2-dy1)/L
    const align::Scalar rxLocal = ( barrelParameters.GetParameter( halfBarrel, 1, 2 ).first - barrelParameters.GetParameter( halfBarrel, 0, 2 ).first ) / theBarrelLength.at( halfBarrel );
    theHalfBarrels.at( halfBarrel )->rotateAroundLocalX( rxLocal );

    // rotation around y axis = (dx1-dx2)/L
    const align::Scalar ryLocal = ( barrelParameters.GetParameter( halfBarrel, 0, 1 ).first - barrelParameters.GetParameter( halfBarrel, 1, 1 ).first ) / theBarrelLength.at( halfBarrel );
    theHalfBarrels.at( halfBarrel )->rotateAroundLocalY( ryLocal );

    // average rotation around z axis = (dphi1+dphi2)/2
    const align::Scalar rzLocal = ( barrelParameters.GetParameter( halfBarrel, 0, 0 ).first + barrelParameters.GetParameter( halfBarrel, 1, 0 ).first ) / 2.;
    theHalfBarrels.at( halfBarrel )->rotateAroundLocalZ( rzLocal );

  }


  // now fit the endcaps into that alignment tube frame. the strategy is the following:
  //
  // 1. apply the parameters from the endcap algorithm to the individual disks
  // 2. the tec as a whole is rotated and moved such that the innermost disk (1) (= halfbarrel inner endface) 
  //    reaches the position determined by the barrel algorithm. 
  // 3. then, the TEC is twisted and sheared until the outermost disk (9) reaches nominal position 
  //    (since it has been fixed there in the alignment tube frame). this resolves any common
  //    shear and torsion within the TEC coordinate frame.


  // shortcut to the z positions of the disks' mechanical structures (TEC+/-, 9*disk)
  std::vector<std::vector<double> > wheelZPositions( 2, std::vector<double>( 9, 0. ) );
  for( int wheel = 0; wheel < 9; ++wheel ) {
    wheelZPositions.at( 0 ).at( wheel ) = theEndcaps.at( 0 )->components().at( wheel )->globalPosition().z();
    wheelZPositions.at( 1 ).at( wheel ) = theEndcaps.at( 1 )->components().at( wheel )->globalPosition().z();
  }

  // we can do this for both TECs in one go;
  // only real difference is the index change in the second argument to barrelParameters::GetParameter:
  // here the disk index changes from 0(+) to 1(-), since the end faces are sorted according to z (-->side=det)

  // factors of -1. within the move() statements are determined empirically... should be understood exactly where this comes from...

  for( int det = 0; det < 2; ++det ) {

    const int& side = det;

    // step 1: apply the endcap algorithm parameters
    for( int wheel = 0; wheel < 9; ++wheel ) {
      theEndcaps.at( det )->components().at( wheel )->rotateAroundLocalZ( endcapParameters.GetDiskParameter( det, wheel, 0 ).first );
      const align::GlobalVector dXY( endcapParameters.GetDiskParameter( det, wheel, 1 ).first / fromMmToCm, endcapParameters.GetDiskParameter( det, wheel, 2 ).first / fromMmToCm, 0. );
      theEndcaps.at( det )->components().at( wheel )->move( -1. * dXY );
    }


    // step 2: attach the innermost disk (1)
    
    // rotation around z of disk 1
    const align::Scalar dphi1 = barrelParameters.GetParameter( det, side, 0 ).first - endcapParameters.GetDiskParameter( det, 0, 0 ).first;
    theEndcaps.at( det )->rotateAroundLocalZ( dphi1 );
    
    // displacement in x,y of disk 1
    const align::GlobalVector dxy1( ( barrelParameters.GetParameter( det, side, 1 ).first - endcapParameters.GetDiskParameter( det, 0, 1 ).first ) / fromMmToCm, 
				    ( barrelParameters.GetParameter( det, side, 2 ).first - endcapParameters.GetDiskParameter( det, 0, 2 ).first ) / fromMmToCm,
				    0. );
    theEndcaps.at( det )->move( -1. * dxy1 );

    std::cout << "KKK: " << ( barrelParameters.GetParameter( det, side, 1 ).first - endcapParameters.GetDiskParameter( det, 0, 1 ).first ) / fromMmToCm
	       << "  " << ( barrelParameters.GetParameter( det, side, 2 ).first - endcapParameters.GetDiskParameter( det, 0, 2 ).first ) / fromMmToCm << std::endl; //#################3

    // determine the resulting phi, x, y of disk 9 after step 2
    const align::Scalar resultingPhi9 = endcapParameters.GetDiskParameter( det, 8, 0 ).first + dphi1; // better calculate this rather than use a getter
    const align::GlobalVector resultingXY9( theEndcaps.at( det )->components().at( 8 )->globalPosition().x(),
					    theEndcaps.at( det )->components().at( 8 )->globalPosition().y(),
					    0. );

    std::cout << "LLL: " << theEndcaps.at( det )->components().at( 8 )->globalPosition().x() << "  " << theEndcaps.at( det )->components().at( 8 )->globalPosition().y() << std::endl; //################
    
    
    // step 3: twist and shear back
    
    // the individual rotation/movement of the wheels is a function of their z-position
    for( int wheel = 0; wheel < 9; ++wheel ) {
      const double reducedZ = fabs( wheelZPositions.at( det ).at( wheel ) - wheelZPositions.at( det ).at( 0 ) ) / theBarrelLength.at( det ) * fromMmToCm;
      theEndcaps.at( det )->components().at( wheel )->rotateAroundLocalZ( -1. * reducedZ * resultingPhi9 ); // twist
      theEndcaps.at( det )->components().at( wheel )->move( -1. * reducedZ * resultingXY9 ); // shear
    }

  } 




  // A T T I C

  
//   // this should now give all zero
//   std::cout << theEndcaps.at( 0 )->components().at( 8 )->rotation().xy() << std::endl; /////////////////////////////////
//   std::cout << "000+: " << theEndcaps.at( 0 )->components().at( 8 )->globalPosition().x() << "  "
// 	    << theEndcaps.at( 0 )->components().at( 8 )->globalPosition().y() << std::endl; /////////////////////////////////
    
//   std::cout << "---------------------------------" << std::endl;

//   std::cout << theEndcaps.at( 1 )->components().at( 8 )->rotation().xy() << std::endl; /////////////////////////////////
//   std::cout << "000-: " << theEndcaps.at( 1 )->components().at( 8 )->globalPosition().x() << "  "
// 	    << theEndcaps.at( 1 )->components().at( 8 )->globalPosition().y() << std::endl; /////////////////////////////////
  
}
