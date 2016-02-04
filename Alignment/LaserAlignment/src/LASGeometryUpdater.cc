
#include "Alignment/LaserAlignment/interface/LASGeometryUpdater.h"


///
/// constructor providing access to the nominal coordinates and the constants -
/// for the moment, both objects are passed here, later it
/// should be sufficient to pass only the constants and to fill
/// the nominalCoordinates locally from that
///
LASGeometryUpdater::LASGeometryUpdater( LASGlobalData<LASCoordinateSet>& aNominalCoordinates, LASConstants& aLasConstants ) {
  
  nominalCoordinates = aNominalCoordinates;
  isReverseDirection = false;
  isMisalignmentFromRefGeometry = false;
  lasConstants = aLasConstants;

}





///
/// this function reads the beam kinks from the lasConstants
/// and applies them to the set of measured global phi positions
///
void LASGeometryUpdater::ApplyBeamKinkCorrections( LASGlobalData<LASCoordinateSet>& measuredCoordinates ) const {

  /// first we apply the endcap beamsplitter kink corrections
  /// for TEC+/- in one go
  for( unsigned int det = 0; det < 2; ++det ) {
    for( unsigned int ring = 0; ring < 2; ++ring ) {
      for( unsigned int beam = 0; beam < 8; ++beam ) {

	// corrections have different sign for TEC+/-
	const double endcapSign = det==0 ? 1.: -1.;

	// the correction is applied to the last 4 disks
	for( unsigned int disk = 5; disk < 9; ++disk ) {

	  measuredCoordinates.GetTECEntry( det, ring, beam, disk ).SetPhi(
	     measuredCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() - 
	     tan( lasConstants.GetEndcapBsKink( det, ring, beam ) ) / lasConstants.GetTecRadius( ring ) *
	     ( lasConstants.GetTecZPosition( det, disk ) - endcapSign * lasConstants.GetTecBsZPosition( det ) )
	  );
	
	}
      }
    }
  }


  /// alignment tube beamsplitter & mirror kink corrections
  /// TBD.

}





///
/// apply the endcap alignment parameters in the LASEndcapAlignmentParameterSet
/// to the Measurements (TEC2TEC only)
/// and the AlignableTracker object
///
void LASGeometryUpdater::EndcapUpdate( LASEndcapAlignmentParameterSet& endcapParameters,  LASGlobalData<LASCoordinateSet>& measuredCoordinates ) {

  // radius of TEC ring4 laser in mm
  const double radius = 564.;
  
  // loop objects and its variables
  LASGlobalLoop moduleLoop;
  int det = 0, beam = 0, disk = 0;



  // update the TEC2TEC measurements
  do {

    // the measured phi value for this module
    const double currentPhi = measuredCoordinates.GetTEC2TECEntry( det, beam, disk ).GetPhi();

    // the correction to phi from the endcap algorithm;
    // it is defined such that the correction is to be subtracted
    double phiCorrection = 0.;

    // plain phi component
    phiCorrection -= endcapParameters.GetDiskParameter( det, disk, 0 ).first;

    // phi component from x deviation
    phiCorrection += sin( nominalCoordinates.GetTEC2TECEntry( det, beam, disk ).GetPhi() ) / radius * endcapParameters.GetDiskParameter( det, disk, 1 ).first;

    // phi component from y deviation
    phiCorrection -= cos( nominalCoordinates.GetTEC2TECEntry( det, beam, disk ).GetPhi() ) / radius * endcapParameters.GetDiskParameter( det, disk, 2 ).first;

    measuredCoordinates.GetTEC2TECEntry( det, beam, disk ).SetPhi( currentPhi - phiCorrection );

  } while( moduleLoop.TEC2TECLoop( det, beam, disk ) );

}





///
/// merge the output from endcap and barrel algorithms
/// and update the AlignableTracker object
///
/// the AlignableTracker input object is expected to be perfectly aligned!!
///
void LASGeometryUpdater::TrackerUpdate( LASEndcapAlignmentParameterSet& endcapParameters,
					LASBarrelAlignmentParameterSet& barrelParameters, 
					AlignableTracker& theAlignableTracker ) {
  

  // this constant defines the sense of *ALL* translations/rotations
  // of the alignables in the AlignableTracker object
  const int direction = ( isReverseDirection || isMisalignmentFromRefGeometry ) ? -1 : 1;

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

  // the z positions of the lower/upper-z halfbarrel_end_faces / outer_TEC_disks (in mm)
  // indices are: halfbarrel (0-5), endface(0=lowerZ,1=upperZ)
  std::vector<std::vector<double> > halfbarrelEndfaceZPositions( 6, std::vector<double>( 2, 0. ) );
  halfbarrelEndfaceZPositions.at( 0 ).at( 0 ) = 1322.5;
  halfbarrelEndfaceZPositions.at( 0 ).at( 1 ) = 2667.5;
  halfbarrelEndfaceZPositions.at( 1 ).at( 0 ) = -2667.5;
  halfbarrelEndfaceZPositions.at( 1 ).at( 1 ) = -1322.5;
  halfbarrelEndfaceZPositions.at( 2 ).at( 0 ) = 300.; 
  halfbarrelEndfaceZPositions.at( 2 ).at( 1 ) = 700.;
  halfbarrelEndfaceZPositions.at( 3 ).at( 0 ) = -700.;  
  halfbarrelEndfaceZPositions.at( 3 ).at( 1 ) = -300.;
  halfbarrelEndfaceZPositions.at( 4 ).at( 0 ) = 300.;
  halfbarrelEndfaceZPositions.at( 4 ).at( 1 ) = 1090.;
  halfbarrelEndfaceZPositions.at( 5 ).at( 0 ) = -1090.;  
  halfbarrelEndfaceZPositions.at( 5 ).at( 1 ) = -300.;

  // z difference of half barrel end faces (= hb-length) in mm
  // do all this geometry stuff more intelligent later..
  std::vector<double> theBarrelLength( 6, 0. );
  theBarrelLength.at( 0 ) = 1345.; // TEC
  theBarrelLength.at( 1 ) = 1345.;
  theBarrelLength.at( 2 ) =  400.; // TIB
  theBarrelLength.at( 3 ) =  400.;
  theBarrelLength.at( 4 ) =  790.; // TOB
  theBarrelLength.at( 5 ) =  790.;


  // the halfbarrel centers as defined in the AlignableTracker object,
  // needed for offset corrections; code to be improved later
  std::vector<double> theHalfbarrelCenters( 6, 0. );
  for( int halfbarrel = 0; halfbarrel < 6; ++halfbarrel ) {
    theHalfbarrelCenters.at( halfbarrel ) = theHalfBarrels.at( halfbarrel )->globalPosition().z() * 10.; // in mm
  }

  

  // mm to cm conversion factor (use by division)
  const double fromMmToCm = 10.;

  // half barrel loop (no TECs -> det>1)
  for( int halfBarrel = 2; halfBarrel < 6; ++halfBarrel ) {
    
    // A word on the factors of -1 in the below move/rotate statements:
    //
    // Since it is not possible to misalign simulated low-level objects like SiStripDigis in CMSSW,
    // LAS monte carlo digis are always ideally aligned, and misalignment is introduced
    // by displacing the reference geometry (AlignableTracker) instead which is used for stripNumber->phi conversion.
    // Hence, in case MC are used in that way, factors of -1 must be introduced
    // for rotations around x,y and translations in x,y. The variable "xyDirection" takes care of this.
    // 
    // However, for rotations around z (phi) there is a complication:
    // in the analytical AlignmentTubeAlgorithm, the alignment parameter phi (z-rotation)
    // is defined such that a positive value results in a positive contribution to the residual. In the real detector,
    // the opposite is true. The resulting additional factor of -1 thus compensates for the abovementioned reference geometry effect.
    //
    // this behavior can be reversed using the "direction" factor.


    // rotation around x axis = (dy1-dy2)/L
    const align::Scalar rxLocal = ( barrelParameters.GetParameter( halfBarrel, 0, 2 ).first - barrelParameters.GetParameter( halfBarrel, 1, 2 ).first ) / theBarrelLength.at( halfBarrel );
    theHalfBarrels.at( halfBarrel )->rotateAroundLocalX( direction * rxLocal );

    // rotation around y axis = (dx2-dx1)/L
    const align::Scalar ryLocal = ( barrelParameters.GetParameter( halfBarrel, 1, 1 ).first - barrelParameters.GetParameter( halfBarrel, 0, 1 ).first ) / theBarrelLength.at( halfBarrel );
    theHalfBarrels.at( halfBarrel )->rotateAroundLocalY( direction * ryLocal );

    // average rotation around z axis = (dphi1+dphi2)/2
    const align::Scalar rzLocal = ( barrelParameters.GetParameter( halfBarrel, 0, 0 ).first + barrelParameters.GetParameter( halfBarrel, 1, 0 ).first ) / 2.;
    theHalfBarrels.at( halfBarrel )->rotateAroundLocalZ( -1. * direction * rzLocal ); // this is phi, additional -1 here, see comment above

    // now that the rotational displacements are removed, the remaining translational offsets can be corrected for.
    // for this, the effect of the rotations is subtracted from the measured endface offsets

    // @@@ the +/-/-/+ signs for the correction parameters are not yet fully understood - 
    // @@@ do they flip when switching from reference-geometry-misalignment to true misalignment???
    std::vector<double> correctedEndfaceOffsetsX( 2, 0. ); // lowerZ/upperZ endface
    correctedEndfaceOffsetsX.at( 0 ) = barrelParameters.GetParameter( halfBarrel, 0, 1 ).first
      + ( theHalfbarrelCenters.at( halfBarrel ) - halfbarrelEndfaceZPositions.at( halfBarrel ).at( 0 ) ) * ryLocal;
    correctedEndfaceOffsetsX.at( 1 ) = barrelParameters.GetParameter( halfBarrel, 1, 1 ).first
      - ( halfbarrelEndfaceZPositions.at( halfBarrel ).at( 1 ) - theHalfbarrelCenters.at( halfBarrel ) ) * ryLocal;

    std::vector<double> correctedEndfaceOffsetsY( 2, 0. ); // lowerZ/upperZ endface
    correctedEndfaceOffsetsY.at( 0 ) = barrelParameters.GetParameter( halfBarrel, 0, 2 ).first
      - ( theHalfbarrelCenters.at( halfBarrel ) - halfbarrelEndfaceZPositions.at( halfBarrel ).at( 0 ) ) * rxLocal;
    correctedEndfaceOffsetsY.at( 1 ) = barrelParameters.GetParameter( halfBarrel, 1, 2 ).first
      + ( halfbarrelEndfaceZPositions.at( halfBarrel ).at( 1 ) - theHalfbarrelCenters.at( halfBarrel ) ) * rxLocal;
    
    // average x displacement = (cd1+cd2)/2
    const align::GlobalVector dxLocal( ( correctedEndfaceOffsetsX.at( 0 ) + correctedEndfaceOffsetsX.at( 1 ) ) / 2. / fromMmToCm, 0., 0. );
    theHalfBarrels.at( halfBarrel )->move( direction * ( dxLocal ) );
      
    // average y displacement = (cd1+cd2)/s
    const align::GlobalVector dyLocal( 0., ( correctedEndfaceOffsetsY.at( 0 ) + correctedEndfaceOffsetsY.at( 1 ) ) / 2. / fromMmToCm, 0. );
    theHalfBarrels.at( halfBarrel )->move( direction * ( dyLocal ) );

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

  for( int det = 0; det < 2; ++det ) {

    const int& side = det;

    // step 1: apply the endcap algorithm parameters

    // factors of -1. within the move/rotate statements: see comment above.
    // difference here: in the the endcap algorithm, the alignment parameter phi (z-rotation) is defined
    // in the opposite sense compared with the AT algorithm, thus the factor of -1 applies also to phi rotations.

    for( int wheel = 0; wheel < 9; ++wheel ) {
      theEndcaps.at( det )->components().at( wheel )->rotateAroundLocalZ( direction * endcapParameters.GetDiskParameter( det, wheel, 0 ).first );
      const align::GlobalVector dXY( endcapParameters.GetDiskParameter( det, wheel, 1 ).first / fromMmToCm, endcapParameters.GetDiskParameter( det, wheel, 2 ).first / fromMmToCm, 0. );
      theEndcaps.at( det )->components().at( wheel )->move( direction * dXY );
    }



    // step 2: attach the innermost disk (disk 1) by rotating/moving the complete endcap
    
    // rotation around z of disk 1
    const align::Scalar dphi1 = barrelParameters.GetParameter( det, side, 0 ).first;
    theEndcaps.at( det )->rotateAroundLocalZ( -1. * direction * dphi1 ); // dphi1 is from the AT algorithm, so additional sign (s.above)
    
    // displacement in x,y of disk 1
    const align::GlobalVector dxy1( barrelParameters.GetParameter( det, side, 1 ).first / fromMmToCm, 
				    barrelParameters.GetParameter( det, side, 2 ).first / fromMmToCm,
				    0. );
    theEndcaps.at( det )->move( direction * dxy1 );




    // determine the resulting phi, x, y of disk 9 after step 2
    // the wrong sign for TEC- is soaked up by the reducedZ in step 3
    const align::Scalar resultingPhi9 = barrelParameters.GetParameter( det, 1, 0 ).first - barrelParameters.GetParameter( det, 0, 0 ).first;
    const align::GlobalVector resultingXY9( ( barrelParameters.GetParameter( det, 1, 1 ).first - barrelParameters.GetParameter( det, 0, 1 ).first ) / fromMmToCm,
					    ( barrelParameters.GetParameter( det, 1, 2 ).first - barrelParameters.GetParameter( det, 0, 2 ).first ) / fromMmToCm,
					    0. );
    
    // step 3: twist and shear back
    
    // the individual rotation/movement of the wheels is a function of their z-position
    for( int wheel = 0; wheel < 9; ++wheel ) {
      const double reducedZ = ( wheelZPositions.at( det ).at( wheel ) - wheelZPositions.at( det ).at( 0 ) ) / theBarrelLength.at( det ) * fromMmToCm;
      theEndcaps.at( det )->components().at( wheel )->rotateAroundLocalZ( -1. * direction * reducedZ * resultingPhi9 ); // twist
      theEndcaps.at( det )->components().at( wheel )->move( direction * reducedZ * resultingXY9 ); // shear
    }

  } 

}





///
///
///
void LASGeometryUpdater::SetReverseDirection( bool isSet ) {

  isReverseDirection = isSet;

}





///
///
///
void LASGeometryUpdater::SetMisalignmentFromRefGeometry( bool isSet ) {

  isMisalignmentFromRefGeometry = isSet;

}
