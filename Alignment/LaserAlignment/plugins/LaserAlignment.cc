/** \file LaserAlignment.cc
 *  LAS reconstruction module
 *
 *  $Date: 2013/01/07 20:26:37 $
 *  $Revision: 1.47 $
 *  \author Maarten Thomas
 *  \author Jan Olzem
 */

#include "Alignment/LaserAlignment/plugins/LaserAlignment.h"
#include "FWCore/Framework/interface/Run.h"




///
///
///
LaserAlignment::LaserAlignment( edm::ParameterSet const& theConf ) :
  theEvents(0), 
  theDoPedestalSubtraction( theConf.getUntrackedParameter<bool>( "SubtractPedestals", true ) ),
  theUseMinuitAlgorithm( theConf.getUntrackedParameter<bool>( "RunMinuitAlignmentTubeAlgorithm", false ) ),
  theApplyBeamKinkCorrections( theConf.getUntrackedParameter<bool>( "ApplyBeamKinkCorrections", true ) ),
  peakFinderThreshold( theConf.getUntrackedParameter<double>( "PeakFinderThreshold", 10. ) ),
  enableJudgeZeroFilter( theConf.getUntrackedParameter<bool>( "EnableJudgeZeroFilter", true ) ),
  judgeOverdriveThreshold( theConf.getUntrackedParameter<unsigned int>( "JudgeOverdriveThreshold", 220 ) ),
  updateFromInputGeometry( theConf.getUntrackedParameter<bool>( "UpdateFromInputGeometry", false ) ),
  misalignedByRefGeometry( theConf.getUntrackedParameter<bool>( "MisalignedByRefGeometry", false ) ),
  theStoreToDB ( theConf.getUntrackedParameter<bool>( "SaveToDbase", false ) ),
  theDigiProducersList( theConf.getParameter<std::vector<edm::ParameterSet> >( "DigiProducersList" ) ),
  theSaveHistograms( theConf.getUntrackedParameter<bool>( "SaveHistograms", false ) ),
  theCompression( theConf.getUntrackedParameter<int>( "ROOTFileCompression", 1 ) ),
  theFileName( theConf.getUntrackedParameter<std::string>( "ROOTFileName", "test.root" ) ),
  theMaskTecModules( theConf.getUntrackedParameter<std::vector<unsigned int> >( "MaskTECModules" ) ),
  theMaskAtModules( theConf.getUntrackedParameter<std::vector<unsigned int> >( "MaskATModules" ) ),
  theSetNominalStrips( theConf.getUntrackedParameter<bool>( "ForceFitterToNominalStrips", false ) ),
  theLasConstants( theConf.getUntrackedParameter<std::vector<edm::ParameterSet> >( "LaserAlignmentConstants" ) ),
  theFile(),
  theAlignableTracker(),
  theAlignRecordName( "TrackerAlignmentRcd" ),
  theErrorRecordName( "TrackerAlignmentErrorExtendedRcd" ),
  firstEvent_(true),
  theParameterSet( theConf )
{


  std::cout << std::endl;
  std::cout <<   "=============================================================="
	    << "\n===         LaserAlignment module configuration            ==="
	    << "\n"
	    << "\n    Write histograms to file       = " << (theSaveHistograms?"true":"false")
	    << "\n    Histogram file name            = " << theFileName
	    << "\n    Histogram file compression     = " << theCompression
	    << "\n    Subtract pedestals             = " << (theDoPedestalSubtraction?"true":"false")
	    << "\n    Run Minuit AT algorithm        = " << (theUseMinuitAlgorithm?"true":"false")
	    << "\n    Apply beam kink corrections    = " << (theApplyBeamKinkCorrections?"true":"false")
	    << "\n    Peak Finder Threshold          = " << peakFinderThreshold
	    << "\n    EnableJudgeZeroFilter          = " << (enableJudgeZeroFilter?"true":"false")
	    << "\n    JudgeOverdriveThreshold        = " << judgeOverdriveThreshold
	    << "\n    Update from input geometry     = " << (updateFromInputGeometry?"true":"false")
	    << "\n    Misalignment from ref geometry = " << (misalignedByRefGeometry?"true":"false")
	    << "\n    Number of TEC modules masked   = " << theMaskTecModules.size() << " (s. below list if > 0)"
	    << "\n    Number of AT modules masked    = " << theMaskAtModules.size()  << " (s. below list if > 0)"
	    << "\n    Store to database              = " << (theStoreToDB?"true":"false")
	    << "\n    ----------------------------------------------- ----------"
	    << (theSetNominalStrips?"\n    Set strips to nominal       =  true":"\n")
	    << "\n=============================================================" << std::endl;

  // tell about masked modules
  if( theMaskTecModules.size() ) {
    std::cout << " ===============================================================================================\n" << std::flush;
    std::cout << " The following " << theMaskTecModules.size() << " TEC modules have been masked out and will not be considered by the TEC algorithm:\n " << std::flush;
    for( std::vector<unsigned int>::iterator moduleIt = theMaskTecModules.begin(); moduleIt != theMaskTecModules.end(); ++moduleIt ) {
      std::cout << *moduleIt << (moduleIt!=--theMaskTecModules.end()?", ":"") << std::flush;
    }
    std::cout << std::endl << std::flush;
    std::cout << " ===============================================================================================\n\n" << std::flush;
  }
  if( theMaskAtModules.size() ) {
    std::cout << " ===============================================================================================\n" << std::flush;
    std::cout << " The following " << theMaskAtModules.size() << " AT modules have been masked out and will not be considered by the AT algorithm:\n " << std::flush;
    for( std::vector<unsigned int>::iterator moduleIt = theMaskAtModules.begin(); moduleIt != theMaskAtModules.end(); ++moduleIt ) {
      std::cout << *moduleIt << (moduleIt!=--theMaskAtModules.end()?", ":"") << std::flush;
    }
    std::cout << std::endl << std::flush;
    std::cout << " ===============================================================================================\n\n" << std::flush;
  }
  


  // alias for the Branches in the root files
  std::string alias ( theConf.getParameter<std::string>("@module_label") );  

  // declare the product to produce
  produces<TkLasBeamCollection, edm::InRun>( "tkLaserBeams" ).setBranchAlias( alias + "TkLasBeamCollection" );

  // switch judge's zero filter depending on cfg
  judge.EnableZeroFilter( enableJudgeZeroFilter );

  // set the upper threshold for zero suppressed data
  judge.SetOverdriveThreshold( judgeOverdriveThreshold );

}





///
///
///
LaserAlignment::~LaserAlignment() {

  if ( theSaveHistograms ) theFile->Write();
  if ( theFile ) { delete theFile; }
  if ( theAlignableTracker ) { delete theAlignableTracker; }

}





///
///
///
void LaserAlignment::beginJob() {


  // write sumed histograms to file (if selected in cfg)
  if( theSaveHistograms ) {

    // creating a new file
    theFile = new TFile( theFileName.c_str(), "RECREATE", "CMS ROOT file" );
    
    // initialize the histograms
    if ( theFile ) {
      theFile->SetCompressionLevel(theCompression);
      singleModulesDir = theFile->mkdir( "single modules" );
    } else 
      throw cms::Exception( " [LaserAlignment::beginJob]") << " ** ERROR: could not open file:"
							   << theFileName.c_str() << " for writing." << std::endl;

  }

  // detector id maps (hard coded)
  fillDetectorId();

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //   PROFILE, HISTOGRAM & FITFUNCTION INITIALIZATION
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // object used to build various strings for names and labels
  std::stringstream nameBuilder;

  // loop variables for use with LASGlobalLoop object
  int det, ring, beam, disk, pos;

  // loop TEC modules
  det = 0; ring = 0; beam = 0; disk = 0;
  do { // loop using LASGlobalLoop functionality
    // init the profiles
    pedestalProfiles.GetTECEntry( det, ring, beam, disk ).SetAllValuesTo( 0. );
    currentDataProfiles.GetTECEntry( det, ring, beam, disk ).SetAllValuesTo( 0. );
    collectedDataProfiles.GetTECEntry( det, ring, beam, disk ).SetAllValuesTo( 0. );

    // init the hit maps
    isAcceptedProfile.SetTECEntry( det, ring, beam, disk, 0 );
    numberOfAcceptedProfiles.SetTECEntry( det, ring, beam, disk, 0 );

    // create strings for histo names
    nameBuilder.clear();
    nameBuilder.str( "" );
    nameBuilder << "TEC";
    if( det == 0 ) nameBuilder << "+"; else nameBuilder << "-";
    nameBuilder << "_Ring";
    if( ring == 0 ) nameBuilder << "4"; else nameBuilder << "6";
    nameBuilder << "_Beam" << beam;
    nameBuilder << "_Disk" << disk;
    theProfileNames.SetTECEntry( det, ring, beam, disk, nameBuilder.str() );

    // init the histograms
    if( theSaveHistograms ) {
      nameBuilder << "_Histo";
      summedHistograms.SetTECEntry( det, ring, beam, disk, new TH1D( nameBuilder.str().c_str(), nameBuilder.str().c_str(), 512, 0, 512 ) );
      summedHistograms.GetTECEntry( det, ring, beam, disk )->SetDirectory( singleModulesDir );
    }
    
  } while ( moduleLoop.TECLoop( det, ring, beam, disk ) );


  // TIB & TOB section
  det = 2; beam = 0; pos = 0;
  do { // loop using LASGlobalLoop functionality
    // init the profiles
    pedestalProfiles.GetTIBTOBEntry( det, beam, pos ).SetAllValuesTo( 0. );
    currentDataProfiles.GetTIBTOBEntry( det, beam, pos ).SetAllValuesTo( 0. );
    collectedDataProfiles.GetTIBTOBEntry( det, beam, pos ).SetAllValuesTo( 0. );

    // init the hit maps
    isAcceptedProfile.SetTIBTOBEntry( det, beam, pos, 0 );
    numberOfAcceptedProfiles.SetTIBTOBEntry( det, beam, pos, 0 );

    // create strings for histo names
    nameBuilder.clear();
    nameBuilder.str( "" );
    if( det == 2 ) nameBuilder << "TIB"; else nameBuilder << "TOB";
    nameBuilder << "_Beam" << beam;
    nameBuilder << "_Zpos" << pos;

    theProfileNames.SetTIBTOBEntry( det, beam, pos, nameBuilder.str() );

    // init the histograms
    if( theSaveHistograms ) {
      nameBuilder << "_Histo";
      summedHistograms.SetTIBTOBEntry( det, beam, pos, new TH1D( nameBuilder.str().c_str(), nameBuilder.str().c_str(), 512, 0, 512 ) );
      summedHistograms.GetTIBTOBEntry( det, beam, pos )->SetDirectory( singleModulesDir );
    }
    
  } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );


  // TEC2TEC AT section
  det = 0; beam = 0; disk = 0;
  do { // loop using LASGlobalLoop functionality
    // init the profiles
    pedestalProfiles.GetTEC2TECEntry( det, beam, disk ).SetAllValuesTo( 0. );
    currentDataProfiles.GetTEC2TECEntry( det, beam, disk ).SetAllValuesTo( 0. );
    collectedDataProfiles.GetTEC2TECEntry( det, beam, disk ).SetAllValuesTo( 0. );
    
    // init the hit maps
    isAcceptedProfile.SetTEC2TECEntry( det, beam, disk, 0 );
    numberOfAcceptedProfiles.SetTEC2TECEntry( det, beam, disk, 0 );

    // create strings for histo names
    nameBuilder.clear();
    nameBuilder.str( "" );
    nameBuilder << "TEC(AT)";
    if( det == 0 ) nameBuilder << "+"; else nameBuilder << "-";
    nameBuilder << "_Beam" << beam;
    nameBuilder << "_Disk" << disk;
    theProfileNames.SetTEC2TECEntry( det, beam, disk, nameBuilder.str() );

    // init the histograms
    if( theSaveHistograms ) {
      nameBuilder << "_Histo";
      summedHistograms.SetTEC2TECEntry( det, beam, disk, new TH1D( nameBuilder.str().c_str(), nameBuilder.str().c_str(), 512, 0, 512 ) );
      summedHistograms.GetTEC2TECEntry( det, beam, disk )->SetDirectory( singleModulesDir );
    }
    
  } while( moduleLoop.TEC2TECLoop( det, beam, disk ) );

  firstEvent_ = true;
}






///
///
///
void LaserAlignment::produce(edm::Event& theEvent, edm::EventSetup const& theSetup)  {

  if (firstEvent_) {

    //Retrieve tracker topology from geometry
    edm::ESHandle<TrackerTopology> tTopoHandle;
    theSetup.get<IdealGeometryRecord>().get(tTopoHandle);
    const TrackerTopology* const tTopo = tTopoHandle.product();

    // access the tracker
    theSetup.get<TrackerDigiGeometryRecord>().get( theTrackerGeometry );
    theSetup.get<IdealGeometryRecord>().get( gD );
    
    // access pedestals (from db..) if desired
    edm::ESHandle<SiStripPedestals> pedestalsHandle;
    if( theDoPedestalSubtraction ) {
      theSetup.get<SiStripPedestalsRcd>().get( pedestalsHandle );
      fillPedestalProfiles( pedestalsHandle );
    }
    
    // global positions
    //  edm::ESHandle<Alignments> theGlobalPositionRcd;
    theSetup.get<TrackerDigiGeometryRecord>().getRecord<GlobalPositionRcd>().get( theGlobalPositionRcd );

    // select the reference geometry
    if( !updateFromInputGeometry ) {
      // the AlignableTracker object is initialized with the ideal geometry
      edm::ESHandle<GeometricDet> theGeometricDet;
      theSetup.get<IdealGeometryRecord>().get(theGeometricDet);
      TrackerGeomBuilderFromGeometricDet trackerBuilder;
      TrackerGeometry* theRefTracker = trackerBuilder.build(&*theGeometricDet, theParameterSet);
      
      theAlignableTracker = new AlignableTracker(&(*theRefTracker), tTopo);
    }
    else {
      // the AlignableTracker object is initialized with the input geometry from DB
      theAlignableTracker = new AlignableTracker( &(*theTrackerGeometry), tTopo );
    }
    
    firstEvent_ = false;
  }

  LogDebug("LaserAlignment") << "==========================================================="
			      << "\n   Private analysis of event #"<< theEvent.id().event() 
			      << " in run #" << theEvent.id().run();


  // do the Tracker Statistics to retrieve the current profiles
  fillDataProfiles( theEvent, theSetup );

  // index variables for the LASGlobalLoop object
  int det, ring, beam, disk, pos;

  //
  // first pre-loop on selected entries to find out
  // whether the TEC or the AT beams have fired
  // (pedestal profiles are left empty if false in cfg)
  // 


  // TEC+- (only ring 6)
  ring = 1;
  for( det = 0; det < 2; ++det ) {
    for( beam = 0; beam < 8; ++ beam ) {
      for( disk = 0; disk < 9; ++disk ) {
	if( judge.IsSignalIn( currentDataProfiles.GetTECEntry( det, ring, beam, disk ) - pedestalProfiles.GetTECEntry( det, ring, beam, disk ), 0 ) ) {
	  isAcceptedProfile.SetTECEntry( det, ring, beam, disk, 1 );
	}
	else { // assume no initialization
	  isAcceptedProfile.SetTECEntry( det, ring, beam, disk, 0 );
	}
      }
    }
  }

  // TIBTOB
  det = 2; beam = 0; pos = 0;
  do {
    // add current event's data and subtract pedestals
    if( judge.IsSignalIn( currentDataProfiles.GetTIBTOBEntry( det, beam, pos ) - pedestalProfiles.GetTIBTOBEntry( det, beam, pos ), getTIBTOBNominalBeamOffset( det, beam, pos ) ) ) {
      isAcceptedProfile.SetTIBTOBEntry( det, beam, pos, 1 );
    }
    else { // dto.
      isAcceptedProfile.SetTIBTOBEntry( det, beam, pos, 0 );
    }

  } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );




  // now come the beam finders
  bool isTECMode = isTECBeam();
  //  LogDebug( " [LaserAlignment::produce]" ) << "LaserAlignment::isTECBeam declares this event " << ( isTECMode ? "" : "NOT " ) << "a TEC event." << std::endl;
  std::cout << " [LaserAlignment::produce] -- LaserAlignment::isTECBeam declares this event " << ( isTECMode ? "" : "NOT " ) << "a TEC event." << std::endl;

  bool isATMode  = isATBeam();
  //  LogDebug( " [LaserAlignment::produce]" ) << "LaserAlignment::isATBeam declares this event "  << ( isATMode ? "" : "NOT " )  << "an AT event." << std::endl;
  std::cout << " [LaserAlignment::produce] -- LaserAlignment::isATBeam declares this event "  << ( isATMode ? "" : "NOT " )  << "an AT event." << std::endl;




  //
  // now pass the pedestal subtracted profiles to the judge
  // if they're accepted, add them on the collectedDataProfiles
  // (pedestal profiles are left empty if false in cfg)
  //


  // loop TEC+- modules
  det = 0; ring = 0; beam = 0; disk = 0;
  do {
    
    LogDebug( "[LaserAlignment::produce]" ) << "Profile is: " << theProfileNames.GetTECEntry( det, ring, beam, disk ) << "." << std::endl;

    // this now depends on the AT/TEC mode, is this a doubly hit module? -> look for it in vector<int> tecDoubleHitDetId
    // (ring == 0 not necessary but makes it a little faster)
    if( ring == 0 && find( tecDoubleHitDetId.begin(), tecDoubleHitDetId.end(), detectorId.GetTECEntry( det, ring, beam, disk ) ) != tecDoubleHitDetId.end() ) {

      if( isTECMode ) { // add profile to TEC collection
	// add current event's data and subtract pedestals
	if( judge.JudgeProfile( currentDataProfiles.GetTECEntry( det, ring, beam, disk ) - pedestalProfiles.GetTECEntry( det, ring, beam, disk ), 0 ) ) {
	  collectedDataProfiles.GetTECEntry( det, ring, beam, disk ) += currentDataProfiles.GetTECEntry( det, ring, beam, disk ) - pedestalProfiles.GetTECEntry( det, ring, beam, disk );
	  numberOfAcceptedProfiles.GetTECEntry( det, ring, beam, disk )++;
	}
      }
    }

    else { // not a doubly hit module, don't care about the mode
      // add current event's data and subtract pedestals
      if( judge.JudgeProfile( currentDataProfiles.GetTECEntry( det, ring, beam, disk ) - pedestalProfiles.GetTECEntry( det, ring, beam, disk ), 0 ) ) {
	collectedDataProfiles.GetTECEntry( det, ring, beam, disk ) += currentDataProfiles.GetTECEntry( det, ring, beam, disk ) - pedestalProfiles.GetTECEntry( det, ring, beam, disk );
	numberOfAcceptedProfiles.GetTECEntry( det, ring, beam, disk )++;
      }
    }
    
  } while( moduleLoop.TECLoop( det, ring, beam, disk ) );


  


  // loop TIB/TOB modules
  det = 2; beam = 0; pos = 0;
  do {
    
    LogDebug( "[LaserAlignment::produce]" ) << "Profile is: " << theProfileNames.GetTIBTOBEntry( det, beam, pos ) << "." << std::endl;
    
    // add current event's data and subtract pedestals
    if( judge.JudgeProfile( currentDataProfiles.GetTIBTOBEntry( det, beam, pos ) - pedestalProfiles.GetTIBTOBEntry( det, beam, pos ), getTIBTOBNominalBeamOffset( det, beam, pos ) ) ) {
      collectedDataProfiles.GetTIBTOBEntry( det, beam, pos ) += currentDataProfiles.GetTIBTOBEntry( det, beam, pos ) - pedestalProfiles.GetTIBTOBEntry( det, beam, pos );
      numberOfAcceptedProfiles.GetTIBTOBEntry( det, beam, pos )++;
    }

  } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );
  


  // loop TEC2TEC modules
  det = 0; beam = 0; disk = 0;
  do {
    
    LogDebug( "[LaserAlignment::produce]" ) << "Profile is: " << theProfileNames.GetTEC2TECEntry( det, beam, disk ) << "." << std::endl;

    // this again depends on the AT/TEC mode, is this a doubly hit module?
    // (ring == 0 not necessary but makes it a little faster)
    if( ring == 0 && find( tecDoubleHitDetId.begin(), tecDoubleHitDetId.end(), detectorId.GetTECEntry( det, ring, beam, disk ) ) != tecDoubleHitDetId.end() ) {

      if( isATMode ) { // add profile to TEC2TEC collection
	// add current event's data and subtract pedestals
	if( judge.JudgeProfile( currentDataProfiles.GetTEC2TECEntry( det, beam, disk ) - pedestalProfiles.GetTEC2TECEntry( det, beam, disk ), 0 ) ) {
	  collectedDataProfiles.GetTEC2TECEntry( det, beam, disk ) += currentDataProfiles.GetTEC2TECEntry( det, beam, disk ) - pedestalProfiles.GetTEC2TECEntry( det, beam, disk );
	  numberOfAcceptedProfiles.GetTEC2TECEntry( det, beam, disk )++;
	}
      }

    }     
    
    else { // not a doubly hit module, don't care about the mode
      // add current event's data and subtract pedestals
      if( judge.JudgeProfile( currentDataProfiles.GetTEC2TECEntry( det, beam, disk ) - pedestalProfiles.GetTEC2TECEntry( det, beam, disk ), 0 ) ) {
	collectedDataProfiles.GetTEC2TECEntry( det, beam, disk ) += currentDataProfiles.GetTEC2TECEntry( det, beam, disk ) - pedestalProfiles.GetTEC2TECEntry( det, beam, disk );
	numberOfAcceptedProfiles.GetTEC2TECEntry( det, beam, disk )++;
      }
    }
      

  } while( moduleLoop.TEC2TECLoop( det, beam, disk ) );



  // total event number counter
  theEvents++;

}





///
///
///
void LaserAlignment::endRunProduce( edm::Run& theRun, const edm::EventSetup& theSetup ) {


  std::cout << " [LaserAlignment::endRun] -- Total number of events processed: " << theEvents << std::endl;

  // for debugging only..
  DumpHitmaps( numberOfAcceptedProfiles );

  // index variables for the LASGlobalLoop objects
  int det, ring, beam, disk, pos;
    
  // measured positions container for the algorithms
  LASGlobalData<LASCoordinateSet> measuredCoordinates;
  
  // fitted peak positions in units of strips (pair for value,error)
  LASGlobalData<std::pair<float,float> > measuredStripPositions;
  
  // the peak finder, a pair (pos/posErr in units of strips) for its results, and the success confirmation
  LASPeakFinder peakFinder;
  peakFinder.SetAmplitudeThreshold( peakFinderThreshold );
  std::pair<double,double> peakFinderResults;
  bool isGoodFit;
  
  // tracker geom. object for calculating the global beam positions
  const TrackerGeometry& theTracker( *theTrackerGeometry );

  // fill LASGlobalData<LASCoordinateSet> nominalCoordinates
  CalculateNominalCoordinates();
  
  // for determining the phi errors
  //    ErrorFrameTransformer errorTransformer; // later...
  




  // do the fits for TEC+- internal
  det = 0; ring = 0; beam = 0; disk = 0;
  do {
    
    // do the fit
    isGoodFit = peakFinder.FindPeakIn( collectedDataProfiles.GetTECEntry( det, ring, beam, disk ), peakFinderResults,
				       summedHistograms.GetTECEntry( det, ring, beam, disk ), 0 ); // offset is 0 for TEC

    // now we have the measured positions in units of strips. 
    if( !isGoodFit ) std::cout << " [LaserAlignment::endRun] ** WARNING: Fit failed for TEC det: "
			       << det << ", ring: " << ring << ", beam: " << beam << ", disk: " << disk
			       << " (id: " << detectorId.GetTECEntry( det, ring, beam, disk ) << ")." << std::endl;

    

    // <- here we will later implement the kink corrections
      
    // access the tracker geometry for this module
    const DetId theDetId( detectorId.GetTECEntry( det, ring, beam, disk ) );
    const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTracker.idToDet( theDetId ) );
      
    if (theStripDet) {
      // first, set the measured coordinates to their nominal values
      measuredCoordinates.SetTECEntry( det, ring, beam, disk, nominalCoordinates.GetTECEntry( det, ring, beam, disk ) );
      
      if( isGoodFit ) { // convert strip position to global phi and replace the nominal phi value/error
	
	measuredStripPositions.GetTECEntry( det, ring, beam, disk ) = peakFinderResults;
	const float positionInStrips =  theSetNominalStrips ? 256. : peakFinderResults.first; // implementation of "ForceFitterToNominalStrips" config parameter
	const GlobalPoint& globalPoint = theStripDet->surface().toGlobal( theStripDet->specificTopology().localPosition( positionInStrips ) );
	measuredCoordinates.GetTECEntry( det, ring, beam, disk ).SetPhi( ConvertAngle( globalPoint.barePhi() ) );
	
	// const GlobalError& globalError = errorTransformer.transform( theStripDet->specificTopology().localError( peakFinderResults.first, pow( peakFinderResults.second, 2 ) ), theStripDet->surface() );
	// measuredCoordinates.GetTECEntry( det, ring, beam, disk ).SetPhiError( globalError.phierr( globalPoint ) );
	measuredCoordinates.GetTECEntry( det, ring, beam, disk ).SetPhiError( 0.00046  ); // PRELIMINARY ESTIMATE
	
      }
      else { // keep nominal position (middle-of-module) but set a giant phi error so that the module can be ignored by the alignment algorithm
	measuredStripPositions.GetTECEntry( det, ring, beam, disk ) = std::pair<float,float>( 256., 1000. );
	const GlobalPoint& globalPoint = theStripDet->surface().toGlobal( theStripDet->specificTopology().localPosition( 256. ) );
	measuredCoordinates.GetTECEntry( det, ring, beam, disk ).SetPhi( ConvertAngle( globalPoint.barePhi() ) );
	measuredCoordinates.GetTECEntry( det, ring, beam, disk ).SetPhiError( 1000. );
      }
    }
      
  } while( moduleLoop.TECLoop( det, ring, beam, disk ) );




  // do the fits for TIB/TOB
  det = 2; beam = 0; pos = 0;
  do {

    // do the fit
    isGoodFit = peakFinder.FindPeakIn( collectedDataProfiles.GetTIBTOBEntry( det, beam, pos ), peakFinderResults, 
				       summedHistograms.GetTIBTOBEntry( det, beam, pos ), getTIBTOBNominalBeamOffset( det, beam, pos ) );

    // now we have the measured positions in units of strips.
    if( !isGoodFit ) std::cout << " [LaserAlignment::endJob] ** WARNING: Fit failed for TIB/TOB det: "
			       << det << ", beam: " << beam << ", pos: " << pos 
			       << " (id: " << detectorId.GetTIBTOBEntry( det, beam, pos ) << ")." << std::endl;

      
    // <- here we will later implement the kink corrections
      
    // access the tracker geometry for this module
    const DetId theDetId( detectorId.GetTIBTOBEntry( det, beam, pos ) );
    const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTracker.idToDet( theDetId ) );
      
    if (theStripDet) {
      // first, set the measured coordinates to their nominal values
      measuredCoordinates.SetTIBTOBEntry( det, beam, pos, nominalCoordinates.GetTIBTOBEntry( det, beam, pos ) );
      
      if( isGoodFit ) { // convert strip position to global phi and replace the nominal phi value/error
	measuredStripPositions.GetTIBTOBEntry( det, beam, pos ) = peakFinderResults;
	const float positionInStrips =  theSetNominalStrips ? 256. + getTIBTOBNominalBeamOffset( det, beam, pos ) : peakFinderResults.first; // implementation of "ForceFitterToNominalStrips" config parameter
	const GlobalPoint& globalPoint = theStripDet->surface().toGlobal( theStripDet->specificTopology().localPosition( positionInStrips ) );
	measuredCoordinates.GetTIBTOBEntry( det, beam, pos ).SetPhi( ConvertAngle( globalPoint.barePhi() ) );
	measuredCoordinates.GetTIBTOBEntry( det, beam, pos ).SetPhiError( 0.00028 ); // PRELIMINARY ESTIMATE
      }
      else { // keep nominal position but set a giant phi error so that the module can be ignored by the alignment algorithm
	measuredStripPositions.GetTIBTOBEntry( det, beam, pos ) = std::pair<float,float>( 256. + getTIBTOBNominalBeamOffset( det, beam, pos ), 1000. );
	const GlobalPoint& globalPoint = theStripDet->surface().toGlobal( theStripDet->specificTopology().localPosition( 256. + getTIBTOBNominalBeamOffset( det, beam, pos ) ) );
	measuredCoordinates.GetTIBTOBEntry( det, beam, pos ).SetPhi( ConvertAngle( globalPoint.barePhi() ) );
	measuredCoordinates.GetTIBTOBEntry( det, beam, pos ).SetPhiError( 1000. );
      }
    }

  } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );




  // do the fits for TEC AT
  det = 0; beam = 0; disk = 0;
  do {

    // do the fit
    isGoodFit = peakFinder.FindPeakIn( collectedDataProfiles.GetTEC2TECEntry( det, beam, disk ), peakFinderResults,
				       summedHistograms.GetTEC2TECEntry( det, beam, disk ), getTEC2TECNominalBeamOffset( det, beam, disk ) );
    // now we have the positions in units of strips.
    if( !isGoodFit ) std::cout << " [LaserAlignment::endRun] ** WARNING: Fit failed for TEC2TEC det: "
			       << det << ", beam: " << beam << ", disk: " << disk
			       << " (id: " << detectorId.GetTEC2TECEntry( det, beam, disk ) << ")." << std::endl;


    // <- here we will later implement the kink corrections
    
    // access the tracker geometry for this module
    const DetId theDetId( detectorId.GetTEC2TECEntry( det, beam, disk ) );
    const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTracker.idToDet( theDetId ) );

    if (theStripDet) {
      // first, set the measured coordinates to their nominal values
      measuredCoordinates.SetTEC2TECEntry( det, beam, disk, nominalCoordinates.GetTEC2TECEntry( det, beam, disk ) );
      
      if( isGoodFit ) { // convert strip position to global phi and replace the nominal phi value/error
	measuredStripPositions.GetTEC2TECEntry( det, beam, disk ) = peakFinderResults;
	const float positionInStrips =  theSetNominalStrips ? 256. + getTEC2TECNominalBeamOffset( det, beam, disk ) : peakFinderResults.first; // implementation of "ForceFitterToNominalStrips" config parameter
	const GlobalPoint& globalPoint = theStripDet->surface().toGlobal( theStripDet->specificTopology().localPosition( positionInStrips ) );
	measuredCoordinates.GetTEC2TECEntry( det, beam, disk ).SetPhi( ConvertAngle( globalPoint.barePhi() ) );
	measuredCoordinates.GetTEC2TECEntry( det, beam, disk ).SetPhiError( 0.00047 ); // PRELIMINARY ESTIMATE
      }
      else { // keep nominal position but set a giant phi error so that the module can be ignored by the alignment algorithm
	measuredStripPositions.GetTEC2TECEntry( det, beam, disk ) = std::pair<float,float>( 256. + getTEC2TECNominalBeamOffset( det, beam, disk ), 1000. );
	const GlobalPoint& globalPoint = theStripDet->surface().toGlobal( theStripDet->specificTopology().localPosition( 256. + getTEC2TECNominalBeamOffset( det, beam, disk ) ) );
	measuredCoordinates.GetTEC2TECEntry( det, beam, disk ).SetPhi( ConvertAngle( globalPoint.barePhi() ) );
	measuredCoordinates.GetTEC2TECEntry( det, beam, disk ).SetPhiError( 1000. );
      }
    }

  } while( moduleLoop.TEC2TECLoop( det, beam, disk ) );
  






  // see what we got (for debugging)
  //  DumpStripFileSet( measuredStripPositions );
  //  DumpPosFileSet( measuredCoordinates );


  



  // CALCULATE PARAMETERS AND UPDATE DB OBJECT
  // for beam kink corrections, reconstructing the geometry and updating the db object
  LASGeometryUpdater geometryUpdater( nominalCoordinates, theLasConstants );

  // apply all beam corrections
  if( theApplyBeamKinkCorrections ) geometryUpdater.ApplyBeamKinkCorrections( measuredCoordinates );

  // if we start with input geometry instead of IDEAL,
  // reverse the adjustments in the AlignableTracker object
  if( updateFromInputGeometry ) geometryUpdater.SetReverseDirection( true );

  // if we have "virtual" misalignment which is introduced via the reference geometry,
  // tell the LASGeometryUpdater to reverse x & y adjustments
  if( misalignedByRefGeometry ) geometryUpdater.SetMisalignmentFromRefGeometry( true );

  // run the endcap algorithm
  LASEndcapAlgorithm endcapAlgorithm;
  LASEndcapAlignmentParameterSet endcapParameters;


  // this basically sets all the endcap modules to be masked 
  // to their nominal positions (since endcapParameters is overall zero)
  if( theMaskTecModules.size() ) {
    ApplyEndcapMaskingCorrections( measuredCoordinates, nominalCoordinates, endcapParameters );
  }

  // run the algorithm
  endcapParameters = endcapAlgorithm.CalculateParameters( measuredCoordinates, nominalCoordinates );

  // 
  // loop to mask out events
  // DESCRIPTION:
  //

  // do this only if there are modules to be masked..
  if( theMaskTecModules.size() ) {
    
    const unsigned int nIterations = 30;
    for( unsigned int iteration = 0; iteration < nIterations; ++iteration ) {
      
      // set the endcap modules to be masked to their positions
      // according to the reconstructed parameters
      ApplyEndcapMaskingCorrections( measuredCoordinates, nominalCoordinates, endcapParameters );
      
      // modifications applied, so re-run the algorithm
      endcapParameters = endcapAlgorithm.CalculateParameters( measuredCoordinates, nominalCoordinates );
      
    }

  } 

  // these are now final, so:
  endcapParameters.Print();



  
  // do a pre-alignment of the endcaps (TEC2TEC only)
  // so that the alignment tube algorithms finds orderly disks
  geometryUpdater.EndcapUpdate( endcapParameters, measuredCoordinates );


  // the alignment tube algorithms, choose from config
  LASBarrelAlignmentParameterSet alignmentTubeParameters;
  // the MINUIT-BASED alignment tube algorithm
  LASBarrelAlgorithm barrelAlgorithm;
  // the ANALYTICAL alignment tube algorithm
  LASAlignmentTubeAlgorithm alignmentTubeAlgorithm;


  // this basically sets all the modules to be masked 
  // to their nominal positions (since alignmentTubeParameters is overall zero)
  if( theMaskAtModules.size() ) {
    ApplyATMaskingCorrections( measuredCoordinates, nominalCoordinates, alignmentTubeParameters );
  }

  if( theUseMinuitAlgorithm ) {
    // run the MINUIT-BASED alignment tube algorithm
    alignmentTubeParameters = barrelAlgorithm.CalculateParameters( measuredCoordinates, nominalCoordinates );
  }
  else {
    // the ANALYTICAL alignment tube algorithm
    alignmentTubeParameters = alignmentTubeAlgorithm.CalculateParameters( measuredCoordinates, nominalCoordinates );
  }



  // 
  // loop to mask out events
  // DESCRIPTION:
  //

  // do this only if there are modules to be masked..
  if( theMaskAtModules.size() ) {
    
    const unsigned int nIterations = 30;
    for( unsigned int iteration = 0; iteration < nIterations; ++iteration ) {
      
      // set the AT modules to be masked to their positions
      // according to the reconstructed parameters
      ApplyATMaskingCorrections( measuredCoordinates, nominalCoordinates, alignmentTubeParameters );
      
      // modifications applied, so re-run the algorithm
      if( theUseMinuitAlgorithm ) {
	alignmentTubeParameters = barrelAlgorithm.CalculateParameters( measuredCoordinates, nominalCoordinates );
      }
      else {
	alignmentTubeParameters = alignmentTubeAlgorithm.CalculateParameters( measuredCoordinates, nominalCoordinates );
      }
      
    }

  } 


  // these are now final, so:
  alignmentTubeParameters.Print();


  // combine the results and update the db object
  geometryUpdater.TrackerUpdate( endcapParameters, alignmentTubeParameters, *theAlignableTracker );
  




    

  /// laser hit section for trackbased interface
  ///
  /// due to the peculiar order of beams in TkLasBeamCollection,
  /// we cannot use the LASGlobalLoop object here
    
    
  // the collection container
  std::auto_ptr<TkLasBeamCollection> laserBeams( new TkLasBeamCollection );

  
  // first for the endcap internal beams
  for( det = 0; det < 2; ++det ) {
    for( ring = 0; ring < 2; ++ring ) {
      for( beam = 0; beam < 8; ++beam ) {
	
	// the beam and its identifier (see TkLasTrackBasedInterface TWiki)
	TkLasBeam currentBeam( 100 * det + 10 * beam + ring );
	
	// order the hits in the beam by increasing z
	const int firstDisk = det==0 ? 0 : 8;
	const int lastDisk  = det==0 ? 8 : 0;
	  
	// count upwards or downwards
	for( disk = firstDisk; det==0 ? disk <= lastDisk : disk >= lastDisk; det==0 ? ++disk : --disk ) {
	    
	  // detId for the SiStripLaserRecHit2D
	  const SiStripDetId theDetId( detectorId.GetTECEntry( det, ring, beam, disk ) );
	    
	  // need this to calculate the localPosition and its error
	  const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTracker.idToDet( theDetId ) );
	    
	  // the hit container
	  const SiStripLaserRecHit2D currentHit(
	    theStripDet->specificTopology().localPosition( measuredStripPositions.GetTECEntry( det, ring, beam, disk ).first ),
	    theStripDet->specificTopology().localError( measuredStripPositions.GetTECEntry( det, ring, beam, disk ).first, measuredStripPositions.GetTECEntry( det, ring, beam, disk ).second ),
	    theDetId
	  );

	  currentBeam.push_back( currentHit );

	}	  
	  
	laserBeams->push_back( currentBeam );
	  
      }
    }
  }
    
    

  // then, following the convention in TkLasTrackBasedInterface TWiki, the alignment tube beams;
  // they comprise hits in TIBTOB & TEC2TEC

  for( beam = 0; beam < 8; ++beam ) {

    // the beam and its identifier (see TkLasTrackBasedInterface TWiki)
    TkLasBeam currentBeam( 100 * 2 /*beamGroup=AT=2*/ + 10 * beam + 0 /*ring=0*/);


    // first: tec-
    det = 1;
    for( disk = 4; disk >= 0; --disk ) {
	
      // detId for the SiStripLaserRecHit2D
      const SiStripDetId theDetId( detectorId.GetTEC2TECEntry( det, beam, disk ) );
	
      // need this to calculate the localPosition and its error
      const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTracker.idToDet( theDetId ) );
	
      // the hit container
      const SiStripLaserRecHit2D currentHit(
        theStripDet->specificTopology().localPosition( measuredStripPositions.GetTEC2TECEntry( det, beam, disk ).first ),
	theStripDet->specificTopology().localError( measuredStripPositions.GetTEC2TECEntry( det, beam, disk ).first, measuredStripPositions.GetTEC2TECEntry( det, beam, disk ).second ),
	theDetId
      );

      currentBeam.push_back( currentHit );
	
    }

      
    // now TIB and TOB in one go
    for( det = 2; det < 4; ++det ) {
      for( pos = 5; pos >= 0; --pos ) { // stupidly, pos is defined from +z to -z in LASGlobalLoop
	  
	// detId for the SiStripLaserRecHit2D
	const SiStripDetId theDetId( detectorId.GetTIBTOBEntry( det, beam, pos ) );
	  
	// need this to calculate the localPosition and its error
	const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTracker.idToDet( theDetId ) );
	  
	// the hit container
	const SiStripLaserRecHit2D currentHit(
	  theStripDet->specificTopology().localPosition( measuredStripPositions.GetTIBTOBEntry( det, beam, pos ).first ),
	  theStripDet->specificTopology().localError( measuredStripPositions.GetTIBTOBEntry( det, beam, pos ).first, measuredStripPositions.GetTIBTOBEntry( det, beam, pos ).second ),
	  theDetId
	);

	currentBeam.push_back( currentHit );
	  
      }
    }
      

    // then: tec+
    det = 0;
    for( disk = 0; disk < 5; ++disk ) {
	
      // detId for the SiStripLaserRecHit2D
      const SiStripDetId theDetId( detectorId.GetTEC2TECEntry( det, beam, disk ) );
	
      // need this to calculate the localPosition and its error
      const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTracker.idToDet( theDetId ) );
	
      // the hit container
      const SiStripLaserRecHit2D currentHit(
        theStripDet->specificTopology().localPosition( measuredStripPositions.GetTEC2TECEntry( det, beam, disk ).first ),
	theStripDet->specificTopology().localError( measuredStripPositions.GetTEC2TECEntry( det, beam, disk ).first, measuredStripPositions.GetTEC2TECEntry( det, beam, disk ).second ),
	theDetId
      );

      currentBeam.push_back( currentHit );
	
    }



    // save this beam to the beamCollection
    laserBeams->push_back( currentBeam );
    
  } // (close beam loop)
  
  
  // now attach the collection to the run
  theRun.put( laserBeams, "tkLaserBeams" );
  


  

  // store the estimated alignment parameters into the DB
  // first get them
  Alignments* alignments =  theAlignableTracker->alignments();
  AlignmentErrorsExtended* alignmentErrors = theAlignableTracker->alignmentErrors();

  if ( theStoreToDB ) {

    std::cout << " [LaserAlignment::endRun] -- Storing the calculated alignment parameters to the DataBase:" << std::endl;

    // Call service
    edm::Service<cond::service::PoolDBOutputService> poolDbService;
    if( !poolDbService.isAvailable() ) // Die if not available
      throw cms::Exception( "NotAvailable" ) << "PoolDBOutputService not available";
    
    // Store

    //     if ( poolDbService->isNewTagRequest(theAlignRecordName) ) {
    //       poolDbService->createNewIOV<Alignments>( alignments, poolDbService->currentTime(), poolDbService->endOfTime(), theAlignRecordName );
    //     }
    //     else {
    //       poolDbService->appendSinceTime<Alignments>( alignments, poolDbService->currentTime(), theAlignRecordName );
    //     }
    poolDbService->writeOne<Alignments>( alignments, poolDbService->beginOfTime(), theAlignRecordName );

    //     if ( poolDbService->isNewTagRequest(theErrorRecordName) ) {
    //       poolDbService->createNewIOV<AlignmentErrorsExtended>( alignmentErrors, poolDbService->currentTime(), poolDbService->endOfTime(), theErrorRecordName );
    //     }
    //     else {
    //       poolDbService->appendSinceTime<AlignmentErrorsExtended>( alignmentErrors, poolDbService->currentTime(), theErrorRecordName );
    //     }
    poolDbService->writeOne<AlignmentErrorsExtended>( alignmentErrors, poolDbService->beginOfTime(), theErrorRecordName );

    std::cout << " [LaserAlignment::endRun] -- Storing done." << std::endl;
    
  }

}





///
///
///
void LaserAlignment::endJob() {
}





///
/// fills the module profiles (LASGlobalLoop<LASModuleProfile> currentDataProfiles)
/// from the event digi containers, distinguishing between SiStripDigi or SiStripRawDigi.
///
void LaserAlignment::fillDataProfiles( edm::Event const& theEvent, edm::EventSetup const& theSetup ) {

  // two handles for the two different kinds of digis
  edm::Handle< edm::DetSetVector<SiStripRawDigi> > theStripRawDigis;
  edm::Handle< edm::DetSetVector<SiStripDigi> > theStripDigis;

  bool isRawDigi = false;

  // indices for the LASGlobalLoop object
  int det = 0, ring = 0, beam = 0, disk = 0, pos = 0;

  // query config set and loop over all PSets in the VPSet
  for ( std::vector<edm::ParameterSet>::iterator itDigiProducersList = theDigiProducersList.begin(); itDigiProducersList != theDigiProducersList.end(); ++itDigiProducersList ) {

    std::string digiProducer = itDigiProducersList->getParameter<std::string>( "DigiProducer" );
    std::string digiLabel = itDigiProducersList->getParameter<std::string>( "DigiLabel" );
    std::string digiType = itDigiProducersList->getParameter<std::string>( "DigiType" );

    // now branch according to digi type (raw or processed);    
    // first we go for raw digis => SiStripRawDigi
    if( digiType == "Raw" ) {
      theEvent.getByLabel( digiProducer, digiLabel, theStripRawDigis );
      isRawDigi = true;
    }
    else if( digiType == "Processed" ) {
      theEvent.getByLabel( digiProducer, digiLabel, theStripDigis );
      isRawDigi = false;
    }
    else {
      throw cms::Exception( " [LaserAlignment::fillDataProfiles]") << " ** ERROR: Invalid digi type: \"" << digiType << "\" specified in configuration." << std::endl;
    }    



    // loop TEC internal modules
    det = 0; ring = 0; beam = 0; disk = 0;
    do {
      
      // first clear the profile
      currentDataProfiles.GetTECEntry( det, ring, beam, disk ).SetAllValuesTo( 0. );

      // retrieve the raw id of that module
      const int detRawId = detectorId.GetTECEntry( det, ring, beam, disk );
      
      if( isRawDigi ) { // we have raw SiStripRawDigis
	
	// search the digis for the raw id
	edm::DetSetVector<SiStripRawDigi>::const_iterator detSetIter = theStripRawDigis->find( detRawId );
	if( detSetIter == theStripRawDigis->end() ) {
	  throw cms::Exception( "[Laser Alignment::fillDataProfiles]" ) << " ** ERROR: No raw DetSet found for det: " << detRawId << "." << std::endl;
	}
      
	// fill the digis to the profiles
	edm::DetSet<SiStripRawDigi>::const_iterator digiRangeIterator = detSetIter->data.begin(); // for the loop
	edm::DetSet<SiStripRawDigi>::const_iterator digiRangeStart = digiRangeIterator; // save starting positions
	
	// loop all digis
	for (; digiRangeIterator != detSetIter->data.end(); ++digiRangeIterator ) {
	  const SiStripRawDigi& digi = *digiRangeIterator;
	  const int channel = distance( digiRangeStart, digiRangeIterator );
	  if ( channel >= 0 && channel < 512 ) currentDataProfiles.GetTECEntry( det, ring, beam, disk ).SetValue( channel, digi.adc() );
	  else throw cms::Exception( "[Laser Alignment::fillDataProfiles]" ) << " ** ERROR: raw digi channel: " << channel << " out of range for det: " << detRawId << "." << std::endl;
	}

      }

      else { // we have zero suppressed SiStripDigis

	// search the digis for the raw id
	edm::DetSetVector<SiStripDigi>::const_iterator detSetIter = theStripDigis->find( detRawId );
	
	// processed DetSets may be missing, just skip
 	if( detSetIter == theStripDigis->end() ) continue;

	// fill the digis to the profiles
	edm::DetSet<SiStripDigi>::const_iterator digiRangeIterator = detSetIter->data.begin(); // for the loop
	
	for(; digiRangeIterator != detSetIter->data.end(); ++digiRangeIterator ) {
	  const SiStripDigi& digi = *digiRangeIterator;
	  if ( digi.strip() < 512 ) currentDataProfiles.GetTECEntry( det, ring, beam, disk ).SetValue( digi.strip(), digi.adc() );
	  else throw cms::Exception( "[Laser Alignment::fillDataProfiles]" ) << " ** ERROR: digi strip: " << digi.strip() << " out of range for det: " << detRawId << "." << std::endl;
	}

      }

      
    } while( moduleLoop.TECLoop( det, ring, beam, disk ) );
    



    
    // loop TIBTOB modules
    det = 2; beam = 0; pos = 0;
    do {

      // first clear the profile
      currentDataProfiles.GetTIBTOBEntry( det, beam, pos ).SetAllValuesTo( 0. );

      // retrieve the raw id of that module
      const int detRawId = detectorId.GetTIBTOBEntry( det, beam, pos );
      
      if( isRawDigi ) { // we have raw SiStripRawDigis
	
	// search the digis for the raw id
	edm::DetSetVector<SiStripRawDigi>::const_iterator detSetIter = theStripRawDigis->find( detRawId );
	if( detSetIter == theStripRawDigis->end() ) {
	  throw cms::Exception( "[Laser Alignment::fillDataProfiles]" ) << " ** ERROR: No raw DetSet found for det: " << detRawId << "." << std::endl;
	}
      
	// fill the digis to the profiles
	edm::DetSet<SiStripRawDigi>::const_iterator digiRangeIterator = detSetIter->data.begin(); // for the loop
	edm::DetSet<SiStripRawDigi>::const_iterator digiRangeStart = digiRangeIterator; // save starting positions
	
	// loop all digis
	for (; digiRangeIterator != detSetIter->data.end(); ++digiRangeIterator ) {
	  const SiStripRawDigi& digi = *digiRangeIterator;
	  const int channel = distance( digiRangeStart, digiRangeIterator );
	  if ( channel >= 0 && channel < 512 ) currentDataProfiles.GetTIBTOBEntry( det, beam, pos ).SetValue( channel, digi.adc() );
	  else throw cms::Exception( "[Laser Alignment::fillDataProfiles]" ) << " ** ERROR: raw digi channel: " << channel << " out of range for det: " << detRawId << "." << std::endl;
	}

      }

      else { // we have zero suppressed SiStripDigis

	// search the digis for the raw id
	edm::DetSetVector<SiStripDigi>::const_iterator detSetIter = theStripDigis->find( detRawId );

	// processed DetSets may be missing, just skip
 	if( detSetIter == theStripDigis->end() ) continue;

	// fill the digis to the profiles
	edm::DetSet<SiStripDigi>::const_iterator digiRangeIterator = detSetIter->data.begin(); // for the loop
	
	for(; digiRangeIterator != detSetIter->data.end(); ++digiRangeIterator ) {
	  const SiStripDigi& digi = *digiRangeIterator;
	  if ( digi.strip() < 512 ) currentDataProfiles.GetTIBTOBEntry( det, beam, pos ).SetValue( digi.strip(), digi.adc() );
	  else throw cms::Exception( "[Laser Alignment::fillDataProfiles]" ) << " ** ERROR: digi strip: " << digi.strip() << " out of range for det: " << detRawId << "." << std::endl;
	}

      }

    } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );



    // loop TEC AT modules
    det = 0; beam = 0; disk = 0;
    do {

      // first clear the profile
      currentDataProfiles.GetTEC2TECEntry( det, beam, disk ).SetAllValuesTo( 0. );

      // retrieve the raw id of that module
      const int detRawId = detectorId.GetTEC2TECEntry( det, beam, disk );
    
      if( isRawDigi ) { // we have raw SiStripRawDigis
      
	// search the digis for the raw id
	edm::DetSetVector<SiStripRawDigi>::const_iterator detSetIter = theStripRawDigis->find( detRawId );
	if( detSetIter == theStripRawDigis->end() ) {
	  throw cms::Exception( "[Laser Alignment::fillDataProfiles]" ) << " ** ERROR: No raw DetSet found for det: " << detRawId << "." << std::endl;
	}
      
	// fill the digis to the profiles
	edm::DetSet<SiStripRawDigi>::const_iterator digiRangeIterator = detSetIter->data.begin(); // for the loop
	edm::DetSet<SiStripRawDigi>::const_iterator digiRangeStart = digiRangeIterator; // save starting positions
      
	// loop all digis
	for (; digiRangeIterator != detSetIter->data.end(); ++digiRangeIterator ) {
	  const SiStripRawDigi& digi = *digiRangeIterator;
	  const int channel = distance( digiRangeStart, digiRangeIterator );
	  if ( channel >= 0 && channel < 512 ) currentDataProfiles.GetTEC2TECEntry( det, beam, disk ).SetValue( channel, digi.adc() );
	  else throw cms::Exception( "[Laser Alignment::fillDataProfiles]" ) << " ** ERROR: raw digi channel: " << channel << " out of range for det: " << detRawId << "." << std::endl;
	}
      
      }
    
      else { // we have zero suppressed SiStripDigis
      
	// search the digis for the raw id
	edm::DetSetVector<SiStripDigi>::const_iterator detSetIter = theStripDigis->find( detRawId );
	
	// processed DetSets may be missing, just skip
 	if( detSetIter == theStripDigis->end() ) continue;
      
	// fill the digis to the profiles
	edm::DetSet<SiStripDigi>::const_iterator digiRangeIterator = detSetIter->data.begin(); // for the loop
      
	for(; digiRangeIterator != detSetIter->data.end(); ++digiRangeIterator ) {
	  const SiStripDigi& digi = *digiRangeIterator;
	  if ( digi.strip() < 512 ) currentDataProfiles.GetTEC2TECEntry( det, beam, disk ).SetValue( digi.strip(), digi.adc() );
	  else throw cms::Exception( "[Laser Alignment::fillDataProfiles]" ) << " ** ERROR: digi strip: " << digi.strip() << " out of range for det: " << detRawId << "." << std::endl;
	}
      
      }
    
    } while( moduleLoop.TEC2TECLoop( det, beam, disk ) );

  } // theDigiProducersList loop

}





///
/// This function fills the pedestal profiles (LASGlobalData<LASModuleProfiles> pedestalProfiles)
/// from the ESHandle (from file or DB)
///
/// Argument: readily connected SiStripPedestals object (get() alredy called)
/// The functionality inside the loops is basically taken from:
/// CommonTools/SiStripZeroSuppression/src/SiStripPedestalsSubtractor.cc
///
void LaserAlignment::fillPedestalProfiles( edm::ESHandle<SiStripPedestals>& pedestalsHandle ) {

  int det, ring, beam, disk, pos;

  // loop TEC modules (yet without AT)
  det = 0; ring = 0; beam = 0; disk = 0;
  do { // loop using LASGlobalLoop functionality
    SiStripPedestals::Range pedRange = pedestalsHandle->getRange( detectorId.GetTECEntry( det, ring, beam, disk ) );
    for( int strip = 0; strip < 512; ++strip ) {
      int thePedestal = int( pedestalsHandle->getPed( strip, pedRange ) );
      if( thePedestal > 895 ) thePedestal -= 1024;
      pedestalProfiles.GetTECEntry( det, ring, beam, disk ).SetValue( strip, thePedestal );
    }
  } while ( moduleLoop.TECLoop( det, ring, beam, disk ) );


  // TIB & TOB section
  det = 2; beam = 0; pos = 0;
  do { // loop using LASGlobalLoop functionality
    SiStripPedestals::Range pedRange = pedestalsHandle->getRange( detectorId.GetTIBTOBEntry( det, beam, pos ) );
    for( int strip = 0; strip < 512; ++strip ) {
      int thePedestal = int( pedestalsHandle->getPed( strip, pedRange ) );
      if( thePedestal > 895 ) thePedestal -= 1024;
      pedestalProfiles.GetTIBTOBEntry( det, beam, pos ).SetValue( strip, thePedestal );
    }
  } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );


  // TEC2TEC AT section
  det = 0; beam = 0; disk = 0;
  do { // loop using LASGlobalLoop functionality
    SiStripPedestals::Range pedRange = pedestalsHandle->getRange( detectorId.GetTEC2TECEntry( det, beam, disk ) );
    for( int strip = 0; strip < 512; ++strip ) {
      int thePedestal = int( pedestalsHandle->getPed( strip, pedRange ) );
      if( thePedestal > 895 ) thePedestal -= 1024;
      pedestalProfiles.GetTEC2TECEntry( det, beam, disk ).SetValue( strip, thePedestal );
    }
  } while( moduleLoop.TEC2TECLoop( det, beam, disk ) );

}





///
/// count useable profiles in TEC,
/// operates on LASGlobalData<int> LaserAlignment::isAcceptedProfile
/// to allow for more elaborate patterns in the future
///
bool LaserAlignment::isTECBeam( void ) {
  
  int numberOfProfiles = 0;

  int ring = 1; // search all ring6 modules for signals
  for( int det = 0; det < 2; ++det ) {
    for( int beam = 0; beam < 8; ++ beam ) {
      for( int disk = 0; disk < 9; ++disk ) {
	if( isAcceptedProfile.GetTECEntry( det, ring, beam, disk ) == 1 ) numberOfProfiles++;
      }
    }
  }

  LogDebug( "[LaserAlignment::isTECBeam]" ) << " Found: " << numberOfProfiles << "hits." << std::endl;
  std::cout << " [LaserAlignment::isTECBeam] -- Found: " << numberOfProfiles << " hits." << std::endl; ////

  if( numberOfProfiles > 10 ) return( true );
  return( false );
 
}





///
/// count useable profiles in TIBTOB,
/// operates on LASGlobalData<bool> LaserAlignment::isAcceptedProfile
/// to allow for more elaborate patterns in the future
///

bool LaserAlignment::isATBeam( void ) {

  int numberOfProfiles = 0;

  int det = 2; int beam = 0; int pos = 0; // search all TIB/TOB for signals
  do {
    if( isAcceptedProfile.GetTIBTOBEntry( det, beam, pos ) == 1 ) numberOfProfiles++;
  } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );

  LogDebug( "[LaserAlignment::isATBeam]" ) << " Found: " << numberOfProfiles << "hits." << std::endl;
  std::cout << " [LaserAlignment::isATBeam] -- Found: " << numberOfProfiles << " hits." << std::endl; /////

  if( numberOfProfiles > 10 ) return( true );
  return( false );
    
}





///
/// not all TIB & TOB modules are hit in the center;
/// this func returns the nominal beam offset locally on a module (in strips)
/// for the ProfileJudge and the LASPeakFinder in strips.
/// (offset = middle of module - nominal position)
///
/// the hard coded numbers will later be supplied by a special geometry class..
/// 
double LaserAlignment::getTIBTOBNominalBeamOffset( unsigned int det, unsigned int beam, unsigned int pos ) {

  if( det < 2 || det > 3 || beam > 7 || pos > 5 ) {
    throw cms::Exception( "[LaserAlignment::getTIBTOBNominalBeamOffset]" ) << " ERROR ** Called with nonexisting parameter set: det " << det << " beam " << beam << " pos " << pos << "." << std::endl;
  }

  const double nominalOffsetsTIB[8] = { 0.00035, 2.10687, -2.10827, -0.00173446, 2.10072, -0.00135114, 2.10105, -2.10401 };

  // in tob, modules have alternating orientations along the rods.
  // this is described by the following pattern.
  // (even more confusing, this pattern is inversed for beams 0, 5, 6, 7)
  const int orientationPattern[6] = { -1, 1, 1, -1, -1, 1 };
  const double nominalOffsetsTOB[8] = { 0.00217408, 1.58678, 117.733, 119.321, 120.906, 119.328, 117.743, 1.58947 };


  if( det == 2 ) return( -1. * nominalOffsetsTIB[beam] );

  else {
    if( beam == 0 or beam > 4 ) return( nominalOffsetsTOB[beam] * orientationPattern[pos] );
    else return( -1. * nominalOffsetsTOB[beam] * orientationPattern[pos] );
  }

}




///
/// not all TEC-AT modules are hit in the center;
/// this func returns the nominal beam offset locally on a module (in strips)
/// for the ProfileJudge and the LASPeakFinder in strips.
/// (offset = middle of module - nominal position)
///
/// the hard coded numbers will later be supplied by a special geometry class..
/// 
double LaserAlignment::getTEC2TECNominalBeamOffset( unsigned int det, unsigned int beam, unsigned int disk ) {

  if( det > 1 || beam > 7 || disk > 5 ) {
    throw cms::Exception( "[LaserAlignment::getTEC2TECNominalBeamOffset]" ) << " ERROR ** Called with nonexisting parameter set: det " << det << " beam " << beam << " disk " << disk << "." << std::endl;
  }

  const double nominalOffsets[8] = { 0., 2.220, -2.221, 0., 2.214, 0., 2.214, -2.217 };
  
  if( det == 0 ) return -1. * nominalOffsets[beam];
  else return nominalOffsets[beam];

}





///
///
///
void LaserAlignment::CalculateNominalCoordinates( void ) {

  //
  // hard coded data yet...
  //

  // nominal phi values of tec beam / alignment tube hits (parameter is beam 0-7)
  const double tecPhiPositions[8]   = { 0.392699, 1.178097, 1.963495, 2.748894, 3.534292, 4.319690, 5.105088, 5.890486 }; // new values calculated by maple
  const double atPhiPositions[8]    = { 0.392699, 1.289799, 1.851794, 2.748894, 3.645995, 4.319690, 5.216791, 5.778784 }; // new values calculated by maple

  // nominal r values (mm) of hits
  const double tobRPosition = 600.;
  const double tibRPosition = 514.;
  const double tecRPosition[2] = { 564., 840. }; // ring 4,6

  // nominal z values (mm) of hits in barrel (parameter is pos 0-6)
  const double tobZPosition[6] = { 1040., 580., 220., -140., -500., -860. };
  const double tibZPosition[6] = { 620., 380., 180., -100., -340., -540. };

  // nominal z values (mm) of hits in tec (parameter is disk 0-8); FOR TEC-: (* -1.)
  const double tecZPosition[9] = { 1322.5, 1462.5, 1602.5, 1742.5, 1882.5, 2057.5, 2247.5, 2452.5, 2667.5 };
  

  //
  // now we fill these into the nominalCoordinates container;
  // errors are zero for nominal values..
  //

  // loop object and its variables
  LASGlobalLoop moduleLoop;
  int det, ring, beam, disk, pos;

  
  // TEC+- section
  det = 0; ring = 0, beam = 0; disk = 0;
  do {
    
    if( det == 0 ) { // this is TEC+
      nominalCoordinates.SetTECEntry( det, ring, beam, disk, LASCoordinateSet( tecPhiPositions[beam], 0., tecRPosition[ring], 0., tecZPosition[disk], 0. ) );
    }
    else { // now TEC-
      nominalCoordinates.SetTECEntry( det, ring, beam, disk, LASCoordinateSet( tecPhiPositions[beam], 0., tecRPosition[ring], 0., -1. * tecZPosition[disk], 0. ) ); // just * -1.
    }
    
  } while( moduleLoop.TECLoop( det, ring, beam, disk ) );



  // TIB & TOB section
  det = 2; beam = 0; pos = 0;
  do {
    if( det == 2 ) { // this is TIB
      nominalCoordinates.SetTIBTOBEntry( det, beam, pos, LASCoordinateSet( atPhiPositions[beam], 0., tibRPosition, 0., tibZPosition[pos], 0. ) );
    }
    else { // now TOB
      nominalCoordinates.SetTIBTOBEntry( det, beam, pos, LASCoordinateSet( atPhiPositions[beam], 0., tobRPosition, 0., tobZPosition[pos], 0. ) );
    }

  } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );




  // TEC2TEC AT section
  det = 0; beam = 0; disk = 0;
  do {
    
    if( det == 0 ) { // this is TEC+, ring4 only
      nominalCoordinates.SetTEC2TECEntry( det, beam, disk, LASCoordinateSet( atPhiPositions[beam], 0., tecRPosition[0], 0., tecZPosition[disk], 0. ) );
    }
    else { // now TEC-
      nominalCoordinates.SetTEC2TECEntry( det, beam, disk, LASCoordinateSet( atPhiPositions[beam], 0., tecRPosition[0], 0., -1. * tecZPosition[disk], 0. ) ); // just * -1.
    }
    
  } while( moduleLoop.TEC2TECLoop( det, beam, disk ) );


}





///
/// convert an angle in the [-pi,pi] range
/// to the [0,2*pi] range
///
double LaserAlignment::ConvertAngle( double angle ) {

  if( angle < -1. * M_PI  || angle > M_PI ) {
    throw cms::Exception(" [LaserAlignment::ConvertAngle] ") << "** ERROR: Called with illegal input angle: " << angle << "." << std::endl;
  }

  if( angle >= 0. ) return angle;
  else return( angle + 2. * M_PI );

}





///
/// debug only, will disappear
///
void LaserAlignment::DumpPosFileSet( LASGlobalData<LASCoordinateSet>& coordinates ) {

  LASGlobalLoop loop;
  int det, ring, beam, disk, pos;

  std:: cout << std::endl << " [LaserAlignment::DumpPosFileSet] -- Dump: " << std::endl;

  // TEC INTERNAL
  det = 0; ring = 0; beam = 0; disk = 0;
  do {
    std::cout << "POS " << det << "\t" << beam << "\t" << disk << "\t" << ring << "\t" << coordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() << "\t" << coordinates.GetTECEntry( det, ring, beam, disk ).GetPhiError() << std::endl;
  } while ( loop.TECLoop( det, ring, beam, disk ) );

  // TIBTOB
  det = 2; beam = 0; pos = 0;
  do {
    std::cout << "POS " << det << "\t" << beam << "\t" << pos << "\t" << "-1" << "\t" << coordinates.GetTIBTOBEntry( det, beam, pos ).GetPhi() << "\t" << coordinates.GetTIBTOBEntry( det, beam, pos ).GetPhiError() << std::endl;
  } while( loop.TIBTOBLoop( det, beam, pos ) );

  // TEC2TEC
  det = 0; beam = 0; disk = 0;
  do {
    std::cout << "POS " << det << "\t" << beam << "\t" << disk << "\t" << "-1" << "\t" << coordinates.GetTEC2TECEntry( det, beam, disk ).GetPhi() << "\t" << coordinates.GetTEC2TECEntry( det, beam, disk ).GetPhiError() << std::endl;
  } while( loop.TEC2TECLoop( det, beam, disk ) );

  std:: cout << std::endl << " [LaserAlignment::DumpPosFileSet] -- End dump: " << std::endl;

}





///
///
///
void LaserAlignment::DumpStripFileSet( LASGlobalData<std::pair<float,float> >& measuredStripPositions ) {

  LASGlobalLoop loop;
  int det, ring, beam, disk, pos;

  std:: cout << std::endl << " [LaserAlignment::DumpStripFileSet] -- Dump: " << std::endl;

  // TEC INTERNAL
  det = 0; ring = 0; beam = 0; disk = 0;
  do {
    std::cout << "STRIP " << det << "\t" << beam << "\t" << disk << "\t" << ring << "\t" << measuredStripPositions.GetTECEntry( det, ring, beam, disk ).first
	      << "\t" << measuredStripPositions.GetTECEntry( det, ring, beam, disk ).second << std::endl;
  } while ( loop.TECLoop( det, ring, beam, disk ) );

  // TIBTOB
  det = 2; beam = 0; pos = 0;
  do {
    std::cout << "STRIP " << det << "\t" << beam << "\t" << pos << "\t" << "-1" << "\t" << measuredStripPositions.GetTIBTOBEntry( det, beam, pos ).first
	      << "\t" << measuredStripPositions.GetTIBTOBEntry( det, beam, pos ).second << std::endl;
  } while( loop.TIBTOBLoop( det, beam, pos ) );

  // TEC2TEC
  det = 0; beam = 0; disk = 0;
  do {
    std::cout << "STRIP " << det << "\t" << beam << "\t" << disk << "\t" << "-1" << "\t" << measuredStripPositions.GetTEC2TECEntry( det, beam, disk ).first
	      << "\t" << measuredStripPositions.GetTEC2TECEntry( det, beam, disk ).second << std::endl;
  } while( loop.TEC2TECLoop( det, beam, disk ) );

  std:: cout << std::endl << " [LaserAlignment::DumpStripFileSet] -- End dump: " << std::endl;
  
  
}





///
///
///
void LaserAlignment::DumpHitmaps( LASGlobalData<int> &numberOfAcceptedProfiles ) {

  std::cout << " [LaserAlignment::DumpHitmaps] -- Dumping hitmap for TEC+:" << std::endl;
  std::cout << " [LaserAlignment::DumpHitmaps] -- Ring4:" << std::endl;
  std::cout << "     disk0   disk1   disk2   disk3   disk4   disk5   disk6   disk7   disk8" << std::endl;

  for( int beam = 0; beam < 8; ++beam ) {
    std::cout << " beam" << beam << ":";
    for( int disk = 0; disk < 9; ++disk ) {
      std::cout << "\t" << numberOfAcceptedProfiles.GetTECEntry( 0, 0, beam, disk );
    }
    std::cout << std::endl;
  }

  std::cout << " [LaserAlignment::DumpHitmaps] -- Ring6:" << std::endl;
  std::cout << "     disk0   disk1   disk2   disk3   disk4   disk5   disk6   disk7   disk8" << std::endl;

  for( int beam = 0; beam < 8; ++beam ) {
    std::cout << " beam" << beam << ":";
    for( int disk = 0; disk < 9; ++disk ) {
      std::cout << "\t" << numberOfAcceptedProfiles.GetTECEntry( 0, 1, beam, disk );
    }
    std::cout << std::endl;
  }

  std::cout << " [LaserAlignment::DumpHitmaps] -- Dumping hitmap for TEC-:" << std::endl;
  std::cout << " [LaserAlignment::DumpHitmaps] -- Ring4:" << std::endl;
  std::cout << "     disk0   disk1   disk2   disk3   disk4   disk5   disk6   disk7   disk8" << std::endl;

  for( int beam = 0; beam < 8; ++beam ) {
    std::cout << " beam" << beam << ":";
    for( int disk = 0; disk < 9; ++disk ) {
      std::cout << "\t" << numberOfAcceptedProfiles.GetTECEntry( 1, 0, beam, disk );
    }
    std::cout << std::endl;
  }

  std::cout << " [LaserAlignment::DumpHitmaps] -- Ring6:" << std::endl;
  std::cout << "     disk0   disk1   disk2   disk3   disk4   disk5   disk6   disk7   disk8" << std::endl;

  for( int beam = 0; beam < 8; ++beam ) {
    std::cout << " beam" << beam << ":";
    for( int disk = 0; disk < 9; ++disk ) {
      std::cout << "\t" << numberOfAcceptedProfiles.GetTECEntry( 1, 1, beam, disk );
    }
    std::cout << std::endl;
  }

  std::cout << " [LaserAlignment::DumpHitmaps] -- End of dump." << std::endl << std::endl;

}





///
/// loop the list of endcap modules to be masked and
/// apply the corrections from the "endcapParameters" to them
///
void LaserAlignment::ApplyEndcapMaskingCorrections( LASGlobalData<LASCoordinateSet>& measuredCoordinates, LASGlobalData<LASCoordinateSet>& nominalCoordinates, LASEndcapAlignmentParameterSet& endcapParameters ) {

  // loop the list of modules to be masked
  for( std::vector<unsigned int>::iterator moduleIt = theMaskTecModules.begin(); moduleIt != theMaskTecModules.end(); ++moduleIt ) {

    // loop variables
    LASGlobalLoop moduleLoop;
    int det, ring, beam, disk;

    // this will calculate the corrections from the alignment parameters
    LASEndcapAlgorithm endcapAlgorithm;

    // find the location of the respective module in the container with this loop
    det = 0; ring = 0; beam = 0; disk = 0;
    do {
	  
      // here we got it
      if( detectorId.GetTECEntry( det, ring, beam, disk ) == *moduleIt ) {
	
	// the nominal phi value for this module
	const double nominalPhi = nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi();
	
	// the offset from the alignment parameters
	const double phiCorrection = endcapAlgorithm.GetAlignmentParameterCorrection( det, ring, beam, disk, nominalCoordinates, endcapParameters );
	
	// apply the corrections
	measuredCoordinates.GetTECEntry( det, ring, beam, disk ).SetPhi( nominalPhi - phiCorrection );
	
      }
      
    } while ( moduleLoop.TECLoop( det, ring, beam, disk ) );
    
  }

}





///
/// loop the list of alignment tube modules to be masked and
/// apply the corrections from the "barrelParameters" to them
///
void LaserAlignment::ApplyATMaskingCorrections( LASGlobalData<LASCoordinateSet>& measuredCoordinates, LASGlobalData<LASCoordinateSet>& nominalCoordinates, LASBarrelAlignmentParameterSet& atParameters ) {

  // loop the list of modules to be masked
  for( std::vector<unsigned int>::iterator moduleIt = theMaskAtModules.begin(); moduleIt != theMaskAtModules.end(); ++moduleIt ) {

    // loop variables
    LASGlobalLoop moduleLoop;
    int det, beam, disk, pos;

    // this will calculate the corrections from the alignment parameters
    LASAlignmentTubeAlgorithm atAlgorithm;


    // find the location of the respective module in the container with these loops:

    // first TIB+TOB
    det = 2; beam = 0; pos = 0;
    do {

      // here we got it
      if( detectorId.GetTIBTOBEntry( det, beam, pos ) == *moduleIt ) {

	// the nominal phi value for this module
	const double nominalPhi = nominalCoordinates.GetTIBTOBEntry( det, beam, pos ).GetPhi();

	// the offset from the alignment parameters
	const double phiCorrection = atAlgorithm.GetTIBTOBAlignmentParameterCorrection( det, beam, pos, nominalCoordinates, atParameters );

	// apply the corrections
	measuredCoordinates.GetTIBTOBEntry( det, beam, pos ).SetPhi( nominalPhi - phiCorrection );

      }

    } while ( moduleLoop.TIBTOBLoop( det, beam, pos ) );
      
    
    
    // then TEC(AT)  
    det = 0; beam = 0; disk = 0;
    do {
	  
      // here we got it
      if( detectorId.GetTEC2TECEntry( det, beam, disk ) == *moduleIt ) {

	// the nominal phi value for this module
	const double nominalPhi = nominalCoordinates.GetTEC2TECEntry( det, beam, disk ).GetPhi();

	// the offset from the alignment parameters
	const double phiCorrection = atAlgorithm.GetTEC2TECAlignmentParameterCorrection( det, beam, disk, nominalCoordinates, atParameters );

	// apply the corrections
	measuredCoordinates.GetTEC2TECEntry( det, beam, disk ).SetPhi( nominalPhi - phiCorrection );

      }
      
    } while ( moduleLoop.TEC2TECLoop( det, beam, disk ) );
    
  }

}





///
/// this function is for debugging and testing only
/// and will disappear..
///
void LaserAlignment::testRoutine( void ) {


  // tracker geom. object for calculating the global beam positions
  const TrackerGeometry& theTracker( *theTrackerGeometry );

  const double atPhiPositions[8] = { 0.392699, 1.289799, 1.851794, 2.748894, 3.645995, 4.319690, 5.216791, 5.778784 };
  const double tecPhiPositions[8] = { 0.392699, 1.178097, 1.963495, 2.748894, 3.534292, 4.319690, 5.105088, 5.890486 };
  const double zPositions[9] = { 125.0, 139.0, 153.0, 167.0, 181.0, 198.5, 217.5, 238.0, 259.5 };
  const double zPositionsTIB[6] = { 62.0, 38.0, 18.0, -10.0, -34.0, -54.0 };
  const double zPositionsTOB[6] = { 104.0, 58.0, 22.0, -14.0, -50.0, -86.0 };

  int det, beam, disk, pos, ring;
  
  // loop TEC+- internal
  det = 0; ring = 0; beam = 0; disk = 0;
  do {

    const double radius = ring?84.0:56.4;

    // access the tracker geometry for this module
    const DetId theDetId( detectorId.GetTECEntry( det, ring, beam, disk ) );
    const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTracker.idToDet( theDetId ) );
    
    if (theStripDet) {
      const GlobalPoint gp( GlobalPoint::Cylindrical( radius, tecPhiPositions[beam], zPositions[disk] ) );
      
      const LocalPoint lp( theStripDet->surface().toLocal( gp ) );
      std::cout << "__TEC: " << 256. - theStripDet->specificTopology().strip( lp ) << std::endl; /////////////////////////////////
    }

  } while( moduleLoop.TECLoop( det, ring, beam, disk ) );


  // loop TIBTOB
  det = 2; beam = 0; pos = 0;
  do {

    const double radius = (det==2?51.4:58.4); /////////////////////////////////////////////////////////////////////////////
    const double theZ = (det==2?zPositionsTIB[pos]:zPositionsTOB[pos]);

    // access the tracker geometry for this module
    const DetId theDetId( detectorId.GetTIBTOBEntry( det, beam, pos ) );
    const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTracker.idToDet( theDetId ) );
    
    if (theStripDet) {
      const GlobalPoint gp( GlobalPoint::Cylindrical( radius, atPhiPositions[beam], theZ ) );
      
      const LocalPoint lp( theStripDet->surface().toLocal( gp ) );
      std::cout << "__TIBTOB det " << det << " beam " << beam << " pos " << pos << "  " << 256. - theStripDet->specificTopology().strip( lp );
      std::cout << "           " << theStripDet->position().perp()<< std::endl; /////////////////////////////////
    }

  } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );

  
  // loop TEC2TEC
  det = 0; beam = 0; disk = 0;
  do {

    const double radius = 56.4;

    // access the tracker geometry for this module
    const DetId theDetId( detectorId.GetTEC2TECEntry( det, beam, disk ) );
    const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTracker.idToDet( theDetId ) );
    
    if (theStripDet) {
      const GlobalPoint gp( GlobalPoint::Cylindrical( radius, atPhiPositions[beam], zPositions[disk] ) );
      
      const LocalPoint lp( theStripDet->surface().toLocal( gp ) );
      std::cout << "__TEC2TEC det " << det << " beam " << beam << " disk " << disk << "  " << 256. - theStripDet->specificTopology().strip( lp ) << std::endl; /////////////////////////////////
    }

  } while( moduleLoop.TEC2TECLoop( det, beam, disk ) );


}









// define the SEAL module
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(LaserAlignment);




// the ATTIC

