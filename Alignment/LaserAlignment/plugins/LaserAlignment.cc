/** \file LaserAlignment.cc
 *  LAS reconstruction module
 *
 *  $Date: 2009/01/14 16:47:54 $
 *  $Revision: 1.32 $
 *  \author Maarten Thomas
 *  \author Jan Olzem
 */

#include "Alignment/LaserAlignment/plugins/LaserAlignment.h"
#include "FWCore/Framework/interface/Event.h" 
#include "TFile.h" 

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/LaserAlignment/interface/LASBeamProfileFit.h"
#include "DataFormats/LaserAlignment/interface/LASBeamProfileFitCollection.h"
#include "DataFormats/LaserAlignment/interface/LASAlignmentParameter.h"
#include "DataFormats/LaserAlignment/interface/LASAlignmentParameterCollection.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"


LaserAlignment::LaserAlignment(edm::ParameterSet const& theConf) 
  : theEvents(0), 
    theDoPedestalSubtraction(theConf.getUntrackedParameter<bool>("SubtractPedestals", true)),
    enableJudgeZeroFilter(theConf.getUntrackedParameter<bool>("EnableJudgeZeroFilter", true)),
    updateFromIdealGeometry(theConf.getUntrackedParameter<bool>("UpdateFromIdealGeometry", false)),
    theStoreToDB(theConf.getUntrackedParameter<bool>("saveToDbase", false)),
    theSaveHistograms(theConf.getUntrackedParameter<bool>("saveHistograms",false)),
    theDebugLevel(theConf.getUntrackedParameter<int>("DebugLevel",0)),
    theNEventsPerLaserIntensity(theConf.getUntrackedParameter<int>("NumberOfEventsPerLaserIntensity",100)),
    theNEventsForAllIntensities(theConf.getUntrackedParameter<int>("NumberOfEventsForAllIntensities",100)),
    theDoAlignmentAfterNEvents(theConf.getUntrackedParameter<int>("DoAlignmentAfterNEvents",1000)),
    /// the following three are hard-coded until the complete package has been refurbished
    //    theAlignPosTEC( false ), // theAlignPosTEC(theConf.getUntrackedParameter<bool>("AlignPosTEC",false)),
    //    theAlignNegTEC( false ), // theAlignNegTEC(theConf.getUntrackedParameter<bool>("AlignNegTEC",false)), 
    //    theAlignTEC2TEC( false ), // theAlignTEC2TEC(theConf.getUntrackedParameter<bool>("AlignTECTIBTOBTEC",false)),
    theIsGoodFit(false),
    theSearchPhiTIB(theConf.getUntrackedParameter<double>("SearchWindowPhiTIB",0.05)),
    theSearchPhiTOB(theConf.getUntrackedParameter<double>("SearchWindowPhiTOB",0.05)),
    theSearchPhiTEC(theConf.getUntrackedParameter<double>("SearchWindowPhiTEC",0.05)),
    theSearchZTIB(theConf.getUntrackedParameter<double>("SearchWindowZTIB",1.0)),
    theSearchZTOB(theConf.getUntrackedParameter<double>("SearchWindowZTOB",1.0)),
    thePhiErrorScalingFactor(theConf.getUntrackedParameter<double>("PhiErrorScalingFactor",1.0)),
    theDigiProducersList(theConf.getParameter<Parameters>("DigiProducersList")),
    theFile(),
    theCompression(theConf.getUntrackedParameter<int>("ROOTFileCompression",1)),
    theFileName(theConf.getUntrackedParameter<std::string>("ROOTFileName","test.root")),
    theBeamFitPS(theConf.getParameter<edm::ParameterSet>("BeamProfileFitter")),
    theAlignmentAlgorithmPS(theConf.getParameter<edm::ParameterSet>("AlignmentAlgorithm")),
    theMinAdcCounts(theConf.getUntrackedParameter<int>("MinAdcCounts",0)),
    theHistogramNames(), theHistograms(),
    theLaserPhi(),
    theLaserPhiError(),
    theNumberOfIterations(0), theNumberOfAlignmentIterations(0),
    theBeamFitter(),
			      //    theLASAlignPosTEC(),
			      //    theLASAlignNegTEC(),
			      //    theLASAlignTEC2TEC(),
			      //    theAlignmentAlgorithmBW(),
    theUseBSFrame(theConf.getUntrackedParameter<bool>("UseBeamSplitterFrame", true)),
    theDigiStore(),
    theBeamProfileFitStore(),
    theDigiVector(),
    theAlignableTracker(),
    theAlignRecordName( "TrackerAlignmentRcd" ),
    theErrorRecordName( "TrackerAlignmentErrorRcd" )

{
  // load the configuration from the ParameterSet  
  edm::LogInfo("LaserAlignment") <<    "==========================================================="
				  << "\n===                Start configuration                  ==="
				  << "\n    theDebugLevel               = " << theDebugLevel
    //				  << "\n    theAlignPosTEC              = " << theAlignPosTEC
    //				  << "\n    theAlignNegTEC              = " << theAlignNegTEC
    //				  << "\n    theAlignTEC2TEC             = " << theAlignTEC2TEC
				  << "\n    theSearchPhiTIB             = " << theSearchPhiTIB
				  << "\n    theSearchPhiTOB             = " << theSearchPhiTOB
				  << "\n    theSearchPhiTEC             = " << theSearchPhiTEC 
				  << "\n    theSearchZTIB               = " << theSearchZTIB
				  << "\n    theSearchZTOB               = " << theSearchZTOB
				  << "\n    theMinAdcCounts             = " << theMinAdcCounts
				  << "\n    theNEventsPerLaserIntensity = " << theNEventsPerLaserIntensity
				  << "\n    theNEventsForAllIntensiteis = " << theNEventsForAllIntensities
				  << "\n    theDoAlignmentAfterNEvents  = " << theDoAlignmentAfterNEvents
				  << "\n    ROOT filename               = " << theFileName
				  << "\n    compression                 = " << theCompression
				  << "\n===========================================================";

  // alias for the Branches in the root files
  std::string alias ( theConf.getParameter<std::string>("@module_label") );  

  // declare the product to produce
  produces<TkLasBeamCollection, edm::InRun>( "tkLaserBeams" ).setBranchAlias( alias + "TkLasBeamCollection" );

  // the alignable tracker parts
  //  theLASAlignTEC2TEC = new LaserAlignmentTEC2TEC;
  
  // the alignment algorithm from Bruno
  //  theAlignmentAlgorithmBW = new AlignmentAlgorithmBW;
  
  // counter for the number of iterations, i.e. the number of BeamProfile fits and
  // local Millepede fits
  theNumberOfIterations = 0;

  // switch judge's zero filter depending on cfg
  judge.EnableZeroFilter( enableJudgeZeroFilter );

}





LaserAlignment::~LaserAlignment() {

  if (theSaveHistograms) {
    closeRootFile();
  }
  
  if (theFile != 0) { delete theFile; }
  
  if (theBeamFitter != 0) { delete theBeamFitter; }
  
  //  if (theLASAlignTEC2TEC != 0) { delete theLASAlignTEC2TEC; }
  if (theAlignableTracker != 0) { delete theAlignableTracker; }
  //  if (theAlignmentAlgorithmBW != 0) { delete theAlignmentAlgorithmBW; }
}





///
///
///
double LaserAlignment::angle(double theAngle) {
  return (theAngle >= 0.0) ? theAngle : theAngle + 2.0*M_PI;
}





///
///
///
void LaserAlignment::beginJob(const edm::EventSetup& theSetup) {

  // the beam profile fitter
  theBeamFitter = new BeamProfileFitter( theBeamFitPS, &theSetup );

  // creating a new file
  theFile = new TFile(theFileName.c_str(),"RECREATE","CMS ROOT file");
  theFile->SetCompressionLevel(theCompression);
      
  // initialize the histograms
  if (theFile) {
    this->initHistograms();
  }
  else {
    throw cms::Exception("LaserAlignment") << "<LaserAlignment::beginJob()>: ERROR!!! something wrong with the RootFile" << std::endl;
  } 


  LogDebug("LaserAlignment:beginJob()") << " access the Tracker Geometry ";

  // detector id maps (hard coded)
  fillDetectorId();

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

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //   PROFILE & HISTOGRAM INITIALIZATION
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    // to be still compatible with Maarten's code
    nameBuilder.clear();
    nameBuilder.str( "" );
    nameBuilder << "Beam" << beam << "Ring";
    if( ring == 0 ) nameBuilder << "4"; else nameBuilder << "6";
    nameBuilder << "Disc" << disk + 1; // +1 is a convention in maarten's code
    if( det == 0 ) nameBuilder << "Pos"; else nameBuilder << "Neg";
    nameBuilder << "TEC"; 
    theProfileNames.SetTECEntry( det, ring, beam, disk, nameBuilder.str() );

    // init the histograms
    nameBuilder << "Histo";
    summedHistograms.SetTECEntry( det, ring, beam, disk, new TH1D( nameBuilder.str().c_str(), nameBuilder.str().c_str(), 512, 0, 512 ) );
    summedHistograms.GetTECEntry( det, ring, beam, disk )->SetDirectory( singleModulesDir );
    
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
    nameBuilder << "Beam" << beam;
    if( det == 2 ) nameBuilder << "TIB"; else nameBuilder << "TOB";
    nameBuilder << "Position" << pos + 1; // +1 is a convention in maarten's code
    theProfileNames.SetTIBTOBEntry( det, beam, pos, nameBuilder.str() );

    // init the histograms
    nameBuilder << "Histo";
    summedHistograms.SetTIBTOBEntry( det, beam, pos, new TH1D( nameBuilder.str().c_str(), nameBuilder.str().c_str(), 512, 0, 512 ) );
    summedHistograms.GetTIBTOBEntry( det, beam, pos )->SetDirectory( singleModulesDir );
    
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
    nameBuilder << "Beam" << beam << "Ring4" << "Disc" << disk + 1;
    if( det == 0 ) nameBuilder << "Pos"; else nameBuilder << "Neg";
    nameBuilder << "TEC2TEC";
    theProfileNames.SetTEC2TECEntry( det, beam, disk, nameBuilder.str() );

    // init the histograms
    nameBuilder << "Histo";
    summedHistograms.SetTEC2TECEntry( det, beam, disk, new TH1D( nameBuilder.str().c_str(), nameBuilder.str().c_str(), 512, 0, 512 ) );
    summedHistograms.GetTEC2TECEntry( det, beam, disk )->SetDirectory( singleModulesDir );
    
  } while( moduleLoop.TEC2TECLoop( det, beam, disk ) );



  // Create the alignable hierarchy
  LogDebug("LaserAlignment:beginJob()") << " create the alignable hierarchy ";
  if( updateFromIdealGeometry ) {
    // the AlignableTracker object is initialized with the ideal geometry
    edm::ESHandle<GeometricDet> theGeometricDet;
    theSetup.get<IdealGeometryRecord>().get(theGeometricDet);
    TrackerGeomBuilderFromGeometricDet trackerBuilder;
    TrackerGeometry* theRefTracker = trackerBuilder.build(&*theGeometricDet);
    theAlignableTracker = new AlignableTracker(&(*theRefTracker));
  }
  else {
    // the AlignableTracker object is initialized with the input geometry from DB
    theAlignableTracker = new AlignableTracker( &(*theTrackerGeometry) );
  }

}






///
///
///
void LaserAlignment::produce(edm::Event& theEvent, edm::EventSetup const& theSetup)  {


  LogDebug("LaserAlignment") << "==========================================================="
			      << "\n   Private analysis of event #"<< theEvent.id().event() 
			      << " in run #" << theEvent.id().run();


  // do the Tracker Statistics to retrieve the current profiles
  //  trackerStatistics( theEvent, theSetup );
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
    if( judge.IsSignalIn( currentDataProfiles.GetTIBTOBEntry( det, beam, pos ) - pedestalProfiles.GetTIBTOBEntry( det, beam, pos ), getTIBTOBProfileOffset( det, beam, pos ) ) ) {
      isAcceptedProfile.SetTIBTOBEntry( det, beam, pos, 1 );
    }
    else { // dto.
      isAcceptedProfile.SetTIBTOBEntry( det, beam, pos, 0 );
    }

  } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );




  // now come the beam finders
  bool isTECMode = isTECBeam();
  LogDebug( " [LaserAlignment::produce]" ) << "LaserAlignment::isTECBeam declares this event " << ( isTECMode ? "" : "NOT " ) << "a TEC event." << std::endl;
  std::cout << " [LaserAlignment::produce] -- LaserAlignment::isTECBeam declares this event " << ( isTECMode ? "" : "NOT " ) << "a TEC event." << std::endl;

  bool isATMode  = isATBeam();
  LogDebug( " [LaserAlignment::produce]" ) << "LaserAlignment::isATBeam declares this event "  << ( isATMode ? "" : "NOT " )  << "an AT event." << std::endl;
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
    if( judge.JudgeProfile( currentDataProfiles.GetTIBTOBEntry( det, beam, pos ) - pedestalProfiles.GetTIBTOBEntry( det, beam, pos ), getTIBTOBProfileOffset( det, beam, pos ) ) ) {
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


  // ----- check if the actual event can be used -----
  /* here we can later on add some criteria for good alignment events!? */
  theEvents++;

}





///
///
///
void LaserAlignment::closeRootFile() {
  theFile->Write();
}





///
///
///
void LaserAlignment::endRun( edm::Run& theRun, const edm::EventSetup& theSetup ) {

  LogDebug("LaserAlignment") << "     Total Event number = " << theEvents;

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
  std::pair<double,double> peakFinderResults;
  bool isGoodFit;
  
  // tracker geom. object for calculating the global beam positions
  const TrackerGeometry& theTracker( *theTrackerGeometry );
  
  // fill LASGlobalData<LASCoordinateSet> nominalCoordinates
  CalculateNominalCoordinates();
  
  // for determining the phi errors
  //    ErrorFrameTransformer errorTransformer;
  
  // do the fits for TEC+- internal
  det = 0; ring = 0; beam = 0; disk = 0;
  do {
    
    // do the fit
    isGoodFit = peakFinder.FindPeakIn( collectedDataProfiles.GetTECEntry( det, ring, beam, disk ), peakFinderResults, 0 ); // offset is 0 for TEC
    // now we have the measured positions in units of strips. 
    if( !isGoodFit ) std::cout << " [LaserAlignment::endRun] ** WARNING: Fit failed for TEC det: "
			       << det << ", ring: " << ring << ", beam: " << beam << ", disk: " << disk << "." << std::endl;
    

    // <- here we will later implement the kink corrections
      
    // access the tracker geometry for this module
    const DetId theDetId( detectorId.GetTECEntry( det, ring, beam, disk ) );
    const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTracker.idToDet( theDetId ) );
      
    // first, set the measured coordinates to their nominal values
    measuredCoordinates.SetTECEntry( det, ring, beam, disk, nominalCoordinates.GetTECEntry( det, ring, beam, disk ) );

    if( isGoodFit ) { // convert strip position to global phi and replace the nominal phi value/error

      measuredStripPositions.GetTECEntry( det, ring, beam, disk ) = peakFinderResults;

      const GlobalPoint& globalPoint = theStripDet->surface().toGlobal( theStripDet->specificTopology().localPosition( peakFinderResults.first ) );
      measuredCoordinates.GetTECEntry( det, ring, beam, disk ).SetPhi( ConvertAngle( globalPoint.barePhi() ) );
      //      measuredCoordinates.GetTECEntry( det, ring, beam, disk ).SetPhi( nominalCoordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() ); // ############## CHEAT 0

      //      const GlobalError& globalError = errorTransformer.transform( theStripDet->specificTopology().localError( peakFinderResults.first, pow( peakFinderResults.second, 2 ) ), theStripDet->surface() );
      //      measuredCoordinates.GetTECEntry( det, ring, beam, disk ).SetPhiError( globalError.phierr( globalPoint ) );
      measuredCoordinates.GetTECEntry( det, ring, beam, disk ).SetPhiError( 0.00046  ); // PRELIMINARY ESTIMATE

    }
    else { // keep nominal position but set a giant phi error so that the module can be ignored by the alignment algorithm
      measuredStripPositions.GetTECEntry( det, ring, beam, disk ) = std::pair<float,float>( 256, 1000. );
      const GlobalPoint& globalPoint = theStripDet->surface().toGlobal( theStripDet->specificTopology().localPosition( 256 ) );
      measuredCoordinates.GetTECEntry( det, ring, beam, disk ).SetPhi( ConvertAngle( globalPoint.barePhi() ) );
      measuredCoordinates.GetTECEntry( det, ring, beam, disk ).SetPhiError( 1000. );
    }
      
    // fill the histograms for saving
    for( int bin = 1; bin <= 512; ++bin ) {
      summedHistograms.GetTECEntry( det, ring, beam, disk )->SetBinContent( bin, collectedDataProfiles.GetTECEntry( det, ring, beam, disk ).GetValue( bin - 1 ) );
    }

  } while( moduleLoop.TECLoop( det, ring, beam, disk ) );




  // do the fits for TIB/TOB
  det = 2; beam = 0; pos = 0;
  do {

    // do the fit
    isGoodFit = peakFinder.FindPeakIn( collectedDataProfiles.GetTIBTOBEntry( det, beam, pos ), peakFinderResults, getTIBTOBProfileOffset( det, beam, pos ) );
    // now we have the measured positions in units of strips.
    if( !isGoodFit ) std::cout << " [LaserAlignment::endJob] ** WARNING: Fit failed for TIB/TOB det: "
			       << det << ", beam: " << beam << ", pos: " << pos << "." << std::endl;
      
    // <- here we will later implement the kink corrections
      
    // access the tracker geometry for this module
    const DetId theDetId( detectorId.GetTIBTOBEntry( det, beam, pos ) );
    const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTracker.idToDet( theDetId ) );
      
    // first, set the measured coordinates to their nominal values
    measuredCoordinates.SetTIBTOBEntry( det, beam, pos, nominalCoordinates.GetTIBTOBEntry( det, beam, pos ) );
      
    if( isGoodFit ) { // convert strip position to global phi and replace the nominal phi value/error

      measuredStripPositions.GetTIBTOBEntry( det, beam, pos ) = peakFinderResults;
      const GlobalPoint& globalPoint = theStripDet->surface().toGlobal( theStripDet->specificTopology().localPosition( peakFinderResults.first ) );
      measuredCoordinates.GetTIBTOBEntry( det, beam, pos ).SetPhi( ConvertAngle( globalPoint.barePhi() ) );
      //      measuredCoordinates.GetTIBTOBEntry( det, beam, pos ).SetPhi( nominalCoordinates.GetTIBTOBEntry( det, beam, pos ).GetPhi() ); // ############################## CHEAT 0
      measuredCoordinates.GetTIBTOBEntry( det, beam, pos ).SetPhiError( 0.00028 ); // PRELIMINARY ESTIMATE
    }
    else { // keep nominal position but set a giant phi error so that the module can be ignored by the alignment algorithm
      measuredStripPositions.GetTIBTOBEntry( det, beam, pos ) = std::pair<float,float>( 256 + getTIBTOBProfileOffset( det, beam, pos ), 1000. );
      measuredCoordinates.GetTIBTOBEntry( det, beam, pos ).SetPhiError( 1000. );
    }
      
    // fill the histograms for saving
    for( int bin = 1; bin <= 512; ++bin ) {
      summedHistograms.GetTIBTOBEntry( det, beam, pos )->SetBinContent( bin, collectedDataProfiles.GetTIBTOBEntry( det, beam, pos ).GetValue( bin - 1 ) );
    }
	
  } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );




  // do the fits for TEC AT
  det = 0; beam = 0; disk = 0;
  do {

    // do the fit
    isGoodFit = peakFinder.FindPeakIn( collectedDataProfiles.GetTEC2TECEntry( det, beam, disk ), peakFinderResults, 0 ); // no offset for TEC modules
    // now we have the positions in units of strips.
    if( !isGoodFit ) std::cout << " [LaserAlignment::endRun] ** WARNING: Fit failed for TEC2TEC det: "
			       << det << ", beam: " << beam << ", disk: " << disk << "." << std::endl;

    // <- here we will later implement the kink corrections
    
    // access the tracker geometry for this module
    const DetId theDetId( detectorId.GetTEC2TECEntry( det, beam, disk ) );
    const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTracker.idToDet( theDetId ) );

    // first, set the measured coordinates to their nominal values
    measuredCoordinates.SetTEC2TECEntry( det, beam, disk, nominalCoordinates.GetTEC2TECEntry( det, beam, disk ) );
    
    if( isGoodFit ) { // convert strip position to global phi and replace the nominal phi value/error
      measuredStripPositions.GetTEC2TECEntry( det, beam, disk ) = peakFinderResults;
      const GlobalPoint& globalPoint = theStripDet->surface().toGlobal( theStripDet->specificTopology().localPosition( peakFinderResults.first ) );
      measuredCoordinates.GetTEC2TECEntry( det, beam, disk ).SetPhi( ConvertAngle( globalPoint.barePhi() ) );
      //      measuredCoordinates.GetTEC2TECEntry( det, beam, disk ).SetPhi( nominalCoordinates.GetTEC2TECEntry( det, beam, disk ).GetPhi() ); // ###################### CHEAT 0
      measuredCoordinates.GetTEC2TECEntry( det, beam, disk ).SetPhiError( 0.00047 ); // PRELIMINARY ESTIMATE
    }
    else { // keep nominal position but set a giant phi error so that the module can be ignored by the alignment algorithm
      measuredStripPositions.GetTEC2TECEntry( det, beam, disk ) = std::pair<float,float>( 256, 1000. );
      const GlobalPoint& globalPoint = theStripDet->surface().toGlobal( theStripDet->specificTopology().localPosition( 256 ) );
      measuredCoordinates.GetTEC2TECEntry( det, beam, disk ).SetPhi( ConvertAngle( globalPoint.barePhi() ) );
      //      measuredCoordinates.GetTEC2TECEntry( det, beam, disk ).SetPhi( nominalCoordinates.GetTEC2TECEntry( det, beam, disk ).GetPhi() ); // ###################### CHEAT 0
      measuredCoordinates.GetTEC2TECEntry( det, beam, disk ).SetPhiError( 1000. );
    }

    // fill the histograms for saving
    for( int bin = 1; bin <= 512; ++bin ) {
      summedHistograms.GetTEC2TECEntry( det, beam, disk )->SetBinContent( bin, collectedDataProfiles.GetTEC2TECEntry( det, beam, disk ).GetValue( bin - 1 ) );
    }

  } while( moduleLoop.TEC2TECLoop( det, beam, disk ) );
  





  // now reconstruct the geometry and update the db object
  LASGeometryUpdater geometryUpdater( nominalCoordinates );

  // run the endcap algorithm
  LASEndcapAlgorithm endcapAlgorithm;
  LASEndcapAlignmentParameterSet endcapParameters = endcapAlgorithm.CalculateParameters( measuredCoordinates, nominalCoordinates );
  endcapParameters.Print();

  // do a pre-alignment of the endcaps (TEC2TEC only)
  // so that the alignment tube algorithms finds orderly disks
  //  geometryUpdater.EndcapUpdate( endcapParameters, measuredCoordinates ); ////////////////////////////////////////////////////////////////////////////////////////

  // run the ANALYTICAL alignment tube algorithm
  LASAlignmentTubeAlgorithm alignmentTubeAlgorithm;
  LASBarrelAlignmentParameterSet alignmentTubeParameters = alignmentTubeAlgorithm.CalculateParameters( measuredCoordinates, nominalCoordinates );
    
  // run the MINUIT-BASED alignment tube algorithm
  //     LASBarrelAlgorithm barrelAlgorithm;
  //     LASBarrelAlignmentParameterSet alignmentTubeParameters = barrelAlgorithm.CalculateParameters( measuredCoordinates, nominalCoordinates );

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
  AlignmentErrors* alignmentErrors = theAlignableTracker->alignmentErrors();

  // Write alignments to DB: have to sort beforhand!
  if ( theStoreToDB ) {

    std::cout << " [LaserAlignment::endRun] -- Storing the calculated alignment parameters to the DataBase:" << std::endl;

    // Call service
    edm::Service<cond::service::PoolDBOutputService> poolDbService;
    if( !poolDbService.isAvailable() ) // Die if not available
      throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
    
    // Store
    if ( poolDbService->isNewTagRequest(theAlignRecordName) ) {
      poolDbService->createNewIOV<Alignments>( alignments, poolDbService->currentTime(), poolDbService->endOfTime(), theAlignRecordName );
    }
    else {
      poolDbService->appendSinceTime<Alignments>( alignments, poolDbService->currentTime(), theAlignRecordName );
    }

    if ( poolDbService->isNewTagRequest(theErrorRecordName) ) {
      poolDbService->createNewIOV<AlignmentErrors>( alignmentErrors, poolDbService->currentTime(), poolDbService->endOfTime(), theErrorRecordName );
    }
    else {
      poolDbService->appendSinceTime<AlignmentErrors>( alignmentErrors, poolDbService->currentTime(), theErrorRecordName );
    }
    
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
/// from the event digi containers;
/// yet only SiStripRawDigis, later also SiStripDigis (switchable or overload) !!
///
void LaserAlignment::fillDataProfiles( edm::Event const& theEvent,edm::EventSetup const& theSetup ) {

  edm::Handle< edm::DetSetVector<SiStripRawDigi> > theStripDigis;

  // query conf for what to fetch from event
  for ( Parameters::iterator itDigiProducersList = theDigiProducersList.begin(); itDigiProducersList != theDigiProducersList.end(); ++itDigiProducersList ) {
    std::string digiProducer = itDigiProducersList->getParameter<std::string>( "DigiProducer" );
    std::string digiLabel = itDigiProducersList->getParameter<std::string>( "DigiLabel" );
    theEvent.getByLabel( digiProducer, digiLabel, theStripDigis );
  }


  // indices for the LASGlobalLoop object
  int det = 0, ring = 0, beam = 0, disk = 0, pos = 0;
  

  // loop TEC internal modules
  det = 0; ring = 0; beam = 0; disk = 0;
  do {
    
    // retrieve the raw id of that module
    const int detRawId = detectorId.GetTECEntry( det, ring, beam, disk );
    
    // search the digis for the raw id
    edm::DetSetVector<SiStripRawDigi>::const_iterator detSetIter = theStripDigis->find( detRawId );
    if( detSetIter == theStripDigis->end() ) {
      throw cms::Exception( "Laser Alignment" ) << " [LaserAlignment::fillDataProfiles] ERROR ** No DetSet found for TEC raw id: " << detRawId << "." << std::endl;
    }

    // fill the digis to the profiles
    edm::DetSet<SiStripRawDigi>::const_iterator digiRangeIterator = detSetIter->data.begin(); // for the loop
    edm::DetSet<SiStripRawDigi>::const_iterator digiRangeStart = digiRangeIterator; // save starting positions

    // loop all digis
    for (; digiRangeIterator != detSetIter->data.end(); ++digiRangeIterator ) {
      const SiStripRawDigi *digi = &*digiRangeIterator;
      const int channel = distance( digiRangeStart, digiRangeIterator );
      if ( channel < 512 ) currentDataProfiles.GetTECEntry( det, ring, beam, disk ).SetValue( channel, digi->adc() );
    }
    
  } while( moduleLoop.TECLoop( det, ring, beam, disk ) );


  // loop TIBTOB modules
  det = 2; beam = 0; pos = 0;
  do {
    
    // retrieve the raw id of that module
    const int detRawId = detectorId.GetTIBTOBEntry( det, beam, pos );
    
    // search the digis for the raw id
    edm::DetSetVector<SiStripRawDigi>::const_iterator detSetIter = theStripDigis->find( detRawId );
    if( detSetIter == theStripDigis->end() ) {
      throw cms::Exception( "Laser Alignment" ) << " [LaserAlignment::fillDataProfiles] ERROR ** No DetSet found for TIBTOB raw id: " << detRawId << "." << std::endl;
    }

    // fill the digis to the profiles
    edm::DetSet<SiStripRawDigi>::const_iterator digiRangeIterator = detSetIter->data.begin(); // for the loop
    edm::DetSet<SiStripRawDigi>::const_iterator digiRangeStart = digiRangeIterator; // save starting positions

    // loop all digis
    for (; digiRangeIterator != detSetIter->data.end(); ++digiRangeIterator ) {
      const SiStripRawDigi *digi = &*digiRangeIterator;
      const int channel = distance( digiRangeStart, digiRangeIterator );
      if ( channel < 512 ) currentDataProfiles.GetTIBTOBEntry( det, beam, pos ).SetValue( channel, digi->adc() );
    }

  } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );


  // loop TEC internal modules
  det = 0; beam = 0; disk = 0;
  do {
    
    // retrieve the raw id of that module
    const int detRawId = detectorId.GetTEC2TECEntry( det, beam, disk );
    
    // search the digis for the raw id
    edm::DetSetVector<SiStripRawDigi>::const_iterator detSetIter = theStripDigis->find( detRawId );
    if( detSetIter == theStripDigis->end() ) {
      throw cms::Exception( "Laser Alignment" ) << " [LaserAlignment::fillDataProfiles] ERROR ** No DetSet found for TEC2TEC raw id: " << detRawId << "." << std::endl;
    }

    // fill the digis to the profiles
    edm::DetSet<SiStripRawDigi>::const_iterator digiRangeIterator = detSetIter->data.begin(); // for the loop
    edm::DetSet<SiStripRawDigi>::const_iterator digiRangeStart = digiRangeIterator; // save starting positions

    // loop all digis
    for (; digiRangeIterator != detSetIter->data.end(); ++digiRangeIterator ) {
      const SiStripRawDigi *digi = &*digiRangeIterator;
      const int channel = distance( digiRangeStart, digiRangeIterator );
      if ( channel < 512 ) currentDataProfiles.GetTEC2TECEntry( det, beam, disk ).SetValue( channel, digi->adc() );
    }
    
  } while( moduleLoop.TEC2TECLoop( det, beam, disk ) );

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
///
///
std::vector<int> LaserAlignment::checkBeam(std::vector<std::string>::const_iterator iHistName, std::map<std::string, std::pair<DetId, TH1D*> >::iterator iHist ) {

  std::vector<int> result;
  std::string stringDisc;
  std::string stringRing;
  std::string stringBeam;
  bool isTEC2TEC = false;
  int theDisc = 0;
  int theRing = 0;
  int theBeam = 0;
  int theTECSide = 0;
  
  // check if we are in the Endcap
  switch (((iHist->second).first).subdetId())
    {
    case StripSubdetector::TIB:
      {
	break;
      }
    case StripSubdetector::TOB:
      {
	break;
      }
    case StripSubdetector::TEC:
      {
	TECDetId theTECDetId(((iHist->second).first).rawId());
	
	theTECSide = theTECDetId.side(); // 1 for TEC-, 2 for TEC+
	
	stringBeam = (*iHistName).at(4);
	stringRing = (*iHistName).at(9);
	stringDisc = (*iHistName).at(14);
	isTEC2TEC = ( (*iHistName).size() > 21 ) ? true : false;
	break;
      }
    }

  if ( stringRing == "4" ) { theRing = 4; }
  else if ( stringRing == "6" ) { theRing = 6; }

  if ( stringDisc == "1" ) { theDisc = 0; }
  else if ( stringDisc == "2" ) { theDisc = 1; }
  else if ( stringDisc == "3" ) { theDisc = 2; } 
  else if ( stringDisc == "4" ) { theDisc = 3; } 
  else if ( stringDisc == "5" ) { theDisc = 4; }
  else if ( stringDisc == "6" ) { theDisc = 5; } 
  else if ( stringDisc == "7" ) { theDisc = 6; } 
  else if ( stringDisc == "8" ) { theDisc = 7; } 
  else if ( stringDisc == "9" ) { theDisc = 8; } 

  if ( theRing == 4 )
    {
      if ( stringBeam == "0" ) { theBeam = 0; } 
      else if ( stringBeam == "1" ) { theBeam = 1; } 
      else if ( stringBeam == "2" ) { theBeam = 2; }
      else if ( stringBeam == "3" ) { theBeam = 3; } 
      else if ( stringBeam == "4" ) { theBeam = 4; }
      else if ( stringBeam == "5" ) { theBeam = 5; } 
      else if ( stringBeam == "6" ) { theBeam = 6; } 
      else if ( stringBeam == "7" ) { theBeam = 7; } 
    }
  else if ( theRing == 6 )
    {
      if ( stringBeam == "0" ) { theBeam = 0; } 
      else if ( stringBeam == "1" ) { theBeam = 1; } 
      else if ( stringBeam == "2" ) { theBeam = 2; }
      else if ( stringBeam == "3" ) { theBeam = 3; } 
      else if ( stringBeam == "4" ) { theBeam = 4; }
      else if ( stringBeam == "5" ) { theBeam = 5; } 
      else if ( stringBeam == "6" ) { theBeam = 6; } 
      else if ( stringBeam == "7" ) { theBeam = 7; } 
    }
  result.push_back(theTECSide);
  result.push_back(theRing);
  result.push_back(theBeam);
  result.push_back(theDisc);
  
  return result;
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

  //  LogDebug( "[LaserAlignment::isTECBeam]" ) << " Found: " << numberOfProfiles << "hits." << std::endl;
  std::cout << " [LaserAlignment::isTECBeam] -- Found: " << numberOfProfiles << " hits." << std::endl; /////////////////////////////////

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

  //  LogDebug( "[LaserAlignment::isATBeam]" ) << " Found: " << numberOfProfiles << "hits." << std::endl;
  std::cout << " [LaserAlignment::isATBeam] -- Found: " << numberOfProfiles << " hits." << std::endl; /////////////////////////////////

  if( numberOfProfiles > 10 ) return( true );
  return( false );
    
}





///
/// tib & TOB modules are not hit in the center;
/// this func returns the approximate beam offset for the ProfileJudge and the LASPeakFinder (in strips)
///
double LaserAlignment::getTIBTOBProfileOffset( int det, int beam, int pos ) {

  if( det < 0 || det > 3 || beam < 0 || beam > 7 || pos < 0 || pos > 5 ) {
    throw cms::Exception( "LaserAlignment" ) << "[LaserAlignment::getTIBTOBProfileOffset] ERROR ** Called with nonexisting parameter set: det " << det << " beam " << beam << " pos " << pos << "." << std::endl;
  }

  // no offsets in TECs
  if( !( det == 2 || det == 3 ) ) return 0;

  // plain offsets of the rods with respect to the beams in rad
  const double tobOffsets[8] = { 0.000000, 0.000506, -0.036896, -0.037402, -0.037891, 0.037402, 0.036896, 0.000506 };
  const double tibOffsets[8] = { 0.000000, 0.000506, 0.000506, 0.000000, -0.000506, 0.000000, -0.000506, 0.000506 };

  // this pattern reflects the orientation of the modules on the rods (flips along z)
  const int pattern[6] = { -1, 1, 1, -1, -1, 1 };

  if( det == 2 ) return( pattern[pos] * tibOffsets[beam] * 514 / 0.120 ); // * radius / pitch
  else return( pattern[pos] * tobOffsets[beam] * 600 / 0.183 ); 
  
}





///
///
///
void LaserAlignment::CalculateNominalCoordinates( void ) {

  //
  // hard coded data
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
  //  const double tecZPosition[9] = { 1250., 1390., 1530., 1670., 1810., 1985., 2175., 2380., 2595. }; // old
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
///
///
void LaserAlignment::DumpPosFileSet( LASGlobalData<LASCoordinateSet>& coordinates ) {

  LASGlobalLoop loop;
  int det, ring, beam, disk, pos;

  std:: cout << std::endl << " [LaserAlignment::DumpPosFileSet] -- Dump: " << std::endl;

  // TEC INTERNAL
  det = 0; ring = 0; beam = 0; disk = 0;
  do {
    std::cout << "### " << det << "\t" << beam << "\t" << disk << "\t" << ring << "\t" << coordinates.GetTECEntry( det, ring, beam, disk ).GetPhi() << "\t" << coordinates.GetTECEntry( det, ring, beam, disk ).GetPhiError() << std::endl;
  } while ( loop.TECLoop( det, ring, beam, disk ) );

  // TIBTOB
  det = 2; beam = 0; pos = 0;
  do {
    std::cout << "### " << det << "\t" << beam << "\t" << pos << "\t" << "-1" << "\t" << coordinates.GetTIBTOBEntry( det, beam, pos ).GetPhi() << "\t" << coordinates.GetTIBTOBEntry( det, beam, pos ).GetPhiError() << std::endl;
  } while( loop.TIBTOBLoop( det, beam, pos ) );

  // TEC2TEC
  det = 0; beam = 0; disk = 0;
  do {
    std::cout << "### " << det << "\t" << beam << "\t" << disk << "\t" << "-1" << "\t" << coordinates.GetTEC2TECEntry( det, beam, disk ).GetPhi() << "\t" << coordinates.GetTEC2TECEntry( det, beam, disk ).GetPhiError() << std::endl;
  } while( loop.TEC2TECLoop( det, beam, disk ) );

  std:: cout << std::endl << " [LaserAlignment::DumpPosFileSet] -- End dump: " << std::endl;

}





///
///
///
void LaserAlignment::DumpHitmaps( LASGlobalData<int> numberOfAcceptedProfiles ) {

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

  std::cout << " [LaserAlignment::DumpHitmaps] -- End of dump." << std::endl << std::endl;

}










// define the SEAL module
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(LaserAlignment);




// the ATTIC

