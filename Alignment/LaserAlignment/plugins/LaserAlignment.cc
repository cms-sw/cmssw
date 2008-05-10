/** \file LaserAlignment.cc
 *  LAS reconstruction module
 *
 *  $Date: 2008/05/01 08:12:26 $
 *  $Revision: 1.23 $
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
    theStoreToDB(theConf.getUntrackedParameter<bool>("saveToDbase", false)),
    theSaveHistograms(theConf.getUntrackedParameter<bool>("saveHistograms",false)),
    theDebugLevel(theConf.getUntrackedParameter<int>("DebugLevel",0)),
    theNEventsPerLaserIntensity(theConf.getUntrackedParameter<int>("NumberOfEventsPerLaserIntensity",100)),
    theNEventsForAllIntensities(theConf.getUntrackedParameter<int>("NumberOfEventsForAllIntensities",100)),
    theDoAlignmentAfterNEvents(theConf.getUntrackedParameter<int>("DoAlignmentAfterNEvents",1000)),
    // the following three are hard-coded until the complete package has been refurbished
    theAlignPosTEC( false ), // theAlignPosTEC(theConf.getUntrackedParameter<bool>("AlignPosTEC",false)),
    theAlignNegTEC( false ), // theAlignNegTEC(theConf.getUntrackedParameter<bool>("AlignNegTEC",false)), 
    theAlignTEC2TEC( false ), // theAlignTEC2TEC(theConf.getUntrackedParameter<bool>("AlignTECTIBTOBTEC",false)),
    theUseBrunosAlgorithm(theConf.getUntrackedParameter<bool>("UseBrunosAlignmentAlgorithm",true)),
    theUseNewAlgorithms(theConf.getUntrackedParameter<bool>("UseNewAlgorithms",true)),
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
    theLASAlignPosTEC(),
    theLASAlignNegTEC(),
    theLASAlignTEC2TEC(),
    theAlignmentAlgorithmBW(),
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
				  << "\n    theAlignPosTEC              = " << theAlignPosTEC
				  << "\n    theAlignNegTEC              = " << theAlignNegTEC
				  << "\n    theAlignTEC2TEC             = " << theAlignTEC2TEC
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
  produces<edm::DetSetVector<SiStripDigi> >().setBranchAlias( alias + "siStripDigis" );
  produces<LASBeamProfileFitCollection>().setBranchAlias( alias + "LASBeamProfileFits" );
  produces<LASAlignmentParameterCollection>().setBranchAlias( alias + "LASAlignmentParameters" );


  // the alignable tracker parts
  theLASAlignPosTEC = new LaserAlignmentPosTEC;
  theLASAlignNegTEC = new LaserAlignmentNegTEC;
  theLASAlignTEC2TEC = new LaserAlignmentTEC2TEC;
  
  // the alignment algorithm from Bruno
  theAlignmentAlgorithmBW = new AlignmentAlgorithmBW;
  
  // counter for the number of iterations, i.e. the number of BeamProfile fits and
  // local Millepede fits
  theNumberOfIterations = 0;
}





LaserAlignment::~LaserAlignment() {

  if (theSaveHistograms) {
    closeRootFile();
  }
  
  if (theFile != 0) { delete theFile; }
  
  if (theBeamFitter != 0) { delete theBeamFitter; }
  
  if (theLASAlignPosTEC != 0) { delete theLASAlignPosTEC; }
  if (theLASAlignNegTEC != 0) { delete theLASAlignNegTEC; }
  if (theLASAlignTEC2TEC != 0) { delete theLASAlignTEC2TEC; }
  if (theAlignableTracker != 0) { delete theAlignableTracker; }
  if (theAlignmentAlgorithmBW != 0) { delete theAlignmentAlgorithmBW; }
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

    // init the hit map
    isAcceptedProfile.SetTECEntry( det, ring, beam, disk, 0 );
    
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

    // init the hit map
    isAcceptedProfile.SetTIBTOBEntry( det, beam, pos, 0 );

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
    
    // init the hit map
    isAcceptedProfile.SetTEC2TECEntry( det, beam, disk, 0 );

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
  theAlignableTracker = new AlignableTracker( &(*theTrackerGeometry) );

}






///
///
///
void LaserAlignment::produce(edm::Event& theEvent, edm::EventSetup const& theSetup)  {


  LogDebug("LaserAlignment") << "==========================================================="
			      << "\n   Private analysis of event #"<< theEvent.id().event() 
			      << " in run #" << theEvent.id().run();


  // do the Tracker Statistics to retrieve the current profiles
  trackerStatistics( theEvent, theSetup );



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
    if( judge.IsSignalIn( currentDataProfiles.GetTIBTOBEntry( det, beam, pos ) - pedestalProfiles.GetTIBTOBEntry( det, beam, pos ), getTOBProfileOffset( det, beam, pos ) ) ) {
      isAcceptedProfile.SetTIBTOBEntry( det, beam, pos, 1 );
    }
    else { // dto.
      isAcceptedProfile.SetTIBTOBEntry( det, beam, pos, 0 );
    }

  } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );




  // now come the beam finders
  bool isTECMode = isTECBeam();
  LogDebug( "[LaserAlignment::produce]" ) << "LaserAlignment::isTECBeam declares this event " << ( isTECMode ? "" : "NOT " ) << "a TEC event." << std::endl;
  std::cout << "[LaserAlignment::produce] -- LaserAlignment::isTECBeam declares this event " << ( isTECMode ? "" : "NOT " ) << "a TEC event." << std::endl;

  bool isATMode  = isATBeam();
  LogDebug( "[LaserAlignment::produce]" ) << "LaserAlignment::isATBeam declares this event "  << ( isATMode ? "" : "NOT " )  << "an AT event." << std::endl;
  std::cout << "[LaserAlignment::produce] -- LaserAlignment::isATBeam declares this event "  << ( isATMode ? "" : "NOT " )  << "an AT event." << std::endl;




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
    if( judge.JudgeProfile( currentDataProfiles.GetTIBTOBEntry( det, beam, pos ) - pedestalProfiles.GetTIBTOBEntry( det, beam, pos ), getTOBProfileOffset( det, beam, pos ) ) ) {
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
void LaserAlignment::endJob() {

  LogDebug("LaserAlignment") << "     Total Event number = " << theEvents;

  // index variables for the LASGlobalLoop objects
  int det, ring, beam, disk, pos;
  
  if( theUseNewAlgorithms ) {

    /////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////
    // this is the new barrel algorithm section, it uses LASPeakFinder to evaluate signals //
    /////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////


    // measured positions container for the algorithms
    LASGlobalData<LASCoordinateSet> measuredCoordinates;
    
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
      if( !isGoodFit ) std::cout << " [LaserAlignment::endJob] ** WARNING: Fit failed for TEC det: "
				 << det << ", ring: " << ring << ", beam: " << beam << ", disk: " << disk << "." << std::endl;

	// <- here we will later implement the kink corrections
    
	// access the tracker geometry for this module
	const DetId theDetId( detectorId.GetTECEntry( det, ring, beam, disk ) );
	const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTracker.idToDet( theDetId ) );

	// first, set the measured coordinates to their nominal values
	measuredCoordinates.SetTECEntry( det, ring, beam, disk, nominalCoordinates.GetTECEntry( det, ring, beam, disk ) );

	if( isGoodFit ) { // convert strip position to global phi and replace the nominal phi value/error
	  const GlobalPoint& globalPoint = theStripDet->surface().toGlobal( theStripDet->specificTopology().localPosition( peakFinderResults.first ) );
	  measuredCoordinates.GetTECEntry( det, ring, beam, disk ).SetPhi( ConvertAngle( globalPoint.barePhi() ) );
	  measuredCoordinates.GetTECEntry( det, ring, beam, disk ).SetPhiError( 0.00046  ); // PRELIMINARY ESTIMATE
	  //      const GlobalError& globalError = errorTransformer.transform( theStripDet->specificTopology().localError( peakFinderResults.first, pow( peakFinderResults.second, 2 ) ), theStripDet->surface() );
	  //      measuredCoordinates.GetTECEntry( det, ring, beam, disk ).SetPhiError( globalError.phierr( globalPoint ) );
	}
	else { // keep nominal position but set a giant phi error so that the module is ignored by the alignment algorithm
	  measuredCoordinates.GetTECEntry( det, ring, beam, disk ).SetPhiError( 1000. );
	}
    
    } while( moduleLoop.TECLoop( det, ring, beam, disk ) );




    // do the fits for TIB/TOB
    det = 2; beam = 0; pos = 0;
    do {

      // do the fit
      isGoodFit = peakFinder.FindPeakIn( collectedDataProfiles.GetTIBTOBEntry( det, beam, pos ), peakFinderResults, getTOBProfileOffset( det, beam, pos ) );
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
	  measuredCoordinates.GetTIBTOBEntry( det, beam, pos ).SetPhi( ConvertAngle( theStripDet->surface().toGlobal( theStripDet->specificTopology().localPosition( peakFinderResults.first ) ).barePhi() ) );
	  measuredCoordinates.GetTIBTOBEntry( det, beam, pos ).SetPhiError( 0.00028 ); // PRELIMINARY ESTIMATE
	}
	else { // keep nominal position but set a giant phi error so that the module is ignored by the alignment algorithm
	  measuredCoordinates.GetTIBTOBEntry( det, beam, pos ).SetPhiError( 1000. );
	}

    } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );




    // do the fits for TEC AT
    det = 0; beam = 0; disk = 0;
    do {

      // do the fit
      isGoodFit = peakFinder.FindPeakIn( collectedDataProfiles.GetTEC2TECEntry( det, beam, disk ), peakFinderResults, 0 ); // no offset for TEC modules
      // now we have the positions in units of strips.
      if( !isGoodFit ) std::cout << " [LaserAlignment::endJob] ** WARNING: Fit failed for TEC2TEC det: "
				 << det << ", beam: " << beam << ", disk: " << disk << "." << std::endl;

	// <- here we will later implement the kink corrections
    
	// access the tracker geometry for this module
	const DetId theDetId( detectorId.GetTEC2TECEntry( det, beam, pos ) );
	const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTracker.idToDet( theDetId ) );

	// first, set the measured coordinates to their nominal values
	measuredCoordinates.SetTEC2TECEntry( det, beam, disk, nominalCoordinates.GetTEC2TECEntry( det, beam, disk ) );
    
	if( isGoodFit ) { // convert strip position to global phi and replace the nominal phi value/error
	  measuredCoordinates.GetTEC2TECEntry( det, beam, disk ).SetPhi( ConvertAngle( theStripDet->surface().toGlobal( theStripDet->specificTopology().localPosition( peakFinderResults.first ) ).barePhi() ) );
	  measuredCoordinates.GetTEC2TECEntry( det, beam, disk ).SetPhiError( 0.00047 ); // PRELIMINARY ESTIMATE
	}
	else { // keep nominal position but set a giant phi error so that the module is ignored by the alignment algorithm
	  measuredCoordinates.GetTEC2TECEntry( det, beam, disk ).SetPhiError( 1000. );
	}

    } while( moduleLoop.TEC2TECLoop( det, beam, disk ) );
  



    // run the algorithms
    LASBarrelAlignmentParameterSet barrelParameters = barrelAlgorithm.CalculateParameters( measuredCoordinates, nominalCoordinates );
    LASEndcapAlignmentParameterSet endcapParameters = endcapAlgorithm.CalculateParameters( measuredCoordinates, nominalCoordinates );

    // combine the results and update the db object
    LASGeometryUpdater geometryUpdater;
    geometryUpdater.Update( endcapParameters, barrelParameters, *theAlignableTracker );
  

  } // if( !theUseBrunosAlgorithm )








  else if( !theUseNewAlgorithms ) {

    /////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////
    // this is the old algorithm section, it uses BeamProfileFitter to evaluate signals,   //
    // this will be replaced and work the same way as the barrel part                      //
    /////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////
    
    
    
    // create an empty output collection
    //  std::auto_ptr<LASBeamProfileFitCollection> theFitOutput(new LASBeamProfileFitCollection);
    //  std::auto_ptr<LASAlignmentParameterCollection> theAlignmentParameterOutput(new LASAlignmentParameterCollection);
    
    theDigiVector.reserve(10000);
    theDigiVector.clear();
    
    // vectors to store the beam positions for usage with Bruno's alignment algorithm
    LASvec2D thePosTECR4BeamPositions(8,9);
    LASvec2D thePosTECR6BeamPositions(8,9);
    LASvec2D theNegTECR4BeamPositions(8,9);
    LASvec2D theNegTECR6BeamPositions(8,9);
    LASvec2D theTEC2TECBeamPositions(8,22);
  
    LASvec2D thePosTECR4BeamPositionErrors(8,9);
    LASvec2D thePosTECR6BeamPositionErrors(8,9);
    LASvec2D theNegTECR4BeamPositionErrors(8,9);
    LASvec2D theNegTECR6BeamPositionErrors(8,9);
    LASvec2D theTEC2TECBeamPositionErrors(8,22);

    LogDebug("LaserAlignment") << "===========================================================";
  
    // the fit of the beam profiles
    //  if (theEvents % theNEventsPerLaserIntensity == 0)
    //  {
    // do the beam profile fit
    //    fit(theSetup);
    //  }

    //  if (theEvents % theNEventsForAllIntensities == 0)
    //  {


  
    std::vector<LASBeamProfileFit> collector;

    // do the fits for TEC+-
    det = 0; ring = 0; beam = 0; disk = 0;
    do {

      // fill histo from collected data.
      // later, the fitter will accept the profile directly
      for( int bin = 0; bin < 512; ++bin ) {
	summedHistograms.GetTECEntry( det, ring, beam, disk )->SetBinContent( 1 + bin, collectedDataProfiles.GetTECEntry( det, ring, beam, disk ).GetValue( bin ) );
      }

      // will store the detids in a separate container later, yet use Maarten's construct here
      std::map<std::string, std::pair<DetId, TH1D*> >::iterator iHist = theHistograms.find( theProfileNames.GetTECEntry( det, ring, beam, disk ) );
      if ( iHist != theHistograms.end() ) {
      
	// here we prepare the preliminary interface to Maarten's code
	int aRing, aBeam, aDet;
	if( ring == 0 ) { // this is Ring4
	  aRing = 4;
	  aBeam = beam;
	}
	else { // this is Ring6
	  aRing = 6;
	  aBeam = beam + 8;
	}
	if( det == 0 ) aDet = 2; // TEC+
	else aDet = 1; // TEC-

      
	// pass the histo to the beam fitter
	collector = theBeamFitter->doFit((iHist->second).first,
					 summedHistograms.GetTECEntry( det, ring, beam, disk ), 
					 theSaveHistograms,
					 numberOfAcceptedProfiles.GetTECEntry( det, ring, beam, disk ), 
					 aBeam, disk, aRing, 
					 aDet, false /*isTEC2TEC*/, theIsGoodFit );

	// if the fit succeeded, add the LASBeamProfileFit to the output collection for storage
	// and additionally add the LASBeamProfileFit to the map for later processing (we need
	// the info from the fit for the Alignment Algorithm)
	if ( theIsGoodFit ) {
	  // add the result of the fit to the map
	  theBeamProfileFitStore[theProfileNames.GetTECEntry( det, ring, beam, disk )] = collector;
	}
      
	// set theIsGoodFit to false again for the next fit
	theIsGoodFit = false;


      }
      else std::cerr << "[LaserAlignment::endJob] ** WARNING: No pair<DetId,TH1D*> found for TEC det " 
		     << det << " ring " << ring << " beam " << beam << " disk " << disk << "." << std::endl;

    } while( moduleLoop.TECLoop( det, ring, beam, disk ) );

  

    // do the fits for TIB/TOB
    det = 2; beam = 0; pos = 0;
    do {

      // fill histo from collected data.
      // later, the fitter will accept the profile directly
      for( int bin = 0; bin < 512; ++bin ) {
	summedHistograms.GetTIBTOBEntry( det, beam, pos )->SetBinContent( 1 + bin, collectedDataProfiles.GetTIBTOBEntry( det, beam, pos ).GetValue( bin ) );
      }

      // will store the detids in a separate container later, yet use Maarten's construct here
      std::map<std::string, std::pair<DetId, TH1D*> >::iterator iHist = theHistograms.find( theProfileNames.GetTIBTOBEntry( det, beam, pos ) );
      if ( iHist != theHistograms.end() ) {
      
      
	// here we prepare the interface to Maarten's code
	int aRing = 0, aBeam = 0, aDet = 0;
	disk = 0;
      
	// pass the histo to the beam fitter
	collector = theBeamFitter->doFit((iHist->second).first,
					 summedHistograms.GetTIBTOBEntry( det, beam, pos ), 
					 theSaveHistograms,
					 numberOfAcceptedProfiles.GetTIBTOBEntry( det, beam, pos ), 
					 aBeam, disk, aRing, 
					 aDet, false /*isTEC2TEC*/, theIsGoodFit );

	// if the fit succeeded, add the LASBeamProfileFit to the output collection for storage
	// and additionally add the LASBeamProfileFit to the map for later processing (we need
	// the info from the fit for the Alignment Algorithm)
	if ( theIsGoodFit ) {
	  // add the result of the fit to the map
	  theBeamProfileFitStore[theProfileNames.GetTIBTOBEntry( det, beam, pos )] = collector;
	}
      
	// set theIsGoodFit to false again for the next fit
	theIsGoodFit = false;


      }
      else std::cerr << "[LaserAlignment::endJob] ** WARNING: No pair<DetId,TH1D*> found for TIBTOB det " 
		     << det << " beam " << beam << " pos " << pos << "." << std::endl;
    
    } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );







    // now the fits for TEC AT
    det = 0; ring = 0; beam = 0; disk = 0;
    do {

      // fill histo from collected data.
      // later, the fitter will accept the profile directly
      for( int bin = 0; bin < 512; ++bin ) {
	summedHistograms.GetTEC2TECEntry( det, beam, disk )->SetBinContent( 1 + bin, collectedDataProfiles.GetTEC2TECEntry( det, beam, disk ).GetValue( bin ) );
      }

      // will store the detids in a separate container later, yet use Maarten's construct here
      std::map<std::string, std::pair<DetId, TH1D*> >::iterator iHist = theHistograms.find( theProfileNames.GetTEC2TECEntry( det, beam, disk ) );
      if ( iHist != theHistograms.end() ) {
      
	// here we prepare the interface to Maarten's code
	int aRing, aBeam, aDet;
	aRing = 4; // always Ring4
	aBeam = beam;
	if( det == 0 ) aDet = 2; // TEC+
	else aDet = 1; // TEC-

      
	// pass the histo to the beam fitter
	collector = theBeamFitter->doFit((iHist->second).first,
					 summedHistograms.GetTEC2TECEntry( det, beam, disk ), 
					 theSaveHistograms,
					 numberOfAcceptedProfiles.GetTEC2TECEntry( det, beam, disk ), 
					 aBeam, disk, aRing, 
					 aDet, true /*isTEC2TEC*/, theIsGoodFit );

	// if the fit succeeded, add the LASBeamProfileFit to the output collection for storage
	// and additionally add the LASBeamProfileFit to the map for later processing (we need
	// the info from the fit for the Alignment Algorithm)
	if ( theIsGoodFit ) {
	  // add the result of the fit to the map
	  theBeamProfileFitStore[theProfileNames.GetTEC2TECEntry( det, beam, disk )] = collector;
	}
      
	// set theIsGoodFit to false again for the next fit
	theIsGoodFit = false;

      }
      else {
	std::cerr << "[LaserAlignment::endJob] ** WARNING: No pair<DetId,TH1D*> found for TEC AT det " 
		  << det << " beam " << beam << " disk " << disk << "." << std::endl;
	std::cerr << "                                     (if beam=0,3,5 it's probably ok)" << std::endl;
      }

    } while( moduleLoop.TEC2TECLoop( det, beam, disk ) );




    // increase the counter for the iterations
    //  theNumberOfIterations++;

    // put the digis of the beams into the StripDigiCollection
    for ( std::map<DetId, std::vector<SiStripRawDigi> >::const_iterator p = theDigiStore.begin(); p != theDigiStore.end(); ++p ) {

      edm::DetSet<SiStripRawDigi> collector((p->first).rawId());
    
      if ( ( p->second ).size() > 0 ) {
	collector.data = (p->second);
	theDigiVector.push_back(collector);
      }
    }

    // clear the map to fill new digis for the next theNEventsForAllIntensities number of events
    theDigiStore.clear();


    // put the LASBeamProfileFits into the LASBeamProfileFitCollection
    // loop over the map with the histograms and lookup the LASBeamFits
    for ( std::vector<std::string>::const_iterator iHistName = theHistogramNames.begin(); iHistName != theHistogramNames.end(); ++iHistName ) {

      std::map<std::string, std::pair<DetId, TH1D*> >::iterator iHist = theHistograms.find(*iHistName);
      if ( iHist != theHistograms.end() ) {
      
	std::map<std::string, std::vector<LASBeamProfileFit> >::iterator iBeamFit = theBeamProfileFitStore.find(*iHistName);
	if ( iBeamFit != theBeamProfileFitStore.end() ) {
	
	  // get the DetId from the map with histograms
	  //	unsigned int theDetId = ((iHist->second).first).rawId();
	  // get the information for the LASBeamProfileFitCollection
	  LASBeamProfileFitCollection::Range inputRange;
	  inputRange.first = (iBeamFit->second).begin();
	  inputRange.second = (iBeamFit->second).end();
	
	  //	theFitOutput->put(inputRange,theDetId); // no production yet
	
	  // now fill the fitted phi position and error into the appropriate vectors
	  // which will be used by the Alignment Algorithm
	  LASBeamProfileFit theFit = (iBeamFit->second).at(0);
	  theLaserPhi.push_back(theFit.phi());
	  theLaserPhiError.push_back(thePhiErrorScalingFactor * theFit.phiError());
	
	  // fill also the LASvec2D for Bruno's algorithm
	  std::vector<int> sectorLocation = checkBeam(iHistName, iHist);
	
	  if (sectorLocation.at(0) == 1) // TEC- beams
	    {
	      if (sectorLocation.at(1) == 4) // Ring 4
		{
		  theNegTECR4BeamPositions[sectorLocation.at(2)][sectorLocation.at(3)] = theFit.pitch() * (theFit.mean()-255.5);
		  theNegTECR4BeamPositionErrors[sectorLocation.at(2)][sectorLocation.at(3)] = theFit.pitch() * theFit.meanError();
		}
	      else if (sectorLocation.at(1) == 6) // Ring 6
		{
		  theNegTECR6BeamPositions[sectorLocation.at(2)][sectorLocation.at(3)] = theFit.pitch() * (theFit.mean()-255.5);
		  theNegTECR6BeamPositionErrors[sectorLocation.at(2)][sectorLocation.at(3)] = theFit.pitch() * theFit.meanError();
		}
	    }
	  else if (sectorLocation.at(0) == 2) // TEC+ beams
	    {
	      if (sectorLocation.at(1) == 4) // Ring 4
		{
		  thePosTECR4BeamPositions[sectorLocation.at(2)][sectorLocation.at(3)] = theFit.pitch() * (theFit.mean()-255.5);
		  thePosTECR4BeamPositionErrors[sectorLocation.at(2)][sectorLocation.at(3)] = theFit.pitch() * theFit.meanError();
		}
	      else if (sectorLocation.at(1) == 6) // Ring 6
		{
		  thePosTECR6BeamPositions[sectorLocation.at(2)][sectorLocation.at(3)] = theFit.pitch() * (theFit.mean()-255.5);
		  thePosTECR6BeamPositionErrors[sectorLocation.at(2)][sectorLocation.at(3)] = theFit.pitch() * theFit.meanError();
		}				
	    }
	  // implementation of TEC2TEC beams has to be done!!!!!!
	}
	else
	  {
	    // no BeamFit found for this layer, use the nominal phi position of the Det for the Alignment Algorithm
	    // 		  // access the tracker
	    // 		  edm::ESHandle<TrackerGeometry> theTrackerGeometry;
	    // 		  theSetup.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
	  
	    double thePhi = angle(theTrackerGeometry->idToDet((iHist->second).first)->surface().position().phi());
	    double thePhiError = 0.1;
	  
	    LogDebug("LaserAlignment") << " no LASBeamProfileFit found for " << (*iHistName) << "! Use nominal phi position (" 
				       << thePhi << ") for alignment ";
	    theLaserPhi.push_back(thePhi);
	    theLaserPhiError.push_back(thePhiErrorScalingFactor * thePhiError);
	  
	    /// for Bruno's algorithm get the pitch at strip 255.5 and multiply it with 255.5 to get the position in cm at the middle of the module
	    //           const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>(theTrackerGeometry->idToDet((iHist->second).first));
	  
	    // double thePosition = 255.5 * theStripDet->specificTopology().localPitch(theStripDet->specificTopology().localPosition(255.5));
	    double thePosition = 0.0;
	    double thePositionError = 0.05; // set the error to half a milllimeter in this case
	  
	    // fill also the LASvec2D for Bruno's algorithm
	    std::vector<int> sectorLocation = checkBeam(iHistName, iHist);
	  
	    if (sectorLocation.at(0) == 1) // TEC- beams
	      {
		if (sectorLocation.at(1) == 4) // Ring 4
		  {
		    theNegTECR4BeamPositions[sectorLocation.at(2)][sectorLocation.at(3)] = thePosition;
		    theNegTECR4BeamPositionErrors[sectorLocation.at(2)][sectorLocation.at(3)] = thePositionError;
		  }
		else if (sectorLocation.at(1) == 6) // Ring 6
		  {
		    theNegTECR6BeamPositions[sectorLocation.at(2)][sectorLocation.at(3)] = thePosition;
		    theNegTECR6BeamPositionErrors[sectorLocation.at(2)][sectorLocation.at(3)] = thePositionError;
		  }
	      }
	    else if (sectorLocation.at(0) == 2) // TEC+ beams
	      {
		if (sectorLocation.at(1) == 4) // Ring 4
		  {
		    thePosTECR4BeamPositions[sectorLocation.at(2)][sectorLocation.at(3)] = thePosition;
		    thePosTECR4BeamPositionErrors[sectorLocation.at(2)][sectorLocation.at(3)] = thePositionError;
		  }
		else if (sectorLocation.at(1) == 6) // Ring 6
		  {
		    thePosTECR6BeamPositions[sectorLocation.at(2)][sectorLocation.at(3)] = thePosition;
		    thePosTECR6BeamPositionErrors[sectorLocation.at(2)][sectorLocation.at(3)] = thePositionError;
		  }				
	      }
	    // implementation of TEC2TEC beams has to be done!!!!!
	  }
      }
      else 
	{ 
	  throw cms::Exception("LaserAlignment") << " You are in serious trouble!!! no entry for " << (*iHistName) << " found. "
						 << " To avoid the calculation of wrong alignment corrections the program will abort! "; 
	}
    }
    // clear the map to fill new LASBeamProfileFits for the next theNEventsForAllIntensities number of events
    theBeamProfileFitStore.clear();
    //  }
  
    // do the alignment of the tracker
    //   if (theEvents % theDoAlignmentAfterNEvents == 0) // do this in any case!!!
    //     {
    //       // Create the alignable hierarchy
    //       theAlignableTracker = new AlignableTracker( &(*gD),
    // 						  &(*theTrackerGeometry) );

    // do the Alignment of the Tracker (with Millepede) ...
    //  alignmentAlgorithm(theAlignmentAlgorithmPS, theAlignableTracker);
    // set the number of iterations to zero for the next alignment round *** NOT NEEDED FOR BRUNO'S ALGO!
    theNumberOfIterations = 0;


    if (theUseBrunosAlgorithm) {			

      // do the Alignment with Bruno's algorithm
      std::vector<LASAlignmentParameter> theAPPosTECR4;
      std::vector<LASAlignmentParameter> theAPPosTECR6;
      std::vector<LASAlignmentParameter> theAPNegTECR4;
      std::vector<LASAlignmentParameter> theAPNegTECR6;
      // which return type do we need to calculate the final correction ((corr_R4 + corr_R6)/2)??
      // get them from the returned std::vector<LASAlignmentParameter>!
      theAPPosTECR4 = theAlignmentAlgorithmBW->run("TEC+Ring4",thePosTECR4BeamPositions, 
						   thePosTECR4BeamPositionErrors, theUseBSFrame, 4);
      theAPPosTECR6 = theAlignmentAlgorithmBW->run("TEC+Ring6",thePosTECR6BeamPositions, 
						   thePosTECR6BeamPositionErrors, theUseBSFrame, 6);
      theAPNegTECR4 = theAlignmentAlgorithmBW->run("TEC-Ring4",theNegTECR4BeamPositions, 
						   theNegTECR4BeamPositionErrors, theUseBSFrame, 4);
      theAPNegTECR6 = theAlignmentAlgorithmBW->run("TEC-Ring6",theNegTECR6BeamPositions, 
						   theNegTECR6BeamPositionErrors, theUseBSFrame, 6);
      // implementation of TEC2TEC has to be done!!!!!
      // theAlignmentAlgorithmBW->run(theTEC2TECBeamPositions, theTEC2TECBeamPositionErrors);

      // put the alignment parameters in the output collection
      // first TEC+ Ring 4
      LASAlignmentParameterCollection::Range inputRangeT1R4;
      inputRangeT1R4.first = theAPPosTECR4.begin();
      inputRangeT1R4.second = theAPPosTECR4.end();
      //        theAlignmentParameterOutput->put(inputRangeT1R4,"TEC+Ring4");
      // now TEC+ Ring 6
      LASAlignmentParameterCollection::Range inputRangeT1R6;
      inputRangeT1R6.first = theAPPosTECR6.begin();
      inputRangeT1R6.second = theAPPosTECR6.end();
      //        theAlignmentParameterOutput->put(inputRangeT1R6,"TEC+Ring6");
      // now TEC- Ring 4
      LASAlignmentParameterCollection::Range inputRangeT2R4;
      inputRangeT2R4.first = theAPNegTECR4.begin();
      inputRangeT2R4.second = theAPNegTECR4.end();
      //        theAlignmentParameterOutput->put(inputRangeT2R4,"TEC-Ring4");
      // finally TEC- Ring 6
      LASAlignmentParameterCollection::Range inputRangeT2R6;
      inputRangeT2R6.first = theAPNegTECR6.begin();
      inputRangeT2R6.second = theAPNegTECR6.end();
      //        theAlignmentParameterOutput->put(inputRangeT2R6,"TEC-Ring6");

      // apply the calculated corrections to the alignable Tracker. This will propagate the corrections from the
      // higher level hierarchy (e.g. the TEC discs) to the lowest level in the Tracker Hierarchy: the Dets
      /**
       * TO BE DONE!!!!!!!!!!
       **/
      const align::Alignables& TECs = theAlignableTracker->endCaps();

      const Alignable *TECplus = TECs[0], *TECminus = TECs[1];

      if (TECplus->globalPosition().z() < 0) // reverse order
	{
	  TECplus = TECs[1]; TECminus = TECs[0];
	}

      const align::Alignables& posDisks = TECplus->components();
      const align::Alignables& negDisks = TECminus->components();

      for (unsigned int i = 0; i < 9; ++i) {
	
	align::GlobalVector translationPos( -1.0 * (theAPPosTECR4.at(0).dxk()[i] + theAPPosTECR6.at(0).dxk()[i])/2,
					    -1.0 * (theAPPosTECR4.at(0).dyk()[i] + theAPPosTECR6.at(0).dyk()[i])/2,
					    0.0);
	align::GlobalVector translationNeg( -1.0 * (theAPNegTECR4.at(0).dxk()[i] + theAPNegTECR6.at(0).dxk()[i])/2,
					    -1.0 * (theAPNegTECR4.at(0).dyk()[i] + theAPNegTECR6.at(0).dyk()[i])/2,
					    0.0);
	// TODO - Implement usage and propagation of errors!!!!
	AlignmentPositionError positionErrorPos(0.0, 0.0, 0.0);
	align::RotationType rotationErrorPos( Basic3DVector<float>(0.0, 0.0, 1.0), 0.0 );
	AlignmentPositionError positionErrorNeg(0.0, 0.0, 0.0);
	align::RotationType rotationErrorNeg( Basic3DVector<float>(0.0, 0.0, 1.0), 0.0 );
	Alignable* posDisk = posDisks[i];
	Alignable* negDisk = negDisks[i];
	
	// TEC+
	posDisk->move(translationPos);
	posDisk->addAlignmentPositionError(positionErrorPos);
	posDisk->rotateAroundGlobalZ(-1.0 * (theAPPosTECR4.at(0).dphik()[i] + theAPPosTECR6.at(0).dphik()[i])/2);
	posDisk->addAlignmentPositionErrorFromRotation(rotationErrorPos);
	// TEC-
	negDisk->move(translationNeg);
	negDisk->addAlignmentPositionError(positionErrorNeg);
	negDisk->rotateAroundGlobalZ(-1.0 * (theAPNegTECR4.at(0).dphik()[i] + theAPNegTECR6.at(0).dphik()[i])/2);
	negDisk->addAlignmentPositionErrorFromRotation(rotationErrorNeg);
      }
    }


  } // if( !theUseNewAlgorithms )
  

  



  // store the estimated alignment parameters into the DB
  // first get them
  Alignments* alignments =  theAlignableTracker->alignments();
  AlignmentErrors* alignmentErrors = theAlignableTracker->alignmentErrors();

  // Write alignments to DB: have to sort beforhand!
  if ( theStoreToDB ) {

    std::cout << " [LaserAlignment::endJob] -- Storing the calculated alignment parameters to the DataBase:" << std::endl;

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
						       
    std::cout << " [LaserAlignment::endJob] -- Storing done." << std::endl;
    
}



  //       // Store result to EventSetup
  //       GeometryAligner aligner;
  //       aligner.applyAlignments<TrackerGeometry>( &(*theTrackerGeometry), &(*alignments), &(*alignmentErrors) );

  //    }

  // create the output collection for the DetSetVector
  std::auto_ptr<edm::DetSetVector<SiStripRawDigi> > theDigiOutput(new edm::DetSetVector<SiStripRawDigi>(theDigiVector));

  // write output to file
  //    theEvent.put(theDigiOutput);
  //    theEvent.put(theFitOutput);
  //    theEvent.put(theAlignmentParameterOutput);

  //   // clear the vector with pairs of DetIds and Histograms
  //   theHistograms.clear();

}





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
void LaserAlignment::fillAdcCounts( TH1D* theHistogram, DetId theDetId,
				    edm::DetSet<SiStripRawDigi>::const_iterator digiRangeIterator,
				    edm::DetSet<SiStripRawDigi>::const_iterator digiRangeIteratorEnd,
				    LASModuleProfile& theProfile ) {

  if (theDebugLevel > 4) std::cout << "<LaserAlignment::fillAdcCounts()>: DetUnit: " << theDetId.rawId() << std::endl;

  // save the first position to calculate the index (= channel/strip number)
  int channel = 0;
  edm::DetSet<SiStripRawDigi>::const_iterator theFirstPosition = digiRangeIterator;

  // loop over all the digis in this det
  for (; digiRangeIterator != digiRangeIteratorEnd; ++digiRangeIterator) {
    const SiStripRawDigi *digi = &*digiRangeIterator;

    // store the digis from the laser beams. They are later on used to create
    // clusters and RecHits. In this way some sort of "Laser Tracks" can be
    // reconstructed, which are useable for Track Based Alignment      
    theDigiStore[theDetId].push_back((*digi));
    
    //    if ( theDebugLevel > 5 ) { 
    //      std::cout << " Channel " << digi->channel() << " has " << digi->adc() << " adc counts " << std::endl; 
    //    }
    
    // fill the number of adc counts in the histogram & array
    channel = distance( theFirstPosition, digiRangeIterator );
    if ( channel < 512 ) {
      Double_t theBinContent = theHistogram->GetBinContent( channel + 1 ) + digi->adc();
      theHistogram->SetBinContent( channel + 1, theBinContent );
      theProfile.SetValue( channel, digi->adc() );
    }
    
  }
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
/// TOB modules of beams 2-6 are not hit in the center;
/// this func returns the approximate beam offset for the ProfileJudge and the LASPeakFinder (in strips)
///
int LaserAlignment::getTOBProfileOffset( int det, int beam, int pos ) {

  if( det < 0 || det > 3 || beam < 0 || beam > 7 || pos < 0 || pos > 5 ) {
    throw cms::Exception("LaserAlignment") << "[LaserAlignment::getTOBProfileOffset] ERROR ** Called with nonexisting par: det " << det << " beam " << beam << " pos " << pos << "." << std::endl;
  }

  // only for TOB
  if( det != 3 ) return 0;

  // some beams have no offset
  if( beam < 2 || beam == 7 ) return 0;

  // two groups of beams with offsets: 2-4 and 5-6
  else {
    if( beam < 5 ) {
      int offsets[] = { 117, -120, -120, 117, 117, -120 };
      return( offsets[pos] );
    }
    else {
      int offsets[] = { -120, 117, 117, -120, -120, 117 };
      return( offsets[pos] );
    }
  }

}





///
///
///
void LaserAlignment::CalculateNominalCoordinates( void ) {

  //
  // hard coded data
  //

  // nominal phi values of alignment tube hits (parameter is beam 0-7)
  const double tobPhiPositions[8]   = { 0.3927, 1.2903, 1.8513, 2.7488, 3.6465, 4.3197, 5.2172, 5.7782 }; // the first three are
  const double tibPhiPositions[8]   = { 0.3927, 1.2903, 1.8513, 2.7488, 3.6465, 4.3197, 5.2172, 5.7782 }; // identical (determined
  const double tecATPhiPositions[8] = { 0.3927, 1.2903, 1.8513, 2.7489, 3.6465, 4.3197, 5.2173, 5.7783 }; // by the AT positions)
  const double tecPhiPositions[8]   = { 0.3927, 1.1781, 1.9635, 2.7489, 3.5343, 4.3197, 5.1051, 5.8905 };

  // nominal r values (mm) of hits
  const double tobRPosition = 600.;
  const double tibRPosition = 514.;
  const double tecRPosition[2] = { 564., 840. }; // ring 4,6

  // nominal z values (mm) of hits in barrel (parameter is pos 0-6)
  const double tobZPosition[6] = { 1040., 580., 220., -140., -500., -860. };
  const double tibZPosition[6] = { 620., 380., 180., -100., -340., -540. };
  // nominal z values (mm) of hits in tec (parameter is disk 0-8); FOR TEC-: (* -1.)
  const double tecZPosition[9] = { 1250., 1390., 1530., 1670., 1810., 1985., 2175., 2380., 2595. };
  

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
      //std:: cout << " [LaserAlignment::CalculateNominalCoordinates] -- Filling TEC+ ring: " << ring << ", beam: " << beam << ", disk: " << disk << std::endl; /////////////////////
      //nominalCoordinates.GetTEC2TECEntry( det, ring, beam, disk ).Dump(); /////////////////////////////////
    }
    else { // now TEC-
      nominalCoordinates.SetTECEntry( det, ring, beam, disk, LASCoordinateSet( tecPhiPositions[beam], 0., tecRPosition[ring], 0., -1. * tecZPosition[disk], 0. ) ); // just * -1.
      //std:: cout << " [LaserAlignment::CalculateNominalCoordinates] -- Filling TEC- ring: " << ring << ", beam: " << beam << ", disk: " << disk << std::endl; /////////////////////
      //nominalCoordinates.GetTECEntry( det, ring, beam, disk ).Dump(); /////////////////////////////////
    }
    
  } while( moduleLoop.TECLoop( det, ring, beam, disk ) );



  // TIB & TOB section
  det = 2; beam = 0; pos = 0;
  do {

    if( det == 2 ) { // this is TIB
      nominalCoordinates.SetTIBTOBEntry( det, beam, pos, LASCoordinateSet( tibPhiPositions[beam], 0., tibRPosition, 0., tibZPosition[pos], 0. ) );
      //std:: cout << " [LASBarrelAlgorithm::CalculateNominalCoordinates] -- Filling TIB beam: " << beam << ", pos: " << pos << std::endl; /////////////////////////////////
      //nominalCoordinates.GetTIBTOBEntry( det, beam, pos ).Dump(); /////////////////////////////////
    }
    else { // now TOB
      nominalCoordinates.SetTIBTOBEntry( det, beam, pos, LASCoordinateSet( tobPhiPositions[beam], 0., tobRPosition, 0., tobZPosition[pos], 0. ) );
      //std:: cout << " [LASBarrelAlgorithm::CalculateNominalCoordinates] -- Filling TOB beam: " << beam << ", pos: " << pos << std::endl; /////////////////////////////////
      //nominalCoordinates.GetTIBTOBEntry( det, beam, pos ).Dump(); /////////////////////////////////
    }

  } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );




  // TEC2TEC AT section
  det = 0; beam = 0; disk = 0;
  do {
    
    if( det == 0 ) { // this is TEC+, ring4 only
      nominalCoordinates.SetTEC2TECEntry( det, beam, disk, LASCoordinateSet( tecATPhiPositions[beam], 0., tecRPosition[0], 0., tecZPosition[disk], 0. ) );
      //std:: cout << " [LASBarrelAlgorithm::CalculateNominalCoordinates] -- Filling TEC+ beam: " << beam << ", disk: " << disk << std::endl; /////////////////////////////////
      //nominalCoordinates.GetTEC2TECEntry( det, beam, disk ).Dump(); /////////////////////////////////
    }
    else { // now TEC-
      nominalCoordinates.SetTEC2TECEntry( det, beam, disk, LASCoordinateSet( tecATPhiPositions[beam], 0., tecRPosition[0], 0., -1. * tecZPosition[disk], 0. ) ); // just * -1.
      //std:: cout << " [LASBarrelAlgorithm::CalculateNominalCoordinates] -- Filling TEC- beam: " << beam << ", disk: " << disk << std::endl; /////////////////////////////////
      //nominalCoordinates.GetTEC2TECEntry( det, beam, disk ).Dump(); /////////////////////////////////
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



// define the SEAL module
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(LaserAlignment);
