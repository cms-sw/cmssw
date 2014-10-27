/*! \class DTPlusTrackProducer
 *  \author Ignazio Lazzizzera
 *  \author Sara Vanini
 *  \author Nicola Pozzobon
 *  \brief EDProducer of L1 DT + Track Trigger for the HL-LHC
 *  \date 2008, Dec 25
 */

#include "L1Trigger/DTPlusTrackTrigger/plugins/DTPlusTrackProducer.h"

/// Constructor
DTPlusTrackProducer::DTPlusTrackProducer( const edm::ParameterSet& pSet ) :
  pSetDT(pSet)
{
  TTStubsInputTag = pSet.getParameter< edm::InputTag >( "TTStubs" );
  TTTracksInputTag = pSet.getParameter< edm::InputTag >( "TTTracks" );

  produces< BtiTrigsCollection >();
  produces< TSPhiTrigsCollection >();
  produces< DTMatchesCollection >();

  /// Get bool flags from the cfg file
  useTSTheta    = pSet.getUntrackedParameter< bool >( "useTSTheta", false );
  useRoughTheta = pSet.getUntrackedParameter< bool >( "useRoughTheta", false );

  /// Get the size of matching windows in terms of sigmas
  numSigmasStub = pSet.getUntrackedParameter< double >( "numSigmasForStubMatch", 4. );
  numSigmasTk = pSet.getUntrackedParameter< double >( "numSigmasForTkMatch", 3. );
  numSigmasPt = pSet.getUntrackedParameter< double >( "numSigmasForPtMatch", 3. );

  /// Minimum Pt of L1 Tracks for matching
  minL1TrackPt = pSet.getUntrackedParameter< double >( "minL1TrackPt", 2. );

  /// Get some constraints for finding the Pt with several methods
  minRInvB = pSet.getUntrackedParameter< double >( "minRInvB", 0.00000045 );
  maxRInvB = pSet.getUntrackedParameter< double >( "maxRInvB", 1.0 );
  station2Correction = pSet.getUntrackedParameter< double >( "station2Correction", 1.0 );
  thirdMethodAccurate = pSet.getUntrackedParameter< bool >( "thirdMethodAccurate", false );
}

/// Destructor
DTPlusTrackProducer::~DTPlusTrackProducer(){}

/// Begin job
void DTPlusTrackProducer::beginJob(){}

/// Begin run
void DTPlusTrackProducer::beginRun( const edm::Run& run, const edm::EventSetup& eventSetup )
{
  /// Prepare the DT Trigger to be used and initialize it
  theDTTrigger = new DTTrig(pSetDT);
  theDTTrigger->createTUs(eventSetup);
}

/// Implement the producer
void DTPlusTrackProducer::produce( edm::Event& event, const edm::EventSetup& eventSetup )
{
  /// Prepare the output
  outputBtiTrigs = new BtiTrigsCollection();
  outputTSPhiTrigs = new TSPhiTrigsCollection();
  outputTSThetaTrigs = new TSThetaTrigsCollection();
  outputDTMatches = new DTMatchesCollection();
  tempDTMatchContainer = new std::map< unsigned int, std::vector< DTMatch* > >();
  for ( unsigned int iStation = 1; iStation <= 2; iStation++ )
  {
    std::vector< DTMatch* > tempVector;
    tempVector.clear();
    tempDTMatchContainer->insert( std::make_pair( iStation, tempVector ) );
  }

  /// Get the Stacked Tracker Geometry (Pt modules container)
  edm::ESHandle< StackedTrackerGeometry > theStackedTrackerGeomHandle;
  eventSetup.get< StackedTrackerGeometryRecord >().get( theStackedTrackerGeomHandle );
  const StackedTrackerGeometry* theStackedTracker = theStackedTrackerGeomHandle.product();

  /// Get the muon geometry: needed for the TStheta trigger
  edm::ESHandle< DTGeometry > theMuonDTGeometryHandle;
  eventSetup.get< MuonGeometryRecord >().get( theMuonDTGeometryHandle );

/*
  /// Get the magnetic field
  const MagneticField* theMagneticField = 0;
  edm::ESHandle< MagneticField > theMagFieldHandle;
  eventSetup.get< IdealMagneticFieldRecord >().get( theMagFieldHandle );
  if ( theMagFieldHandle.isValid() )
  {
    theMagneticField = &(*theMagFieldHandle);
std::cerr<<theMagneticField<<std::endl;
  }
  else
  {
    std::cout << "W A R N I N G! Unable to get valid Magnetic Field Handle" << std::endl;
    return;
  }
*/

  /// Get the TTStub container
  edm::Handle< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > > TTStubHandle;
  event.getByLabel( TTStubsInputTag, TTStubHandle );

  /// Get the TTTrack container
  edm::Handle< std::vector< TTTrack< Ref_PixelDigi_ > > > TTTrackHandle;
  event.getByLabel( TTTracksInputTag, TTTrackHandle );

  /// Check if the DT Trigger is correcly created
  if ( !theDTTrigger )
  {
    std::cerr << "E R R O R! Invalid L1 DT reconstruction ..." << std::endl;
    std::cerr << "           exiting!" << std::endl;
    return;
  }

  /// Run the DT Trigger workflow
  theDTTrigger->triggerReco( event, eventSetup );

  /// Prepare the Utilities
  DTUtilities* theseUtilities = new DTUtilities( theDTTrigger,
                                                 outputBtiTrigs,
                                                 outputTSPhiTrigs,
                                                 outputTSThetaTrigs,
                                                 useTSTheta, useRoughTheta,
                                                 theMuonDTGeometryHandle,
                                                 tempDTMatchContainer );

  /// Get the DTTrigger
  theseUtilities->getDTTrigger();

#ifdef npDEBUG
  std::cerr << std::endl;
  std::cerr << "*********************************" << std::endl;
  std::cerr << "* DT TRIGGER IS BUILT           *" << std::endl;
  std::cerr << "*********************************" << std::endl;
  std::cerr << "* how many BTI's?           " << outputBtiTrigs->size() << std::endl;
  std::cerr << "* how many TSPhi's?         " << outputTSPhiTrigs->size() << std::endl;
  std::cerr << "* how many TSTheta's?       " << outputTSThetaTrigs->size() << std::endl;
  std::cerr << "* how many MB1 DTMatches? " << tempDTMatchContainer->at(1).size() << std::endl;
  std::cerr << "* how many MB2 DTMatches? " << tempDTMatchContainer->at(2).size() << std::endl;
  std::cerr << "*********************************" << std::endl;
#endif

  /// Next, order DT triggers in ascending order
  /// (the lower the best) by: 
  /// 1. higher code
  /// 2. lower phib
  theseUtilities->orderDTTriggers();

#ifdef npDEBUG
  std::cerr << std::endl;  
  std::cerr << "*********************************" << std::endl;
  std::cerr << "* DT TRIGGERS ARE ORDERED       *" << std::endl;
  std::cerr << "*********************************" << std::endl;

  for ( unsigned int i = 1; i <= 2; i++ )
  {
    for ( unsigned int j = 0; j < tempDTMatchContainer->at(i).size(); j++ )
    {
      DTMatch* abc = tempDTMatchContainer->at(i).at(j);
      std::cerr << "* DT Match in MB" << i << ", no. " << j << std::endl;
      std::cerr << "                    wh. " << abc->getDTWheel() << " sec. " << abc->getDTSector() << std::endl;
      std::cerr << "                   code " << abc->getDTCode() << " phiB " << abc->getDTTSPhiB() << std::endl;
      std::cerr << "                  order " << abc->getDTTTrigOrder() << std::endl;
     }
  }
#endif

  /// Extrapolate each DTMatch to each Tracker layer
  /// NOTE: this must be done before the "remove redundancies" step
  theseUtilities->extrapolateDTTriggers();

#ifdef npDEBUG
  std::cerr << std::endl;  
  std::cerr << "*********************************" << std::endl;
  std::cerr << "* DT TRIGGERS ARE EXTRAPOLATED  *" << std::endl;
  std::cerr << "*********************************" << std::endl;

  for ( unsigned int i = 1; i <= 2; i++ )
  {
    for ( unsigned int j = 0; j < tempDTMatchContainer->at(i).size(); j++ )
    {
      DTMatch* abc = tempDTMatchContainer->at(i).at(j);
      std::cerr << "* DT Match in MB" << i << ", no. " << j << std::endl;
      std::cerr << "                    phi " << abc->getPredVtxPhi() << " +/- " << abc->getPredVtxSigmaPhi() << std::endl;
      std::cerr << "                  theta " << abc->getPredVtxTheta() << " +/- " << abc->getPredVtxSigmaTheta() << std::endl;
      for ( unsigned int lay = 1; lay <= 6; lay++ )
      {
        std::cerr << "   projection to layer " << lay << std::endl;
        std::cerr << "                    phi " << abc->getPredStubPhi(lay) << " +/- " << abc->getPredStubSigmaPhi(lay) << std::endl;
        std::cerr << "                  theta " << abc->getPredStubTheta(lay) << " +/- " << abc->getPredStubSigmaTheta(lay) << std::endl;
      }
      std::cerr << "   flag reject BEFORE removing redundancies "<<abc->getRejectionFlag()<<std::endl;
    }
  }
#endif

  /// Get rid of redundancies (in fact set a rejection flag)
  theseUtilities->removeRedundantDTTriggers();

#ifdef npDEBUG
  std::cerr << std::endl;
  std::cerr << "*********************************" << std::endl;
  std::cerr << "* REDUNDANT DT TRIGGERS FLAGGED *" << std::endl;
  std::cerr << "*********************************" << std::endl;

  for ( unsigned int i = 1; i <= 2; i++ )
  {
    for ( unsigned int j = 0; j < tempDTMatchContainer->at(i).size(); j++ )
    {
      DTMatch* abc = tempDTMatchContainer->at(i).at(j);
      std::cerr << "* DT Match in MB" << i << ", no. " << j << std::endl;
      std::cerr << "   flag reject AFTER removing redundancies "<<abc->getRejectionFlag()<<std::endl;
    }
  }
#endif

  /// Prepare the TTStub map
  std::map< unsigned int, std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_> >, TTStub< Ref_PixelDigi_ > > > > mapStubByLayer;
  mapStubByLayer.clear();

  /// Loop over the container of stubs (edmNew::DetSetVector)
  edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >::const_iterator iterDSV;
  edmNew::DetSet< TTStub< Ref_PixelDigi_ > >::const_iterator iterTTStub;
  for ( iterDSV = TTStubHandle->begin();
        iterDSV != TTStubHandle->end();
        ++iterDSV )
  {
    /// Get the DetId of this Pt-module
    DetId thisStackedDetId = iterDSV->id();
    StackedTrackerDetId tkDetId = StackedTrackerDetId( thisStackedDetId );

    /// Skip if we are looking at Endcap stubs
    if ( tkDetId.isEndcap() )
    {
      continue;
    }

    /// Here we are supposed to be in the Barrel only
    int iLayer = tkDetId.iLayer();

    /// Prepare the map
    if ( mapStubByLayer.find( iLayer ) == mapStubByLayer.end() )
    {
      std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_> >, TTStub< Ref_PixelDigi_ > > > tempVector;
      tempVector.clear();
      mapStubByLayer.insert( std::make_pair( iLayer, tempVector ) );
    }

    /// Get the stubs
    edmNew::DetSet< TTStub< Ref_PixelDigi_ > > theStubs = (*TTStubHandle)[ thisStackedDetId ];

    /// Loop over the stubs of this Pt-module
    for ( iterTTStub = theStubs.begin();
          iterTTStub != theStubs.end();
          ++iterTTStub )
    {
      /// Safety and consistency check
      if ( tkDetId != iterTTStub->getDetId() )
      {
        std::cerr << " E R R O R!!! module and object DetId are different!!!" << std::endl;
        continue;
      }

      /// Make the reference to be put in the map
      edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > tempStubRef = edmNew::makeRefTo( TTStubHandle, iterTTStub );

      /// Put it in the map
      mapStubByLayer.find( iLayer )->second.push_back( tempStubRef );
    }
  }

#ifdef npDEBUG
  std::cerr << std::endl;
  std::cerr << "*********************************" << std::endl;
  std::cerr << "* NOW READY TO DT-TK MATCHING   *" << std::endl;
  std::cerr << "*********************************" << std::endl;
#endif

  /// Now all barrel stubs are stored in mapStubByLayer
  /// Loop over the DTMatches
  for ( unsigned int i = 1; i <= 2; i++ )
  {
    for ( unsigned int j = 0; j < tempDTMatchContainer->at(i).size(); j++ )
    {
      /// Consider only trigger primitive at right bx
      /// Skip rejected DTMatches
      DTMatch* thisDTMatch = tempDTMatchContainer->at(i).at(j);

      if ( !thisDTMatch->getFlagBXOK() ||
           thisDTMatch->getRejectionFlag() )
      {
        continue;
      }

      /// Here the DTMatch is a good one

      /// Loop over the layers
      for ( unsigned int lay = 1; lay <= 6; lay++ )      
      {
        /// Get all the stubs from this layer
        if ( mapStubByLayer.find( lay ) != mapStubByLayer.end() )
        {
          std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > theStubsFromThisLayer = mapStubByLayer[lay];

          /// Prepare a Ref for the closest stub
          int minDistance = 9999999;
          edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > theClosestStub = theStubsFromThisLayer.at(0);

          /// Loop over the stubs
          for ( unsigned int jStub = 0; jStub < theStubsFromThisLayer.size(); jStub++ )
          {
            edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > tempStub = theStubsFromThisLayer.at(jStub);

            /// Compare the positions
            GlobalPoint stubPos = theStackedTracker->findGlobalPosition( tempStub.get() ); 

            int stubPhi = static_cast< int >( (double)(stubPos.phi()) * 4096. );
            int stubTheta = static_cast< int >( (double)(stubPos.theta()) * 4096. );

            int thisDeltaPhi = thisDTMatch->findStubDeltaPhi( stubPhi, lay );

#ifdef npDEBUG
            std::cerr << "* checking stub match in layer " << lay << std::endl;
            std::cerr << "   stub phi " << stubPos.phi() << std::endl;
            std::cerr << "   converted " << ((double)(stubPos.phi()) * 4096.) << std::endl;
            std::cerr << "   to integer " << stubPhi << std::endl;
            std::cerr << "   predicted phi " << thisDTMatch->getPredStubPhi(lay)
                      << " +/- " << thisDTMatch->getPredStubSigmaPhi(lay) << std::endl;
            std::cerr << "   deltaPhi " << thisDeltaPhi << std::endl;
            std::cerr << "   STUB is it in phi window? " << thisDTMatch->checkStubPhiMatch( stubPhi, lay, numSigmasStub ) << std::endl;
#endif

            /// If within the window, and update the closest stub if any
            if ( thisDTMatch->checkStubPhiMatch( stubPhi, lay, numSigmasStub ) &&
                 thisDTMatch->checkStubThetaMatch( stubTheta, lay, numSigmasStub ) )
            {
#ifdef npDEBUG
              std::cerr << "*** found good match in layer " << lay << std::endl;
#endif

              if ( thisDeltaPhi < minDistance )
              {
#ifdef npDEBUG
                std::cerr << "****** updating good match in layer " << lay << std::endl;
#endif

                minDistance = thisDeltaPhi;
                theClosestStub = theStubsFromThisLayer.at(jStub);

#ifdef npDEBUG
                std::cerr << "********* which is now " << *(theClosestStub.get());
#endif
              }
            }
          } /// End of loop over the stubs

          if ( minDistance < 9999999 )
          {
#ifdef npDEBUG
                std::cerr << "*********** storing the good match in layer " << lay << std::endl;
                std::cerr << theClosestStub->print();
#endif

            /// Add the stub to the match
            thisDTMatch->addMatchedStubRef( theClosestStub,
                                            theStackedTracker->findGlobalPosition( theClosestStub.get() ) );
          }

        } /// End of get all the stubs from this layer
      } /// End of loop over the layers

      /// Then match to the tracks
      /// This is the "classical" outside-inside matching

      /// Get all the TTTracks within matching window
      std::vector< edm::Ptr< TTTrack< Ref_PixelDigi_ > > > allTracksInWindow;

      unsigned int jTrack = 0; /// Counter needed to build the edm::Ptr to the TTTrack
      typename std::vector< TTTrack< Ref_PixelDigi_ > >::const_iterator inputIter;
      for ( inputIter = TTTrackHandle->begin();
            inputIter != TTTrackHandle->end();
            ++inputIter )
      {
        /// Make the pointer to be put in the map
        edm::Ptr< TTTrack< Ref_PixelDigi_ > > tempTrackPtr( TTTrackHandle, jTrack++ );

        /// Redundant Pt threshold
        if ( inputIter->getMomentum().perp() < minL1TrackPt )
        {
          continue;
        }

        /// Additional quality cut
        if ( inputIter->getChi2() >= 100 )
          continue;
        if ( fabs(inputIter->getPOCA().z()) >= 25. )
          continue;
        if ( inputIter->getStubRefs().size() < 3 )
          continue;

        /// Check distance with the TTTrack
        GlobalVector tkMom = tempTrackPtr->getMomentum();

        int tkPhi = static_cast< int >( (double)(tkMom.phi()) * 4096. );
        int tkTheta = static_cast< int >( (double)(tkMom.theta()) * 4096. );

#ifdef npDEBUG
        int thisDeltaPhi = thisDTMatch->findVtxDeltaPhi( tkPhi );

        std::cerr << "* checking track match at vertex" << std::endl;
        std::cerr << "   track phi " << tkMom.phi() << std::endl;
        std::cerr << "   converted " << ((double)(tkMom.phi()) * 4096.) << std::endl;
        std::cerr << "   to integer " << tkPhi << std::endl;
        std::cerr << "   predicted phi " << thisDTMatch->getPredVtxPhi()
                  << " +/- " << thisDTMatch->getPredVtxSigmaPhi() << std::endl;
        std::cerr << "   deltaPhi " << thisDeltaPhi << std::endl;
        std::cerr << "   TRACK is it in phi window? " << thisDTMatch->checkVtxPhiMatch( tkPhi, numSigmasTk ) << std::endl;
        std::cerr << "   track theta converted to integer " << tkTheta << std::endl;
        std::cerr << "   predicted theta " << thisDTMatch->getPredVtxTheta()
                  << " +/- " << thisDTMatch->getPredVtxSigmaTheta() << std::endl;
        std::cerr << "   TRACK is it in theta window? " << thisDTMatch->checkVtxThetaMatch( tkTheta, numSigmasTk ) << std::endl;
#endif

        /// If within the window, and update the closest stub if any
        if ( thisDTMatch->checkVtxPhiMatch( tkPhi, numSigmasTk ) &&
             thisDTMatch->checkVtxThetaMatch( tkTheta, numSigmasTk ) )
        {
#ifdef npDEBUG
          std::cerr << "*** found good match at vertex" << std::endl;
#endif

          /// Add the track to the possible matches
          thisDTMatch->addInWindowTrackPtr( tempTrackPtr );
        }
      }

#ifdef npDEBUG
      std::cerr << std::endl;
      std::cerr << "*********************************" << std::endl;
      std::cerr << "* COUNTING DT-TO-TK MATCHES     *" << std::endl;
      std::cerr << "*********************************" << std::endl;

      DTMatch* abc = tempDTMatchContainer->at(i).at(j);
      std::cerr << "* flag reject cross check " << abc->getRejectionFlag() << std::endl;
      if ( abc->getRejectionFlag() == false )
      {
        std::cerr << "* DT Match in MB" << i << ", no. " << j << std::endl;
        std::cerr << "   for DT at phi " << abc->getDTTSPhi() << " and theta " << abc->getDTTSTheta() << j << std::endl;
        std::cerr << "   matched stubs " << abc->getMatchedStubRefs().size() << std::endl;

        std::map< unsigned int, edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > thisStubMap
          = abc->getMatchedStubRefs();
        std::map< unsigned int, edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > >::iterator stubMapIter;

        unsigned int kStub = 0;
        for ( stubMapIter = thisStubMap.begin();
              stubMapIter != thisStubMap.end();
              ++stubMapIter )
        {
          std::cerr << "** stub number " << kStub++ << std::endl;
          const TTStub< Ref_PixelDigi_ >* thisStub = stubMapIter->second.get();
          std::cerr << thisStub->print();
        }

        std::cerr << "   tracks in window " << abc->getInWindowTrackPtrs().size() << std::endl;
        for ( unsigned int kTrack = 0; kTrack < abc->getInWindowTrackPtrs().size(); kTrack++ )
        {
          std::cerr << "** track number " << kTrack << std::endl;
          std::cerr << "   vtx " << abc->getInWindowTrackPtrs().at(kTrack)->getPOCA() << std::endl;
          std::cerr << "   mom " << abc->getInWindowTrackPtrs().at(kTrack)->getMomentum() << std::endl;
        }
      }
      else
      {
        std::cerr << "* wait, there is also a rejected DT object" << std::endl;
      }

      std::cerr << std::endl;
      std::cerr << "*********************************" << std::endl;
      std::cerr << "* FIND CLOSEST TRACK            *" << std::endl;
      std::cerr << "*********************************" << std::endl;
#endif

      /// Check the DT-to-track best Pt match
      /// first, get the muon Pt information
      int dtPt = thisDTMatch->getDTPt();
      int dtPtMin = thisDTMatch->getDTPtMin( numSigmasPt );
      int dtPtMax = thisDTMatch->getDTPtMax( numSigmasPt );

#ifdef npDEBUG
      std::cerr << "* DT Match in MB" << i << ", no. " << j << std::endl;
      std::cerr << "  has this Pt range " << dtPtMin << " .. " << dtPt << " .. " << dtPtMax << " with no. sigmas = " << numSigmasPt << std::endl;
#endif

      /// Now check the best track
      if ( thisDTMatch->getInWindowTrackPtrs().size() > 0 )
      {
        /// Prepare a Ptr for the closest track
        int minPtDifference = 999999;
        edm::Ptr< TTTrack< Ref_PixelDigi_ > > theClosestTrack = thisDTMatch->getInWindowTrackPtrs().at(0);

        /// Loop over the tracks
        for ( unsigned int jTrack = 0; jTrack < thisDTMatch->getInWindowTrackPtrs().size(); jTrack++ )
        {
          edm::Ptr< TTTrack< Ref_PixelDigi_ > > tempTrackPtr = thisDTMatch->getInWindowTrackPtrs().at(jTrack);

          int trackPt = static_cast< int >( tempTrackPtr->getMomentum().perp() );

          /// Check if they match
          if ( trackPt >= dtPtMin &&
               trackPt <= dtPtMax )
          {
#ifdef npDEBUG
              std::cerr << "*** found good Pt match at vertex" << std::endl;
#endif

            int ptDiff = abs( dtPt - trackPt );
            if ( ptDiff < minPtDifference )
            {
#ifdef npDEBUG
                std::cerr << "****** updating good Pt match at vertex" << std::endl;
#endif

              minPtDifference = ptDiff;
              theClosestTrack = tempTrackPtr;

#ifdef npDEBUG
                std::cerr << "********* which is now " << jTrack << std::endl;
                std::cerr << "   vtx " << tempTrackPtr->getPOCA() << std::endl;
                std::cerr << "   mom " << tempTrackPtr->getMomentum() << std::endl;
#endif
            }
          }
        } /// End of loop over tracks

        /// If any, store the matched track
        if ( minPtDifference < 999999 )
        {
          thisDTMatch->setPtMatchedTrackPtr( theClosestTrack );
        }
      }

#ifdef npDEBUG
      if ( abc->getPtMatchedTrackPtr().isNull() == false )
      {
        std::cerr << "************ so, the best one is" << std::endl;
        std::cerr << "   vtx " << abc->getPtMatchedTrackPtr()->getPOCA() << std::endl;
        std::cerr << "   mom " << abc->getPtMatchedTrackPtr()->getMomentum() << std::endl;
      }
      else
      {
        std::cerr << "************ unfortunately, no best track match is available" << std::endl;
      }
#endif

      /// Now, set all the possible Pt information
      thisDTMatch->setPtMethods( station2Correction, thirdMethodAccurate,
                                 minRInvB, maxRInvB );

      /// Find the priority and the average Pt's according to the specific encoding
      /// defined within the data format class
      thisDTMatch->findPtPriority();
      thisDTMatch->findPtAverage();

      /// Then, assign all the possible Pt bins
      thisDTMatch->findPtPriorityBin();
      thisDTMatch->findPtAverageBin();
      thisDTMatch->findPtTTTrackBin();
      thisDTMatch->findPtMajorityFullTkBin();
      thisDTMatch->findPtMajorityBin();
      thisDTMatch->findPtMixedModeBin();

#ifdef npDEBUG
      std::cerr << std::endl;
      std::cerr << "*********************************" << std::endl;
      std::cerr << "* DT-TK PT PRINTOUT             *" << std::endl;
      std::cerr << "*********************************" << std::endl;

      std::cerr << "* DT Match in MB" << i << ", no. " << j << std::endl;
      std::cerr << "   priority Pt bin:  " << abc->getPtPriorityBin() << " GeV (" << abc->getPtPriority() << " GeV)" << std::endl;
      std::cerr << "   average Pt bin:   " << abc->getPtAverageBin() << " GeV (" << abc->getPtAverage() << " GeV)" << std::endl;
      std::cerr << "   TTTrack Pt bin:   " << abc->getPtTTTrackBin() << " GeV";
      if ( abc->getPtMatchedTrackPtr().isNull() == false )
        std::cerr << " (" << abc->getPtMatchedTrackPtr()->getMomentum().perp() << " GeV)" << std::endl;
      else
        std::cerr << std::endl;
      std::cerr << "   full maj. Pt bin: " << abc->getPtMajorityFullTkBin() << " GeV" << std::endl;
      std::cerr << "   majority Pt bin:  " << abc->getPtMajorityBin() << " GeV" << std::endl;
      std::cerr << "   mixedmode Pt bin: " << abc->getPtMixedModeBin() << " GeV" << std::endl;

      std::map< std::string, DTMatchPt* > abcMap = abc->getPtMethodsMap();
      std::map< std::string, DTMatchPt* >::const_iterator abcMapIter;

      for ( abcMapIter = abcMap.begin();
            abcMapIter != abcMap.end();
            ++abcMapIter )
      {
        std::cerr << "     >> " << abcMapIter->first << " Pt: " << abcMapIter->second->getPt() << " GeV" << std::endl;
      }

      std::cerr << std::endl;
#endif

      outputDTMatches->push_back( *thisDTMatch );

    } /// End of loop over DT matches
  }

  event.put( std::auto_ptr< BtiTrigsCollection >( outputBtiTrigs ) );
  event.put( std::auto_ptr< TSPhiTrigsCollection >( outputTSPhiTrigs ) );
  event.put( std::auto_ptr< DTMatchesCollection >( outputDTMatches ) );

  return;
}

/// End job
void DTPlusTrackProducer::endJob(){}

