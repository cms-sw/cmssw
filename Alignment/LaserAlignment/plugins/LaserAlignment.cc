/** \file LaserAlignment.cc
 *  LAS reconstruction module
 *
 *  $Date: Sun Mar 18 19:38:57 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/plugins/LaserAlignment.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "DataFormats/LaserAlignment/interface/LASBeamProfileFit.h"
#include "DataFormats/LaserAlignment/interface/LASBeamProfileFitCollection.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"


LaserAlignment::LaserAlignment(edm::ParameterSet const& theConf) 
  : theEvents(0), 
    theStoreToDB(theConf.getUntrackedParameter<bool>("saveToDbase", false)),
    theSaveHistograms(theConf.getUntrackedParameter<bool>("saveHistograms",false)),
    theDebugLevel(theConf.getUntrackedParameter<int>("DebugLevel",0)),
    theNEventsPerLaserIntensity(theConf.getUntrackedParameter<int>("NumberOfEventsPerLaserIntensity",100)),
    theNEventsForAllIntensities(theConf.getUntrackedParameter<int>("NumberOfEventsForAllIntensities",100)),
    theDoAlignmentAfterNEvents(theConf.getUntrackedParameter<int>("DoAlignmentAfterNEvents",1000)),
    theAlignPosTEC(theConf.getUntrackedParameter<bool>("AlignPosTEC",false)),
    theAlignNegTEC(theConf.getUntrackedParameter<bool>("AlignNegTEC",false)), 
    theAlignTEC2TEC(theConf.getUntrackedParameter<bool>("AlignTECTIBTOBTEC",false)),
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
    theDigiStore(),
    theBeamProfileFitStore(),
    theDigiVector(),
    theAlignableTracker(),
	  theAlignRecordName( "TrackerAlignmentRcd" ),
	  theErrorRecordName( "TrackerAlignmentErrorRcd" )
{
  // load the configuration from the ParameterSet  
  edm::LogInfo("LaserAlignment") << "==========================================================="
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

  // the beam profile fitter
  theBeamFitter = new BeamProfileFitter(theBeamFitPS);

  // the alignable tracker parts
  theLASAlignPosTEC = new LaserAlignmentPosTEC;

  theLASAlignNegTEC = new LaserAlignmentNegTEC;

  theLASAlignTEC2TEC = new LaserAlignmentTEC2TEC;
  
  // counter for the number of iterations, i.e. the number of BeamProfile fits and
  // local Millepede fits
  theNumberOfIterations = 0;
}

LaserAlignment::~LaserAlignment()
{
  if (theSaveHistograms)
    {
      // close the rootfile
      closeRootFile();
    }

  if (theFile != 0) { delete theFile; }

  if (theBeamFitter != 0) { delete theBeamFitter; }

  if (theLASAlignPosTEC != 0) { delete theLASAlignPosTEC; }
  if (theLASAlignNegTEC != 0) { delete theLASAlignNegTEC; }
  if (theLASAlignTEC2TEC != 0) { delete theLASAlignTEC2TEC; }
  if (theAlignableTracker != 0) { delete theAlignableTracker; }
}

double LaserAlignment::angle(double theAngle)
{
  return (theAngle >= 0.0) ? theAngle : theAngle + 2.0*M_PI;
}

void LaserAlignment::beginJob(const edm::EventSetup& theSetup)
{
  // creating a new file
  theFile = new TFile(theFileName.c_str(),"RECREATE","CMS ROOT file");
  theFile->SetCompressionLevel(theCompression);
      
  // initialize the histograms
  if (theFile) 
    {
      this->initHistograms();
    }
  else 
    {
      throw cms::Exception("LaserAlignment") << "<LaserAlignment::beginJob()>: ERROR!!! something wrong with the RootFile" << std::endl;
    } 

  LogDebug("LaserAlignment:beginJob()") << " access the Tracker Geometry ";
  // access the tracker
  theSetup.get<TrackerDigiGeometryRecord>().get( theTrackerGeometry );
  theSetup.get<IdealGeometryRecord>().get( gD );

  // Create the alignable hierarchy
  LogDebug("LaserAlignment:beginJob()") << " create the alignable hierarchy ";
  theAlignableTracker = new AlignableTracker( &(*gD),
					      &(*theTrackerGeometry) );

}

void LaserAlignment::produce(edm::Event& theEvent, edm::EventSetup const& theSetup) 
{
  LogDebug("LaserAlignment") << "==========================================================="
			      << "\n   Private analysis of event #"<< theEvent.id().event() 
			      << " in run #" << theEvent.id().run();
  
  // ----- check if the actual event can be used -----
  /* here we can later on add some criteria for good alignment events!? */
  theEvents++;
  
  LogDebug("LaserAlignment") << "     Total Event number = " << theEvents;
  
  
  // create an empty output collection
  std::auto_ptr<LASBeamProfileFitCollection> theFitOutput(new LASBeamProfileFitCollection);
  
  theDigiVector.reserve(10000);
  theDigiVector.clear();
  
  // do the Tracker Statistics
  trackerStatistics(theEvent, theSetup);
  
  LogDebug("LaserAlignment") << "===========================================================";
  
  // the fit of the beam profiles
  if (theEvents % theNEventsPerLaserIntensity == 0)
    {
      // do the beam profile fit
      fit(theSetup);
    }

  if (theEvents % theNEventsForAllIntensities == 0)
    {
      // increase the counter for the iterations
      theNumberOfIterations++;
      
      // put the digis of the beams into the StripDigiCollection
      for (std::map<DetId, std::vector<SiStripDigi> >::const_iterator p = theDigiStore.begin();
	   p != theDigiStore.end(); ++p)
	{
	  edm::DetSet<SiStripDigi> collector((p->first).rawId());
	  
	  if ((p->second).size()>0)
	    {
	      collector.data = (p->second);
	      
	      theDigiVector.push_back(collector);
	    }
	}
      // clear the map to fill new digis for the next theNEventsForAllIntensities number of events
      theDigiStore.clear();

      // put the LASBeamProfileFits into the LASBeamProfileFitCollection
      // loop over the map with the histograms and lookup the LASBeamFits
      for (std::vector<std::string>::const_iterator iHistName = theHistogramNames.begin(); iHistName != theHistogramNames.end(); ++iHistName)
	{
	  std::map<std::string, std::pair<DetId, TH1D*> >::iterator iHist = theHistograms.find(*iHistName);
	  if ( iHist != theHistograms.end() )
	    {
	      std::map<std::string, std::vector<LASBeamProfileFit> >::iterator iBeamFit = theBeamProfileFitStore.find(*iHistName);
	      if ( iBeamFit != theBeamProfileFitStore.end() )
		{
		  // get the DetId from the map with histograms
		  unsigned int theDetId = ((iHist->second).first).rawId();
		  // get the information for the LASBeamProfileFitCollection
		  LASBeamProfileFitCollection::Range inputRange;
		  inputRange.first = (iBeamFit->second).begin();
		  inputRange.second = (iBeamFit->second).end();
	      
		  theFitOutput->put(inputRange,theDetId);

		  // now fill the fitted phi position and error into the appropriate vectors
		  // which will be used by the Alignment Algorithm
		  LASBeamProfileFit theFit = (iBeamFit->second).at(0);
		  theLaserPhi.push_back(theFit.phi());
		  theLaserPhiError.push_back(thePhiErrorScalingFactor * theFit.phiError());
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
    }
  
  // do the alignment of the tracker
  if (theEvents % theDoAlignmentAfterNEvents == 0)
    {

//       // Create the alignable hierarchy
//       theAlignableTracker = new AlignableTracker( &(*gD),
// 						  &(*theTrackerGeometry) );

      // do the Alignment of the Tracker (with Millepede) ...
      alignmentAlgorithm(theAlignmentAlgorithmPS, theAlignableTracker);

      // set the number of iterations to zero for the next alignment round
      theNumberOfIterations = 0;

      // store the estimated alignment parameters into the DB
      // first get them
      Alignments* alignments =  theAlignableTracker->alignments();
      AlignmentErrors* alignmentErrors = theAlignableTracker->alignmentErrors();
  
      // Write alignments to DB: have to sort beforhand!
      if ( theStoreToDB )
        {
	  LogDebug("LaserAlignment") << " storing the calculated alignment parameters to the DataBase";
	      // Call service
	      edm::Service<cond::service::PoolDBOutputService> poolDbService;
	      if( !poolDbService.isAvailable() ) // Die if not available
	        throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";

		  // Store
	      if ( poolDbService->isNewTagRequest(theAlignRecordName) )
	        poolDbService->createNewIOV<Alignments>( alignments, poolDbService->endOfTime(), 
	                                                 theAlignRecordName );
	      else
	        poolDbService->appendSinceTime<Alignments>( alignments, poolDbService->currentTime(), 
	                                                   theAlignRecordName );
	      if ( poolDbService->isNewTagRequest(theErrorRecordName) )
	        poolDbService->createNewIOV<AlignmentErrors>( alignmentErrors,
	                                                      poolDbService->endOfTime(), 
	                                                      theErrorRecordName );
	      else
	        poolDbService->appendSinceTime<AlignmentErrors>( alignmentErrors,
	                                                         poolDbService->currentTime(), 
	                                                         theErrorRecordName );
	}

//       // Store result to EventSetup
//       GeometryAligner aligner;
//       aligner.applyAlignments<TrackerGeometry>( &(*theTrackerGeometry), &(*alignments), &(*alignmentErrors) );
    }
  
  // create the output collection for the DetSetVector
  std::auto_ptr<edm::DetSetVector<SiStripDigi> > theDigiOutput(new edm::DetSetVector<SiStripDigi>(theDigiVector));

  // write output to file
  theEvent.put(theDigiOutput);
  theEvent.put(theFitOutput);
  
//   // clear the vector with pairs of DetIds and Histograms
//   theHistograms.clear();
}


void LaserAlignment::closeRootFile()
{
  theFile->Write();
}

void LaserAlignment::fillAdcCounts(TH1D * theHistogram, DetId theDetId,
				    edm::DetSet<SiStripDigi>::const_iterator digiRangeIterator,
				    edm::DetSet<SiStripDigi>::const_iterator digiRangeIteratorEnd)
{
  if (theDebugLevel > 4) std::cout << "<LaserAlignment::fillAdcCounts()>: DetUnit: " << theDetId.rawId() << std::endl;

  // loop over all the digis in this det
  for (; digiRangeIterator != digiRangeIteratorEnd; ++digiRangeIterator) 
    {
      const SiStripDigi *digi = &*digiRangeIterator;

      // store the digis from the laser beams. They are later on used to create
      // clusters and RecHits. In this way some sort of "Laser Tracks" can be
      // reconstructed, which are useable for Track Based Alignment      
      theDigiStore[theDetId].push_back((*digi));
      
      if ( theDebugLevel > 5 ) 
	{ std::cout << " Channel " << digi->channel() << " has " << digi->adc() << " adc counts " << std::endl; }

      // fill the number of adc counts in the histogram
      if (digi->channel() < 512)
	{
	  Double_t theBinContent = theHistogram->GetBinContent(digi->channel()) + digi->adc();
	  theHistogram->SetBinContent(digi->channel(), theBinContent);
	}
    }
}
// define the SEAL module
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(LaserAlignment);
