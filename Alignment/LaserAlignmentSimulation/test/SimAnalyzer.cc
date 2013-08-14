/** \file SimAnalyzer.cc
 *  Get some statistics and plots about the simulation of the Laser Alignment System
 *
 *  $Date: 2011/09/16 06:49:08 $
 *  $Revision: 1.7 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignmentSimulation/test/SimAnalyzer.h"
#include "FWCore/Framework/interface/Event.h" 
#include "FWCore/Framework/interface/ESHandle.h" 
#include "FWCore/ParameterSet/interface/ParameterSet.h" 
#include "FWCore/Framework/interface/EventSetup.h" 
#include "FWCore/Utilities/interface/EDMException.h" 
#include "FWCore/MessageLogger/interface/MessageLogger.h" 
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h" 
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h" 
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h" 
#include "DataFormats/DetId/interface/DetId.h" 
#include "SimDataFormats/TrackingHit/interface/PSimHit.h" 
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h" 
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h" 
#include "TFile.h" 

	SimAnalyzer::SimAnalyzer(edm::ParameterSet const& theConf) 
	: theEvents(0), 
	theDebugLevel(theConf.getUntrackedParameter<int>("DebugLevel",0)),
	theSearchPhiTIB(theConf.getUntrackedParameter<double>("SearchWindowPhiTIB",0.05)),
	theSearchPhiTOB(theConf.getUntrackedParameter<double>("SearchWindowPhiTOB",0.05)),
	theSearchPhiTEC(theConf.getUntrackedParameter<double>("SearchWindowPhiTEC",0.05)),
	theSearchZTIB(theConf.getUntrackedParameter<double>("SearchWindowZTIB",1.0)),
	theSearchZTOB(theConf.getUntrackedParameter<double>("SearchWindowZTOB",1.0)),
	theFile(),
	theCompression(theConf.getUntrackedParameter<int>("ROOTFileCompression",1)),
	theFileName(theConf.getUntrackedParameter<std::string>("ROOTFileName","test.root")),
	theBarrelSimHitsX(0),
	theBarrelSimHitsY(0),
	theBarrelSimHitsZ(0),
	theBarrelSimHitsYvsX(0),
	theBarrelSimHitsXvsZ(0),
	theBarrelSimHitsYvsZ(0),
	theBarrelSimHitsRvsZ(0),
	theBarrelSimHitsPhivsX(0),
	theBarrelSimHitsPhivsY(0),
	theBarrelSimHitsPhivsZ(0),
	// the histograms for Endcap Hits
	theEndcapSimHitsX(0),
	theEndcapSimHitsY(0),
	theEndcapSimHitsZ(0),
	theEndcapSimHitsYvsX(0),
	theEndcapSimHitsXvsZ(0),
	theEndcapSimHitsYvsZ(0),
	theEndcapSimHitsRvsZ(0),
	theEndcapSimHitsPhivsX(0),
	theEndcapSimHitsPhivsY(0),
	theEndcapSimHitsPhivsZ(0),
	// the histograms for all SimHits
	theSimHitsRvsZ(0),
	theSimHitsPhivsZ(0)  
{
	// load the configuration from the ParameterSet  
	edm::LogInfo("SimAnalyzer") << "==========================================================="  
		<< "===                Start configuration                  ==="
		<< "    theDebugLevel                  = " << theDebugLevel
		<< "    theSearchPhiTIB                = " << theSearchPhiTIB
		<< "    theSearchPhiTOB                = " << theSearchPhiTOB 
		<< "    theSearchPhiTEC                = " << theSearchPhiTEC
		<< "    theSearchZTIB                  = " << theSearchZTIB 
		<< "    theSearchZTOB                  = " << theSearchZTOB
		<< "    ROOT filename                  = " << theFileName 
		<< "    compression                    = " << theCompression
		<< "===========================================================";  
}

SimAnalyzer::~SimAnalyzer()
{
	if (theFile != 0) {
	        // close the rootfile
	        closeRootFile();
	  
   	        delete theFile; 
	}

}

void SimAnalyzer::analyze(edm::Event const& theEvent, edm::EventSetup const& theSetup) 
{
	LogDebug("SimAnalyzer") << "==========================================================="
		<< "   Private analysis of event #"<< theEvent.id().event() 
		<< " in run #" << theEvent.id().run();

	// ----- check if the actual event can be used -----
	/* here we can later on add some criteria for good alignment events!? */

	theEvents++;

	LogDebug("SimAnalyzer") << "     Total Event number = " << theEvents;

	// do the Tracker Statistics
	trackerStatistics(theEvent, theSetup);

	LogDebug("SimAnalyzer") << "===========================================================";
}

void SimAnalyzer::beginJob() 
{
	LogDebug("SimAnalyzer") << "==========================================================="
		<< "===                Start beginJob()                     ==="
		<< "     creating a CMS Root Tree ...";
	// creating a new file
	theFile = new TFile(theFileName.c_str(),"RECREATE","CMS ROOT file");

	// initialize the histograms
	if (theFile) 
	{
         	theFile->SetCompressionLevel(theCompression);
		this->initHistograms();
	}
	else 
	{
		throw edm::Exception(edm::errors::LogicError,
			"<SimAnalyzer::initSetup()>: ERROR!!! something wrong with the RootFile");
	} 


	LogDebug("SimAnalyzer") << "===                Done beginJob()                      ==="
		<< "===========================================================";
}

double SimAnalyzer::angle(double theAngle)
{
	return theAngle += (theAngle >= 0.0) ? 0.0 : 2.0 * M_PI;
}

void SimAnalyzer::closeRootFile()
{
	LogDebug("SimAnalyzer") << " writing all information into a file ";

	theFile->Write();
}

void SimAnalyzer::initHistograms()
{
	LogDebug("SimAnalyzer") << "     Start of initialisation monitor histograms ...";

	// the directories in the ROOT file
	TDirectory * SimHitDir = theFile->mkdir("SimHits");
	TDirectory * BarrelDir = SimHitDir->mkdir("Barrel");
	TDirectory * EndcapDir = SimHitDir->mkdir("Endcap");

	// histograms for Barrel Hits
	theBarrelSimHitsX = new TH1D("BarrelSimHitsX","X Position of the SimHits", 200, -100.0, 100.0);
	theBarrelSimHitsX->SetDirectory(BarrelDir);
	theBarrelSimHitsX->Sumw2();

	theBarrelSimHitsY = new TH1D("BarrelSimHitsY","Y Position of the SimHits", 200, -100.0, 100.0);
	theBarrelSimHitsY->SetDirectory(BarrelDir);
	theBarrelSimHitsY->Sumw2();

	theBarrelSimHitsZ = new TH1D("BarrelSimHitsZ","Z Position of the SimHits", 240, -120.0, 120.0);
	theBarrelSimHitsZ->SetDirectory(BarrelDir);
	theBarrelSimHitsZ->Sumw2();

	theBarrelSimHitsYvsX = new TH2D("BarrelSimHitsYvsX","Y vs X Position of the SimHits", 
		200, -100.0, 100.0, 200, -100.0, 100.0);
	theBarrelSimHitsYvsX->SetDirectory(BarrelDir);
	theBarrelSimHitsYvsX->Sumw2();

	theBarrelSimHitsXvsZ = new TH2D("BarrelSimHitsXvsZ","X vs Z Position of the SimHits",
		240, -120.0, 120.0, 200, -100.0, 100.0);
	theBarrelSimHitsXvsZ->SetDirectory(BarrelDir);
	theBarrelSimHitsXvsZ->Sumw2();

	theBarrelSimHitsYvsZ = new TH2D("BarrelSimHitsYvsZ","Y vs Z Position of the SimHits",
		240, -120.0, 120.0, 200, -100.0, 100.0);
	theBarrelSimHitsYvsZ->SetDirectory(BarrelDir);
	theBarrelSimHitsYvsZ->Sumw2();

	theBarrelSimHitsRvsZ = new TH2D("BarrelSimHitsRvsZ","R vs Z Position of the SimHits",
		240, -120.0, 120.0, 130, 0.0, 65.0);
	theBarrelSimHitsRvsZ->SetDirectory(BarrelDir);
	theBarrelSimHitsRvsZ->Sumw2();

	theBarrelSimHitsPhivsX = new TH2D("BarrelSimHitsPhivsX","Phi [rad] vs X Position of the SimHits",
		200, -100.0, 100.0, 70, 0.0, 7.0);
	theBarrelSimHitsPhivsX->SetDirectory(BarrelDir);
	theBarrelSimHitsPhivsX->Sumw2();

	theBarrelSimHitsPhivsY = new TH2D("BarrelSimHitsPhivsY","Phi [rad] vs Y Position of the SimHits",
		200, -100.0, 100.0, 70, 0.0, 7.0);
	theBarrelSimHitsPhivsY->SetDirectory(BarrelDir);
	theBarrelSimHitsPhivsY->Sumw2();

	theBarrelSimHitsPhivsZ = new TH2D("BarrelSimHitsPhivsZ","Phi [rad] vs Z Position of the SimHits",
		240, -120.0, 120.0, 70, 0.0, 7.0);
	theBarrelSimHitsPhivsZ->SetDirectory(BarrelDir);
	theBarrelSimHitsPhivsZ->Sumw2();

	// histograms for Endcap Hits
	theEndcapSimHitsX = new TH1D("EndcapSimHitsX","X Position of the SimHits", 200, -100.0, 100.0);
	theEndcapSimHitsX->SetDirectory(EndcapDir);
	theEndcapSimHitsX->Sumw2();

	theEndcapSimHitsY = new TH1D("EndcapSimHitsY","Y Position of the SimHits", 200, -100.0, 100.0);
	theEndcapSimHitsY->SetDirectory(EndcapDir);
	theEndcapSimHitsY->Sumw2();

	theEndcapSimHitsZ = new TH1D("EndcapSimHitsZ","Z Position of the SimHits", 600, -300.0, 300.0);
	theEndcapSimHitsZ->SetDirectory(EndcapDir);
	theEndcapSimHitsZ->Sumw2();

	theEndcapSimHitsYvsX = new TH2D("EndcapSimHitsYvsX","Y vs X Position of the SimHits", 
		200, -100.0, 100.0, 200, -100.0, 100.0);
	theEndcapSimHitsYvsX->SetDirectory(EndcapDir);
	theEndcapSimHitsYvsX->Sumw2();

	theEndcapSimHitsXvsZ = new TH2D("EndcapSimHitsXvsZ","X vs Z Position of the SimHits",
		600, -300.0, 300.0, 200, -100.0, 100.0);
	theEndcapSimHitsXvsZ->SetDirectory(EndcapDir);
	theEndcapSimHitsXvsZ->Sumw2();

	theEndcapSimHitsYvsZ = new TH2D("EndcapSimHitsYvsZ","Y vs Z Position of the SimHits",
		600, -300.0, 300.0, 200, -100.0, 100.0);
	theEndcapSimHitsYvsZ->SetDirectory(EndcapDir);
	theEndcapSimHitsYvsZ->Sumw2();

	theEndcapSimHitsRvsZ = new TH2D("EndcapSimHitsRvsZ","R vs Z Position of the SimHits",
		600, -300.0, 300.0, 1000, 0.0, 100.0);
	theEndcapSimHitsRvsZ->SetDirectory(EndcapDir);
	theEndcapSimHitsRvsZ->Sumw2();

	theEndcapSimHitsPhivsX = new TH2D("EndcapSimHitsPhivsX","Phi [rad] vs X Positon of the SimHits",
		200, -100.0, 100.0, 70, 0.0, 7.0);
	theEndcapSimHitsPhivsX->SetDirectory(EndcapDir);
	theEndcapSimHitsPhivsX->Sumw2();

	theEndcapSimHitsPhivsY = new TH2D("EndcapSimHitsPhivsY","Phi [rad] vs Y Position of the SimHits",
		200, -100.0, 100.0, 70, 0.0, 7.0);
	theEndcapSimHitsPhivsY->SetDirectory(EndcapDir);
	theEndcapSimHitsPhivsY->Sumw2();

	theEndcapSimHitsPhivsZ = new TH2D("EndcapSimHitsPhivsZ","Phi [rad] vs Z Position of the SimHits",
		600, -300.0, 300.0, 70, 0.0, 7.0);
	theEndcapSimHitsPhivsZ->SetDirectory(EndcapDir);
	theEndcapSimHitsPhivsZ->Sumw2();

	// histograms for all SimHits
	theSimHitsRvsZ = new TH2D("SimHitsRvsZ","R vs Z Position of the SimHits", 600, -300.0, 300.0, 1000, 0.0, 100.0);
	theSimHitsRvsZ->SetDirectory(SimHitDir);
	theSimHitsRvsZ->Sumw2();

	theSimHitsPhivsZ = new TH2D("SimHitsPhivsZ","Phi [rad] vs Z Position of the SimHits", 600, -300.0, 300.0, 700, 0.0, 7.0);
	theSimHitsPhivsZ->SetDirectory(SimHitDir);
	theSimHitsPhivsZ->Sumw2();
}

void SimAnalyzer::trackerStatistics(edm::Event const& theEvent, edm::EventSetup const& theSetup)
{
	LogDebug("SimAnalyzer") << "<SimAnalyzer::trackerStatistics(edm::Event const& theEvent): filling the histograms ... ";

	int theBarrelHits = 0;
	int theEndcapHits = 0;

	// access the tracker
	edm::ESHandle<TrackerGeometry> theTrackerGeometry;
	theSetup.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
	const TrackerGeometry& theTracker(*theTrackerGeometry);

	// the DetUnits
	TrackingGeometry::DetContainer theDetUnits = theTracker.dets();

	// get the SimHitContainers
	std::vector<edm::Handle<edm::PSimHitContainer> > theSimHitContainers;
	theEvent.getManyByType(theSimHitContainers);

	LogDebug("SimAnalyzer") << " Geometry node for TrackingGeometry is  " << &(*theTrackerGeometry)
		<< "\n I have " << theTrackerGeometry->dets().size() << " detectors "
		<< "\n I have " << theTrackerGeometry->detTypes().size() << " types "
		<< "\n theDetUnits has " << theDetUnits.size() << " dets ";

	// theSimHits contains all the sim hits in this event
	std::vector<PSimHit> theSimHits;
	for (int i = 0; i < int(theSimHitContainers.size()); i++)
	{
		theSimHits.insert(theSimHits.end(),theSimHitContainers.at(i)->begin(),theSimHitContainers.at(i)->end());
	}

	// loop over the SimHits and fill the histograms
	for (std::vector<PSimHit>::const_iterator iHit = theSimHits.begin();
	iHit != theSimHits.end(); iHit++)
	{
			// create a DetId from the detUnitId
		DetId theDetUnitId((*iHit).detUnitId());

			// get the DetUnit via the DetUnitId and cast it to a StripGeomDetUnit
		const GeomDet * theDet = theTracker.idToDet(theDetUnitId);

			// the detector part
		std::string thePart = "";

		LogDebug("SimAnalyzer:SimHitInfo") << " ============================================================ "
			<< "\n Some Information for this SimHit: "
			<< "\n" << *iHit 
			<< "\n GlobalPosition = (" << theDet->surface().toGlobal((*iHit).localPosition()).x() << ","
			<< theDet->surface().toGlobal((*iHit).localPosition()).y() << "," 
			<< theDet->surface().toGlobal((*iHit).localPosition()).z() << ") "
			<< "\n R = " << theDet->surface().toGlobal((*iHit).localPosition()).perp()
			<< "\n phi = " << theDet->surface().toGlobal((*iHit).localPosition()).phi()
			<< "\n angle(phi) = " << angle(theDet->surface().toGlobal((*iHit).localPosition()).phi())
			<< "\n GlobalDirection = (" << theDet->surface().toGlobal((*iHit).localDirection()).x() << ","
			<< theDet->surface().toGlobal((*iHit).localDirection()).y() << "," 
			<< theDet->surface().toGlobal((*iHit).localDirection()).z() << ") "
			<< "\n Total momentum = " << (*iHit).pabs()
			<< "\n ParticleType = " << (*iHit).particleType()
			<< "\n detUnitId = " << (*iHit).detUnitId();


			// select hits in Barrel and Endcaps
		switch (theDetUnitId.subdetId())
		{
			case StripSubdetector::TIB:
			{
				thePart = "TIB";
				break;
			}
			case StripSubdetector::TOB:
			{
				thePart = "TOB";
				break;
			}
			case StripSubdetector::TEC:
			{
				thePart = "TEC";
				break;
			}
		}

		if ( (thePart == "TIB") || (thePart == "TOB") )
		{
			theBarrelHits++;

		// histograms for the Barrel
			theBarrelSimHitsX->Fill(theDet->surface().toGlobal((*iHit).localPosition()).x());
			theBarrelSimHitsY->Fill(theDet->surface().toGlobal((*iHit).localPosition()).y());
			theBarrelSimHitsZ->Fill(theDet->surface().toGlobal((*iHit).localPosition()).z());
			theBarrelSimHitsYvsX->Fill(theDet->surface().toGlobal((*iHit).localPosition()).x(),
				theDet->surface().toGlobal((*iHit).localPosition()).y());
			theBarrelSimHitsXvsZ->Fill(theDet->surface().toGlobal((*iHit).localPosition()).z(),
				theDet->surface().toGlobal((*iHit).localPosition()).x());
			theBarrelSimHitsYvsZ->Fill(theDet->surface().toGlobal((*iHit).localPosition()).z(),
				theDet->surface().toGlobal((*iHit).localPosition()).y());
			theBarrelSimHitsRvsZ->Fill(theDet->surface().toGlobal((*iHit).localPosition()).z(),
				theDet->surface().toGlobal((*iHit).localPosition()).perp());
			theBarrelSimHitsPhivsX->Fill(theDet->surface().toGlobal((*iHit).localPosition()).x(),
				angle(theDet->surface().toGlobal((*iHit).localPosition()).phi()));
			theBarrelSimHitsPhivsY->Fill(theDet->surface().toGlobal((*iHit).localPosition()).y(),
				angle(theDet->surface().toGlobal((*iHit).localPosition()).phi()));
			theBarrelSimHitsPhivsZ->Fill(theDet->surface().toGlobal((*iHit).localPosition()).z(),
				angle(theDet->surface().toGlobal((*iHit).localPosition()).phi()));
		}
		else if ( thePart == "TEC" )
		{
			theEndcapHits++;

		// histograms for the Endcaps
			theEndcapSimHitsX->Fill(theDet->surface().toGlobal((*iHit).localPosition()).x());
			theEndcapSimHitsY->Fill(theDet->surface().toGlobal((*iHit).localPosition()).y());
			theEndcapSimHitsZ->Fill(theDet->surface().toGlobal((*iHit).localPosition()).z());
			theEndcapSimHitsYvsX->Fill(theDet->surface().toGlobal((*iHit).localPosition()).x(),
				theDet->surface().toGlobal((*iHit).localPosition()).y());
			theEndcapSimHitsXvsZ->Fill(theDet->surface().toGlobal((*iHit).localPosition()).z(),
				theDet->surface().toGlobal((*iHit).localPosition()).x());
			theEndcapSimHitsYvsZ->Fill(theDet->surface().toGlobal((*iHit).localPosition()).z(),
				theDet->surface().toGlobal((*iHit).localPosition()).y());
			theEndcapSimHitsRvsZ->Fill(theDet->surface().toGlobal((*iHit).localPosition()).z(),
				theDet->surface().toGlobal((*iHit).localPosition()).perp());
			theEndcapSimHitsPhivsX->Fill(theDet->surface().toGlobal((*iHit).localPosition()).x(),
				angle(theDet->surface().toGlobal((*iHit).localPosition()).phi()));
			theEndcapSimHitsPhivsY->Fill(theDet->surface().toGlobal((*iHit).localPosition()).y(),
				angle(theDet->surface().toGlobal((*iHit).localPosition()).phi()));
			theEndcapSimHitsPhivsZ->Fill(theDet->surface().toGlobal((*iHit).localPosition()).z(),
				angle(theDet->surface().toGlobal((*iHit).localPosition()).phi()));
		}

			// histograms for all SimHits (both Barrel and Endcaps)
		if ( (thePart == "TIB") || (thePart == "TOB") || (thePart == "TEC") )
		{
			theSimHitsRvsZ->Fill(theDet->surface().toGlobal((*iHit).localPosition()).z(),
				theDet->surface().toGlobal((*iHit).localPosition()).perp());
			theSimHitsPhivsZ->Fill(theDet->surface().toGlobal((*iHit).localPosition()).z(),
				angle(theDet->surface().toGlobal((*iHit).localPosition()).phi()));
		}


	} // end loop over SimHits

	// some statistics for this event
	edm::LogInfo("SimAnalyzer") << "     number of SimHits = " << theBarrelHits << "/" 
		<< theEndcapHits << " (Barrel/Endcap) ";

}

