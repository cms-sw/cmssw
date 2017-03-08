#ifndef SiPixelPhase1Analyzer__H_
#define SiPixelPhase1Analyzer__H_

/**\class SiPixelPhase1Analyzer SiPixelPhase1Analyzer.cc EJTerm/SiPixelPhase1Analyzer/plugins/SiPixelPhase1Analyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Pawel Jurgielewicz
//         Created:  Tue, 21 Feb 2017 09:42:19 GMT
//
//

// system include files
#include <memory>

// #include <iostream>
#include <fstream>
#include <string>

#include <algorithm> 

#include <vector>
#include <map>


// user include files
#include "DQM/SiPixelPhase1Analyzer/interface/mat4.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h" 

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h" 
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"  
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h" 
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h" 
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "RecoTracker/TkDetLayers/src/DiskSectorBounds.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h" 

#include "TH2.h"
#include "TProfile2D.h"
#include "TH2Poly.h"
#include "TGraph.h"

#define CODE_FORWARD(s, d, b) ((unsigned short)((b << 8) + (d << 4) + s))

// #define DEBUG_MODE

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<> and also remove the line from
// constructor "usesResource("TFileService");"
// This will improve performance in multithreaded jobs.

using namespace std;
using namespace edm;

enum OperationMode {MODE_ANALYZE = 0, MODE_REMAP = 1};

class SiPixelPhase1Analyzer : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
public:
	explicit SiPixelPhase1Analyzer(const edm::ParameterSet&);
	~SiPixelPhase1Analyzer();
	
	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
	
private:
	virtual void beginJob() override;
	virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
	virtual void endJob() override;
	
	void BookHistograms();
	
	void BookBarrelHistograms(TDirectory* currentDir, const string& currentHistoName);
	void BookForwardHistograms(TDirectory* currentDir, const string& currentHistoName);
	
	void BookBins(ESHandle < TrackerGeometry >& theTrackerGeometry, const TrackerTopology* tt);
	void BookBarrelBins(ESHandle < TrackerGeometry >& theTrackerGeometry, const TrackerTopology* tt);
	void BookForwardBins(ESHandle < TrackerGeometry >& theTrackerGeometry, const TrackerTopology* tt);
	
	void SaveDetectorVertices(const TrackerTopology* tt);
	
	void FillBins(edm::Handle<reco::TrackCollection> *tracks, ESHandle < TrackerGeometry >& theTrackerGeometry, const TrackerTopology* tt);
	
	void FillBarrelBinsAnalyze(ESHandle < TrackerGeometry >& theTrackerGeometry, const TrackerTopology* tt, unsigned rawId, const GlobalPoint& globalPoint);
	void FillForwardBinsAnalyze(ESHandle < TrackerGeometry >& theTrackerGeometry, const TrackerTopology* tt, unsigned rawId, const GlobalPoint& globalPoint);
	
	void FillBarrelBinsRemap(ESHandle < TrackerGeometry >& theTrackerGeometry, const TrackerTopology* tt);
	void FillForwardBinsRemap(ESHandle < TrackerGeometry >& theTrackerGeometry, const TrackerTopology* tt);
	
	// ----------member data ---------------------------
	OperationMode opMode;
	
	edm::EDGetTokenT<reco::TrackCollection> tracksToken;
	
	string debugFileName;
	std::ofstream debugFile;  
	
	edm::Service < TFileService > fs;
	
	bool firstEvent;
	
	map<uint32_t, TGraph*> bins, binsSummary;
	
	map< string, vector<TH2Poly*> > th2PolyBarrel;
	map< string, TH2Poly* > th2PolyBarrelSummary;
	
#ifdef DEBUG_MODE
	map< string, vector<TH2*> >th2PolyBarrelDebug;
#endif
	
	map< string, vector<TH2Poly*> > pxfTh2PolyForward;
	map< string, TH2Poly* > pxfTh2PolyForwardSummary;
	
#ifdef DEBUG_MODE
	map< string, vector<TH2*> > pxfTh2PolyForwardDebug;
#endif
	
	mat4 orthoProjectionMatrix;
	
	struct complementaryElements
	{
		mat4 mat[2];
		unsigned rawId[2];
	};
	// used to hold information about elements': ids & matrices which are of the same side, disk and barrel but different panel 
	// to build trapezoidal ring elements
	map<unsigned short, complementaryElements> mapOfComplementaryElements;
	
	//Input root file handle;
	TFile *rootFileHandle;
	
	// read input histograms
	vector<unsigned> isBarrelSource;
	vector<string> analazedRootFileName;
	vector<string> pathToHistograms;
	vector<string> baseHistogramName;
	
	// temporal functionality
	void SaveDetectorData(bool isBarrel, unsigned rawId, int shell_hc, int layer_disk, int ladder_blade)
	{
		static std::ofstream file("det.data", std::ofstream::out);
		
		file << isBarrel << "\t" << rawId << "\t" << shell_hc << "\t" << layer_disk << "\t" << ladder_blade << endl;
	}
};

#endif