#ifndef SiPixelMonitorRecHit_h
#define SiPixelMonitorRecHit_h

/** \class SiPixelMonitorRecHit
 * File: SiPixelMonitorRecHit.h
 * \author Jason Shaev, JHU
 * Created: 6/7/06
 * Modified by Keith Rose
 * July 2007
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DWM histogram services
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//Simhit stuff
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

#include "FWCore/ParameterSet/interface/InputTag.h"

#include <string>

//using namespace std;
//using namespace edm;

class SiPixelMonitorRecHit : public edm::EDAnalyzer {

   public:
	//Constructor
	SiPixelMonitorRecHit(const edm::ParameterSet& conf);

	//Destructor
	~SiPixelMonitorRecHit();

   protected:

	virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
	void beginJob(const edm::EventSetup& c);
	void endJob();

   private:
	DaqMonitorBEInterface* dbe_;
	std::string outputFile_;

	edm::ParameterSet conf_;

	void fillBarrel(const SiPixelRecHit &,const PSimHit &, DetId, const PixelGeomDetUnit *);	
	void fillForward(const SiPixelRecHit &, const PSimHit &, DetId, const PixelGeomDetUnit *);

	//RecHits BPIX
	MonitorElement* recHitXResAllB;
	MonitorElement* recHitYResAllB;
	MonitorElement* recHitXFullModules;
	MonitorElement* recHitXHalfModules;
	MonitorElement* recHitYAllModules;
	MonitorElement* recHitXResFlippedLadderLayers[3];
	MonitorElement* recHitXResNonFlippedLadderLayers[3];
	MonitorElement* recHitYResLayer1Modules[8];
	MonitorElement* recHitYResLayer2Modules[8];
	MonitorElement* recHitYResLayer3Modules[8];
	MonitorElement* recHitMultLayer1;
	MonitorElement* recHitMultLayer2;
	MonitorElement* recHitMultLayer3;

	//RecHits FPIX
	MonitorElement* recHitXResAllF;
	MonitorElement* recHitYResAllF;
	MonitorElement* recHitXPlaquetteSize1;
	MonitorElement* recHitXPlaquetteSize2;
	MonitorElement* recHitYPlaquetteSize2;
	MonitorElement* recHitYPlaquetteSize3;
	MonitorElement* recHitYPlaquetteSize4;
	MonitorElement* recHitYPlaquetteSize5;
	MonitorElement* recHitXResDisk1Plaquettes[7];
	MonitorElement* recHitXResDisk2Plaquettes[7];
	MonitorElement* recHitYResDisk1Plaquettes[7];
	MonitorElement* recHitYResDisk2Plaquettes[7];

	// Pull distributions
	//RecHits BPIX
	MonitorElement* recHitXPullAllB;
	MonitorElement* recHitYPullAllB;

	MonitorElement* recHitXPullFlippedLadderLayers[3];
	MonitorElement* recHitXPullNonFlippedLadderLayers[3];
	MonitorElement* recHitYPullLayer1Modules[8];
	MonitorElement* recHitYPullLayer2Modules[8];
	MonitorElement* recHitYPullLayer3Modules[8];
	MonitorElement* recHitXYPosLayer1Modules[8];
	MonitorElement* recHitXYPosLayer2Modules[8];
	MonitorElement* recHitXYPosLayer3Modules[8];

	//RecHits FPIX
	MonitorElement* recHitXPullAllF;
	MonitorElement* recHitYPullAllF;

	MonitorElement* recHitXPullDisk1Plaquettes[7];
	MonitorElement* recHitXPullDisk2Plaquettes[7];
	MonitorElement* recHitYPullDisk1Plaquettes[7];
	MonitorElement* recHitYPullDisk2Plaquettes[7];
        MonitorElement* recHitXYPosDisk1Plaquettes[7];
	MonitorElement* recHitXYPosDisk2Plaquettes[7];
        edm::InputTag src_;
};

#endif
