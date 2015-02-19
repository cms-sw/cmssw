#ifndef SiPixelRecHitsInputDistributionsMaker_h
#define SiPixelRecHitsInputDistributionsMaker_h

/** \class SiPixelRecHitsInputDistributionsMaker
 * File: SiPixelRecHitsInputDistributionsMaker.h
 * \author Mario Galanti, Universit\'a di Catania
 * Created: 31/1/2008
 * Based on class SiPixelRecHitsValid by Jason Shaev
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DWM histogram services
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

//#include "FWCore/ServiceRegistry/interface/Service.h"

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

#include "FWCore/Utilities/interface/InputTag.h"

#include <string>

class SiPixelRecHitsInputDistributionsMaker : public DQMEDAnalyzer {

   public:
	//Constructor
  
  SiPixelRecHitsInputDistributionsMaker(const edm::ParameterSet& conf);
  
  //Destructor
  ~SiPixelRecHitsInputDistributionsMaker();

   protected:

	void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override; 
	virtual void analyze(const edm::Event& e, const edm::EventSetup& c) override;
	void beginJob();
	void endJob();
	void dqmBeginRun(const edm::Run& r, const edm::EventSetup& c){};

   private:

	std::string outputFile_;

	edm::ParameterSet conf_;

	void fillBarrel(const SiPixelRecHit &,const PSimHit &, DetId, const PixelGeomDetUnit *);	
	void fillForward(const SiPixelRecHit &, const PSimHit &, DetId, const PixelGeomDetUnit *);

        // alpha and beta
        MonitorElement* simHitAlphaBarrel;
        MonitorElement* simHitAlphaForward;
        MonitorElement* simHitBetaBarrel;
        MonitorElement* simHitBetaForward;
        MonitorElement* simHitAlphaBarrelBigPixel;
        MonitorElement* simHitAlphaForwardBigPixel;
        MonitorElement* simHitBetaBarrelBigPixel;
        MonitorElement* simHitBetaForwardBigPixel;

        // alpha and beta according to multiplicities
        MonitorElement* simHitAlphaMultiBarrel[5];
        MonitorElement* simHitAlphaMultiForward[4];
        MonitorElement* simHitBetaMultiBarrel[8];
        MonitorElement* simHitBetaMultiForward[4];
        MonitorElement* simHitAlphaMultiBarrelBigPixel[5];
        MonitorElement* simHitAlphaMultiForwardBigPixel[4];
        MonitorElement* simHitBetaMultiBarrelBigPixel[8];
        MonitorElement* simHitBetaMultiForwardBigPixel[4];

        // RecHit resolutions in barrel according to beta multiplicity
        MonitorElement* recHitResBetaBarrel00[7];
        MonitorElement* recHitResBetaBarrel02[7];
        MonitorElement* recHitResBetaBarrel04[7];
        MonitorElement* recHitResBetaBarrel06[7];
        MonitorElement* recHitResBetaBarrel08[7];
        MonitorElement* recHitResBetaBarrel10[7];
        MonitorElement* recHitResBetaBarrel12[7];
        MonitorElement* recHitResBetaBarrelBigPixel00[7];
        MonitorElement* recHitResBetaBarrelBigPixel02[7];
        MonitorElement* recHitResBetaBarrelBigPixel04[7];
        MonitorElement* recHitResBetaBarrelBigPixel06[7];
        MonitorElement* recHitResBetaBarrelBigPixel08[7];
        MonitorElement* recHitResBetaBarrelBigPixel10[7];
        MonitorElement* recHitResBetaBarrelBigPixel12[7];

        // RecHit resolutions in barrel according to alpha multiplicity
        MonitorElement* recHitResAlphaBarrel0201[4];
        MonitorElement* recHitResAlphaBarrel0100[4];
        MonitorElement* recHitResAlphaBarrel0001[4];
        MonitorElement* recHitResAlphaBarrel0102[4];
        MonitorElement* recHitResAlphaBarrelBigPixel0201[4];
        MonitorElement* recHitResAlphaBarrelBigPixel0100[4];
        MonitorElement* recHitResAlphaBarrelBigPixel0001[4];
        MonitorElement* recHitResAlphaBarrelBigPixel0102[4];

        // RecHit X and Y resolutions in forward
        MonitorElement* recHitResXForward[3];
        MonitorElement* recHitResYForward[3];
        MonitorElement* recHitResXForwardBigPixel[3];
        MonitorElement* recHitResYForwardBigPixel[3];

        std::vector<std::string> trackerContainers;

        edm::InputTag src_;
};

#endif
