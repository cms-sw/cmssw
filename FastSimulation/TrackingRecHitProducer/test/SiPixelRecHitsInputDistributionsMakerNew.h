#ifndef SiPixelRecHitsInputDistributionsMakerNew_h
#define SiPixelRecHitsInputDistributionsMakerNew_h

/** \class SiPixelRecHitsInputDistributionsMaker
 * File: SiPixelRecHitsInputDistributionsMaker.h
 * \author Mario Galanti, Universit\'a di Catania
 * Created: 31/1/2008
 * Based on class SiPixelRecHitsValid by Jason Shaev
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DWM histogram services
//#include "DQMServices/Core/interface/DQMStore.h"
//#include "DQMServices/Core/interface/MonitorElement.h"
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

class SiPixelRecHitsInputDistributionsMakerNew : public DQMEDAnalyzer {

   public:
	//Constructor
	SiPixelRecHitsInputDistributionsMakerNew(const edm::ParameterSet& conf);

	//Destructor
	~SiPixelRecHitsInputDistributionsMakerNew();

   protected:

	void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup &);
	virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

   private:

	edm::ParameterSet conf_;

        // qbin, cotAlpha, cotBeta
        static const int qBins_ = 4;
        static const int qBinWidth_ = 1;
        double   cotAlphaLowEdgeBarrel_;
        double   cotAlphaBinWidthBarrel_;
        static const int      cotAlphaBinsBarrel_     = 5;
        double   cotBetaLowEdgeBarrel_;
        double   cotBetaBinWidthBarrel_;
        static const int      cotBetaBinsBarrel_      = 11;
	static const int cotAlphaBinsForward_	= 4;
	static const int cotBetaBinsForward_	= 4;
	double   cotAlphaLowEdgeForward_;
	double   cotAlphaBinWidthForward_;
	double   cotBetaLowEdgeForward_;
	double   cotBetaBinWidthForward_;

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

        // RecHit resolutions in barrel according to cotAlpha, cotBeta, qBin
        MonitorElement* recHitResBarrelEdgeX[cotAlphaBinsBarrel_][cotBetaBinsBarrel_][qBins_];
        MonitorElement* recHitResBarrelEdgeY[cotAlphaBinsBarrel_][cotBetaBinsBarrel_][qBins_];
        MonitorElement* recHitResBarrelMultiPixBigX[cotAlphaBinsBarrel_][cotBetaBinsBarrel_][qBins_];
        MonitorElement* recHitResBarrelSinglePixBigX[cotAlphaBinsBarrel_][cotBetaBinsBarrel_];
        MonitorElement* recHitResBarrelMultiPixX[cotAlphaBinsBarrel_][cotBetaBinsBarrel_][qBins_];
        MonitorElement* recHitResBarrelSinglePixX[cotAlphaBinsBarrel_][cotBetaBinsBarrel_];
        MonitorElement* recHitResBarrelMultiPixBigY[cotAlphaBinsBarrel_][cotBetaBinsBarrel_][qBins_];
        MonitorElement* recHitResBarrelSinglePixBigY[cotAlphaBinsBarrel_][cotBetaBinsBarrel_];
        MonitorElement* recHitResBarrelMultiPixY[cotAlphaBinsBarrel_][cotBetaBinsBarrel_][qBins_];
        MonitorElement* recHitResBarrelSinglePixY[cotAlphaBinsBarrel_][cotBetaBinsBarrel_];

	MonitorElement* recHitClusterInfo;
	MonitorElement* recHitcotAlpha[10];
	MonitorElement* recHitcotBeta[10];
        MonitorElement* recHitqBin[11];

        // RecHit X and Y resolutions in forward
        MonitorElement* recHitResForwardEdgeX[cotAlphaBinsForward_][cotBetaBinsForward_][qBins_];
        MonitorElement* recHitResForwardEdgeY[cotAlphaBinsForward_][cotBetaBinsForward_][qBins_];
        MonitorElement* recHitResForwardX[cotAlphaBinsForward_][cotBetaBinsForward_][qBins_];
	MonitorElement* recHitResForwardY[cotAlphaBinsForward_][cotBetaBinsForward_][qBins_];
	MonitorElement* recHitResForwardSingleBX[cotAlphaBinsForward_][cotBetaBinsForward_];
	MonitorElement* recHitResForwardSingleBY[cotAlphaBinsForward_][cotBetaBinsForward_];
	MonitorElement* recHitResForwardSingleX[cotAlphaBinsForward_][cotBetaBinsForward_];
	MonitorElement* recHitResForwardSingleY[cotAlphaBinsForward_][cotBetaBinsForward_];

        edm::InputTag src_;

};

#endif
