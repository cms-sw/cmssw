// Copied from https://raw.githubusercontent.com/EmyrClement/StubKiller/master/StubKiller.cc
// on 9th May 2018.

// Changed to point to location we put it.
//#include "StubKiller.h"
#include "L1Trigger/TrackFindingTMTT/interface/StubKiller.h"

using namespace std;

StubKiller::StubKiller() :
	killScenario_(0),
	trackerTopology_(0),
	trackerGeometry_(0),
	layersToKill_(vector<int>()),
	minPhiToKill_(0),
	maxPhiToKill_(0),
	minZToKill_(0),
	maxZToKill_(0),
	minRToKill_(0),
	maxRToKill_(0),
	fractionOfStubsToKillInLayers_(0),
	fractionOfStubsToKillEverywhere_(0),
	fractionOfModulesToKillEverywhere_(0)
{
}

void StubKiller::initialise(unsigned int killScenario, const TrackerTopology* trackerTopology, const TrackerGeometry* trackerGeometry)
{
	killScenario_ = killScenario;
	trackerTopology_ = trackerTopology;
	trackerGeometry_ = trackerGeometry;

	// These sceanrios correspond to slide 12 of  https://indico.cern.ch/event/719985/contributions/2970687/attachments/1634587/2607365/StressTestTF-Acosta-Apr18.pdf
	// Sceanrio 1
	// kill layer 5 in one quadrant +5 % random module loss to connect to what was done before
	if ( killScenario_ == 1 ) {
		layersToKill_ = {5};
		minPhiToKill_ = 0;
		maxPhiToKill_ = TMath::PiOver2();
		minZToKill_ = -1000;
		maxZToKill_ = 0;
		minRToKill_ = 0;
		maxRToKill_ = 1000;
		fractionOfStubsToKillInLayers_ = 1;
		fractionOfStubsToKillEverywhere_ = 0;
		fractionOfModulesToKillEverywhere_ = 0.05;
	}
	// Sceanrio 2
	// kill layer 1 in one quadrant +5 % random module loss
	else if ( killScenario_ == 2 ) {
		layersToKill_ = {1};
		minPhiToKill_ = 0;
		maxPhiToKill_ = TMath::PiOver2();
		minZToKill_ = -1000;
		maxZToKill_ = 0;
		minRToKill_ = 0;
		maxRToKill_ = 1000;
		fractionOfStubsToKillInLayers_ = 1;
		fractionOfStubsToKillEverywhere_ = 0;
		fractionOfModulesToKillEverywhere_ = 0.05;
	}
	// Scenario 3
	// kill layer 1 + layer 2, both in same quadrant
	else if ( killScenario_ == 3 ) {
		layersToKill_ = {1, 2};
		minPhiToKill_ = 0;
		maxPhiToKill_ = TMath::PiOver2();
		minZToKill_ = -1000;
		maxZToKill_ = 0;
		minRToKill_ = 0;
		maxRToKill_ = 1000;
		fractionOfStubsToKillInLayers_ = 1;
		fractionOfStubsToKillEverywhere_ = 0;
		fractionOfModulesToKillEverywhere_ = 0;
	}
	// Scenario 4
	// kill layer 1 and disk 1, both in same quadrant
	else if ( killScenario_ == 4 ) {
		layersToKill_ = {1, 11};
		minPhiToKill_ = 0;
		maxPhiToKill_ = TMath::PiOver2();
		minZToKill_ = -1000;
		maxZToKill_ = 0;
		minRToKill_ = 0;
		maxRToKill_ = 66.5;
		fractionOfStubsToKillInLayers_ = 1;
		fractionOfStubsToKillEverywhere_ = 0;
		fractionOfModulesToKillEverywhere_ = 0;
	}
	// An extra scenario not listed in the slides
	// 5% random module loss throughout tracker
	else if ( killScenario_ == 5 ) {
		layersToKill_ = {};
		fractionOfStubsToKillInLayers_ = 0;
		fractionOfStubsToKillEverywhere_ = 0.;
		fractionOfModulesToKillEverywhere_ = 0.05;
	}

	deadModules_.clear();
	if ( fractionOfModulesToKillEverywhere_ > 0 ) {
		this->chooseModulesToKill();
	}
	this->addDeadLayerModulesToDeadModuleList();
}

void StubKiller::chooseModulesToKill() {
	TRandom randomGenerator;
	for (const GeomDetUnit* gd : trackerGeometry_->detUnits()) {
		if ( !trackerTopology_->isLower( gd->geographicalId() ) ) continue;
		if ( randomGenerator.Rndm() < fractionOfModulesToKillEverywhere_ ) {
			deadModules_[ gd->geographicalId() ] = 1;
		}
	}
}

void StubKiller::addDeadLayerModulesToDeadModuleList() {
	for (const GeomDetUnit* gd : trackerGeometry_->detUnits()) {
		float moduleR = gd->position().perp();
		float moduleZ = gd->position().z();
		float modulePhi = gd->position().phi();
		DetId geoDetId = gd->geographicalId();
		bool isInBarrel = geoDetId.subdetId()==StripSubdetector::TOB || geoDetId.subdetId()==StripSubdetector::TIB;

		int layerID = 0;
		if (isInBarrel) {
		  layerID = trackerTopology_->layer( geoDetId );
		} else {
		  layerID = 10*trackerTopology_->side( geoDetId ) + trackerTopology_->tidWheel( geoDetId );
		}
		if ( find(layersToKill_.begin(), layersToKill_.end(), layerID ) != layersToKill_.end() ) {
			if ( modulePhi < -1.0 * TMath::Pi() ) modulePhi += 2.0 * TMath::Pi();
			else if ( modulePhi > TMath::Pi() ) modulePhi -= 2.0 * TMath::Pi();

			if ( modulePhi > minPhiToKill_ && modulePhi < maxPhiToKill_ &&
				moduleZ > minZToKill_ && moduleZ < maxZToKill_ &&
				moduleR > minRToKill_ && moduleR < maxRToKill_ ) {
			
				if ( deadModules_.find( gd->geographicalId() ) == deadModules_.end() ) {
					deadModules_[ gd->geographicalId() ] = fractionOfStubsToKillInLayers_;
				}
			}
		}
	}
}

bool StubKiller::killStub( const TTStub<Ref_Phase2TrackerDigi_>* stub ) {
	if ( killScenario_ == 0 ) return false;
	else {
		bool killStubRandomly = killStub( stub, layersToKill_, minPhiToKill_, maxPhiToKill_,
						minZToKill_, maxZToKill_, minRToKill_, maxRToKill_,
						fractionOfStubsToKillInLayers_, fractionOfStubsToKillEverywhere_ );
		bool killStubInDeadModules = killStubInDeadModule( stub );
		return killStubRandomly || killStubInDeadModules;
	}
}

// layersToKill - a vector stating the layers we are killing stubs in.  Can be an empty vector.
// Barrel layers are encoded as 1-6. The endcap layers are encoded as 11-15 (-z) and 21-25 (+z)
// min/max Phi/Z/R - stubs within the region specified by these boundaries and layersToKill are flagged for killing
// fractionOfStubsToKillInLayers - The fraction of stubs to kill in the specified layers/region.
// fractionOfStubsToKillEverywhere - The fraction of stubs to kill throughout the tracker

bool StubKiller::killStub( const TTStub<Ref_Phase2TrackerDigi_>* stub,
		const vector<int> layersToKill,
		const int minPhiToKill,
		const int maxPhiToKill,
		const int minZToKill,
		const int maxZToKill,
		const int minRToKill,
		const int maxRToKill,
		const double fractionOfStubsToKillInLayers,
		const double fractionOfStubsToKillEverywhere
	) {

	// Only kill stubs in specified layers
	if ( layersToKill.size() > 0 ) {
		// Get the layer the stub is in, and check if it's in the layer you want to kill
		DetId stackDetid = stub->getDetId();
		DetId geoDetId(stackDetid.rawId() + 1);

		bool isInBarrel = geoDetId.subdetId()==StripSubdetector::TOB || geoDetId.subdetId()==StripSubdetector::TIB;

		int layerID = 0;
		if (isInBarrel) {
		  layerID = trackerTopology_->layer( geoDetId );
		} else {
		  layerID = 10*trackerTopology_->side( geoDetId ) + trackerTopology_->tidWheel( geoDetId );
		}

		if ( find(layersToKill.begin(), layersToKill.end(), layerID ) != layersToKill.end() ) {
			// Get the phi and z of stub, and check if it's in the region you want to kill
			const GeomDetUnit* det0 = trackerGeometry_->idToDetUnit( geoDetId );
			const PixelGeomDetUnit* theGeomDet = dynamic_cast< const PixelGeomDetUnit* >( det0 );
			const PixelTopology* topol = dynamic_cast< const PixelTopology* >( &(theGeomDet->specificTopology()) );
			MeasurementPoint measurementPoint = stub->clusterRef(0)->findAverageLocalCoordinatesCentered();
			LocalPoint clustlp   = topol->localPosition(measurementPoint);
			GlobalPoint pos  =  theGeomDet->surface().toGlobal(clustlp);

			// Just in case phi is outside of -pi -> pi
			double stubPhi = pos.phi();
			if ( stubPhi < -1.0 * TMath::Pi() ) stubPhi += 2.0 * TMath::Pi();
			else if ( stubPhi > TMath::Pi() ) stubPhi -= 2.0 * TMath::Pi();

			if ( stubPhi > minPhiToKill && stubPhi < maxPhiToKill &&
				pos.z() > minZToKill && pos.z() < maxZToKill &&
				pos.perp() > minRToKill && pos.perp() < maxRToKill ) {

				// Kill fraction of stubs
				if ( fractionOfStubsToKillInLayers == 1 ) {
					return true;
				}
				else {
					static TRandom randomGenerator;
					if ( randomGenerator.Rndm() < fractionOfStubsToKillInLayers ) {
						return true;
					}					
				}
			}
		}
	}

	// Kill fraction of stubs throughout tracker
	if ( fractionOfStubsToKillEverywhere > 0 ) {
		static TRandom randomGenerator;
		if ( randomGenerator.Rndm() < fractionOfStubsToKillEverywhere ) {
			return true;
		}
	}

	return false;
}

bool StubKiller::killStubInDeadModule( const TTStub<Ref_Phase2TrackerDigi_>* stub ) {
	if ( deadModules_.size() > 0 ) {
		DetId stackDetid = stub->getDetId();
		DetId geoDetId(stackDetid.rawId() + 1);
		if ( deadModules_.find( geoDetId ) != deadModules_.end() ) return true;
	}

	return false;
}
