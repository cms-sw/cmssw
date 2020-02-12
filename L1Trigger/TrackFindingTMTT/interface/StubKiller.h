// Copied from https://raw.githubusercontent.com/EmyrClement/StubKiller/master/StubKiller.cc
// on 9th May 2018.

#ifndef __STUBKILLER_H__
#define __STUBKILLER_H__

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "TRandom.h"
#include "TMath.h"

using namespace std;

class StubKiller {

public:
  
  StubKiller();
  ~StubKiller() {}

  void initialise( unsigned int killScenario, const TrackerTopology* trackerTopology, const TrackerGeometry* trackerGeometry );

  bool killStub(
  		const TTStub<Ref_Phase2TrackerDigi_>* stub,
		const vector<int> layersToKill,
		const int minPhiToKill,
		const int maxPhiToKill,
		const int minZToKill,
		const int maxZToKill,
		const int minRToKill,
		const int maxRToKill,
		const double fractionOfStubsToKillInLayers,
		const double fractionOfStubsToKillEverywhere
  	);

  bool killStub( const TTStub<Ref_Phase2TrackerDigi_>* stub );

  bool killStubInDeadModule( const TTStub<Ref_Phase2TrackerDigi_>* stub );

  map<DetId, float> getListOfDeadModules() { return deadModules_ ;}

private:
	void chooseModulesToKill();
	void addDeadLayerModulesToDeadModuleList();

	unsigned int killScenario_;
	const TrackerTopology* trackerTopology_;
	const TrackerGeometry* trackerGeometry_;

	vector<int> layersToKill_;
	int minPhiToKill_;
	int maxPhiToKill_;
	int minZToKill_;
	int maxZToKill_;
	int minRToKill_;
	int maxRToKill_;
	double fractionOfStubsToKillInLayers_;
	double fractionOfStubsToKillEverywhere_;
	double fractionOfModulesToKillEverywhere_;

	map<DetId, float> deadModules_;
};

#endif
