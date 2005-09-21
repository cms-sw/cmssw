// File: SiStrip1DLocalMeasurementConverterAlgorithm.cc
// Description:  An algorithm for CMS track reconstruction.
// Author:  O/ Gutsche
// Creation Date:  OGU Aug. 1, 2005   

#include <vector>
#include <iostream>

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStrip1DLocalMeasurementConverterAlgorithm.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStrip1DLocalMeasurement.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/Vector/interface/LocalPoint.h"

#include "Geometry/Surface/interface/LocalError.h"

#include "Geometry/CommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonDetAlgo/interface/MeasurementError.h"

#include "Geometry/TrackerSimAlgo/interface/TrackerGeom.h"
#include "Geometry/TrackerSimAlgo/interface/StripGeomDetUnit.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "Geometry/CommonTopologies/interface/Topology.h"

using namespace std;

SiStrip1DLocalMeasurementConverterAlgorithm::SiStrip1DLocalMeasurementConverterAlgorithm(const edm::ParameterSet& conf) : conf_(conf) {}

SiStrip1DLocalMeasurementConverterAlgorithm::~SiStrip1DLocalMeasurementConverterAlgorithm() {
}


void SiStrip1DLocalMeasurementConverterAlgorithm::run(const SiStripClusterCollection* input, const edm::EventSetup& es, SiStrip1DLocalMeasurementCollection &output)
{

  int number_detunits          = 0;
  int number_local1dmeasurements = 0;

  // get geometry
  edm::eventsetup::ESHandle<TrackerGeom> pDD;
  es.get<TrackerDigiGeometryRecord>().get(pDD);
  const TrackerGeom& tracker(*pDD);


  // get vector of detunit ids
  const std::vector<unsigned int> detIDs = input->detIDs();
    
  // loop over detunits
  for ( std::vector<unsigned int>::const_iterator detunit_iterator = detIDs.begin(); detunit_iterator != detIDs.end(); ++detunit_iterator ) {
    unsigned int id = *detunit_iterator;
    ++number_detunits;
    const SiStripClusterCollection::Range clusterRange = input->get(id);
    SiStripClusterCollection::ContainerIterator clusterRangeIteratorBegin = clusterRange.first;
    SiStripClusterCollection::ContainerIterator clusterRangeIteratorEnd   = clusterRange.second;
      
    vector<SiStrip1DLocalMeasurement> collector;

    for ( ; clusterRangeIteratorBegin != clusterRangeIteratorEnd; ++clusterRangeIteratorBegin ) {
      SiStripCluster cluster = *clusterRangeIteratorBegin;

      // check if detid valid
      cms::DetId id = cluster.geographicalId();
      if ( id.rawId() != 999999999 ) {

	// convert cluster to local measurement
	// get topology for detid
      
	GeomDetUnit *det = tracker.idToDet(cluster.geographicalId());
	const Topology& top = det->topology();
	
	// fill cluster in measurementpoint structure
	const MeasurementPoint measurementP(cluster.barycenter(),0.);
	const MeasurementError measurementE(cluster.barycenter_error(),0.,0.);
	
	// convert
	const LocalPoint localP = top.localPosition(measurementP);
	const LocalError localE = top.localError(measurementP,measurementE);
	
	collector.push_back(SiStrip1DLocalMeasurement(cluster.geographicalId(),localP.x(),localE.xx()));
      }
    }

    
    SiStrip1DLocalMeasurementCollection::Range inputRange;
    inputRange.first = collector.begin();
    inputRange.second = collector.end();
    output.put(inputRange,id);
    number_local1dmeasurements += collector.size();
  }

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    cout << "[SiStrip1DLocalMeasurementConverterAlgorithm] converted " << number_local1dmeasurements << " SiStripClusters in " << number_detunits << " DetUnits." << endl; 
  }
};
