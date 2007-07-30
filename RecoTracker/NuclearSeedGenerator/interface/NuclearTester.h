#ifndef CD_NuclearTester_H_
#define CD_NuclearTester_H_
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"

class NuclearTester {
private:

  typedef TrajectoryMeasurement TM;
  typedef std::vector<TM> TMContainer;
  typedef TrajectoryMeasurement::ConstRecHitPointer    ConstRecHitPointer;

public :
  NuclearTester(const edm::EventSetup& es, const edm::ParameterSet& iConfig);

  bool isNuclearInteraction();

  double meanHitDistance(const std::vector<TrajectoryMeasurement>& vecTM) const;

  std::vector<GlobalPoint> HitPositions(const std::vector<TrajectoryMeasurement>& vecTM) const;

  double meanEstimate(const std::vector<TrajectoryMeasurement>& vecTM) const;

  std::vector<TM>::const_iterator lastValidTM(const std::vector<TM>& vecTM) const;

  void push_back(TMContainer vecTM) { allTM.push_back(vecTM); compatible_hits.push_back(vecTM.size()); }

  const TMContainer& back() const { return allTM.back(); }

  double meanHitDistance() const { return meanHitDistance( back() ); }

  double meanEstimate() const { return meanEstimate( back() ); }

  void reset() { allTM.clear(); compatible_hits.clear(); }

  int nuclearIndex() const { return NuclearIndex; }

  const TMContainer& compatibleTM() const { return *(allTM.begin()+nuclearIndex()); }

private :

  edm::ESHandle<TrackerGeometry>  trackerGeom;
  std::vector< TMContainer > allTM;
  std::vector< int > compatible_hits;
  int NuclearIndex;
  bool checkCompletedTrack;
};
#endif
