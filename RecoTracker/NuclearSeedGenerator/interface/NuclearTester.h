//----------------------------------------------------------------------------
//! \class NuclearTester
//! \brief Class used to test if a track has interacted nuclearly
//!
//! \description Using the properties of all the compatible TMs of the TMs associated to a track, the method
//! isNuclearInteraction return 1 in case the track has interacted nuclearly, 0 else. 
//-----------------------------------------------------------------------------
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
  typedef std::pair<TrajectoryMeasurement, TMContainer > TMPair;
  typedef std::vector< TMPair >  TMPairVector;

public :
  NuclearTester(const edm::EventSetup& es, const edm::ParameterSet& iConfig);

  bool isNuclearInteraction();

  double meanHitDistance(const std::vector<TrajectoryMeasurement>& vecTM) const;

  std::vector<GlobalPoint> HitPositions(const std::vector<TrajectoryMeasurement>& vecTM) const;

  double meanEstimate(const std::vector<TrajectoryMeasurement>& vecTM) const;

  std::vector<TM>::const_iterator lastValidTM(const std::vector<TM>& vecTM) const;

  void push_back(const TM & init_tm, const TMContainer & vecTM) { 
             allTM.push_back(std::make_pair(init_tm, vecTM) ); 
             compatible_hits.push_back(vecTM.size()); 
  }

  const TMContainer& back() const { return allTM.back().second; }

  double meanHitDistance() const { return meanHitDistance( back() ); }

  double meanEstimate() const { return meanEstimate( back() ); }

  void reset(unsigned int nMeasurements) { 
               allTM.clear(); 
               compatible_hits.clear(); 
               maxHits = (nMeasurements < maxHits) ? nMeasurements : maxHits; 
  }

  int nuclearIndex() const { return NuclearIndex; }

  const TMPair& goodTMPair() const { return *(allTM.begin()+nuclearIndex()-1); }

  unsigned int nHitsChecked() const { return compatible_hits.size(); }

  std::vector<int> compatibleHits() const { return compatible_hits; }
  
private :

  edm::ESHandle<TrackerGeometry>  trackerGeom;
  TMPairVector allTM;
  std::vector< int > compatible_hits;
  int NuclearIndex;
  unsigned int maxHits;
  bool checkWithMultiplicity();
};
#endif
