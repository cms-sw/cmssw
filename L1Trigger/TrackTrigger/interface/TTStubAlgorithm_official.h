/*! \class   TTStubAlgorithm_official
 *  \brief   Class for "official" algorithm to be used
 *           in TTStubBuilder
 *  \details HW-friendly algorithm: layer-wise LUT.
 *           After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \author Sebastien Viret
 *  \date   2013, Jul 18
 *
 */

#ifndef L1_TRACK_TRIGGER_STUB_ALGO_official_H
#define L1_TRACK_TRIGGER_STUB_ALGO_official_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithmRecord.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include <memory>
#include <string>
#include <map>
#include <typeinfo>

template <typename T>
class TTStubAlgorithm_official : public TTStubAlgorithm<T> {
private:
  /// Data members
  bool mPerformZMatchingPS;
  bool mPerformZMatching2S;
  bool m_tilted;
  std::string className_;

  std::vector<double> barrelCut;
  std::vector<std::vector<double>> ringCut;
  std::vector<std::vector<double>> tiltedCut;
  std::vector<double> barrelNTilt;

public:
  /// Constructor
  TTStubAlgorithm_official(const TrackerGeometry *const theTrackerGeom,
                           const TrackerTopology *const theTrackerTopo,
                           std::vector<double> setBarrelCut,
                           std::vector<std::vector<double>> setRingCut,
                           std::vector<std::vector<double>> setTiltedCut,
                           std::vector<double> setBarrelNTilt,
                           bool aPerformZMatchingPS,
                           bool aPerformZMatching2S)
      : TTStubAlgorithm<T>(theTrackerGeom, theTrackerTopo, __func__) {
    barrelCut = setBarrelCut;
    ringCut = setRingCut;
    tiltedCut = setTiltedCut;
    barrelNTilt = setBarrelNTilt;
    mPerformZMatchingPS = aPerformZMatchingPS;
    mPerformZMatching2S = aPerformZMatching2S;
  }

  /// Destructor
  ~TTStubAlgorithm_official() override {}

  /// Matching operations
  void PatternHitCorrelation(bool &aConfirmation,
                             int &aDisplacement,
                             int &anOffset,
                             float &anHardBend,
                             const TTStub<T> &aTTStub) const override;

  float degradeBend(bool psModule, int window, int bend) const;

};  /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Matching operations
template <>
void TTStubAlgorithm_official<Ref_Phase2TrackerDigi_>::PatternHitCorrelation(
    bool &aConfirmation,
    int &aDisplacement,
    int &anOffset,
    float &anHardBend,
    const TTStub<Ref_Phase2TrackerDigi_> &aTTStub) const;

template <>
float TTStubAlgorithm_official<Ref_Phase2TrackerDigi_>::degradeBend(bool psModule, int window, int bend) const;

/*! \class   ES_TTStubAlgorithm_official
 *  \brief   Class to declare the algorithm to the framework
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

template <typename T>
class ES_TTStubAlgorithm_official : public edm::ESProducer {
private:
  /// Data members
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> mGeomToken;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> mTopoToken;

  /// Windows
  std::vector<double> setBarrelCut;
  std::vector<std::vector<double>> setRingCut;
  std::vector<std::vector<double>> setTiltedCut;

  std::vector<double> setBarrelNTilt;

  /// Z-matching
  bool mPerformZMatchingPS;
  bool mPerformZMatching2S;

public:
  /// Constructor
  ES_TTStubAlgorithm_official(const edm::ParameterSet &p) {
    mPerformZMatchingPS = p.getParameter<bool>("zMatchingPS");
    mPerformZMatching2S = p.getParameter<bool>("zMatching2S");
    setBarrelCut = p.getParameter<std::vector<double>>("BarrelCut");
    setBarrelNTilt = p.getParameter<std::vector<double>>("NTiltedRings");

    std::vector<edm::ParameterSet> vPSet = p.getParameter<std::vector<edm::ParameterSet>>("EndcapCutSet");
    std::vector<edm::ParameterSet> vPSet2 = p.getParameter<std::vector<edm::ParameterSet>>("TiltedBarrelCutSet");

    std::vector<edm::ParameterSet>::const_iterator iPSet;
    for (iPSet = vPSet.begin(); iPSet != vPSet.end(); iPSet++) {
      setRingCut.push_back(iPSet->getParameter<std::vector<double>>("EndcapCut"));
    }

    for (iPSet = vPSet2.begin(); iPSet != vPSet2.end(); iPSet++) {
      setTiltedCut.push_back(iPSet->getParameter<std::vector<double>>("TiltedCut"));
    }

    setWhatProduced(this).setConsumes(mGeomToken).setConsumes(mTopoToken);
  }

  /// Destructor
  ~ES_TTStubAlgorithm_official() override {}

  /// Implement the producer
  std::unique_ptr<TTStubAlgorithm<T>> produce(const TTStubAlgorithmRecord &record) {
    return std::make_unique<TTStubAlgorithm_official<T>>(&record.get(mGeomToken),
                                                         &record.get(mTopoToken),
                                                         setBarrelCut,
                                                         setRingCut,
                                                         setTiltedCut,
                                                         setBarrelNTilt,
                                                         mPerformZMatchingPS,
                                                         mPerformZMatching2S);
  }
};

#endif
