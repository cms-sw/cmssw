/** \class MuonMillepedeTrackRefitter
*
*  This class produces a collection of TrackForAlignment using as input
*  Tracks and 4DSegments from AlcaReco.
*  Calculation of predicted states is performed here.
*
*  $Date: 2009/12/15 15:07:16 $
*  $Revision: 1.1 $
*  \author P. Martinez Ruiz del Arbol, IFCA (CSIC-UC)  <Pablo.Martinez@cern.ch>
*/

#ifndef Alignment_MuonMillepedeTrackRefitter_MuonMillepedeTrackRefitter_H
#define Alignment_MuonMillepedeTrackRefitter_MuonMillepedeTrackRefitter_H

// Base Class Headers
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"

#include <string>
#include <vector>

class SegmentToTrackAssociator;
class TrackingRecHit;

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
  class InputTag;
}  // namespace edm

typedef std::vector<std::vector<int> > intDVector;
typedef std::vector<TrackingRecHit*> RecHitVector;

class MuonMillepedeTrackRefitter : public edm::stream::EDProducer<> {
public:
  typedef AlignmentAlgorithmBase::ConstTrajTrackPair ConstTrajTrackPair;
  typedef AlignmentAlgorithmBase::ConstTrajTrackPairCollection ConstTrajTrackPairCollection;

  /// Constructor
  MuonMillepedeTrackRefitter(const edm::ParameterSet& pset);

  /// Destructor
  ~MuonMillepedeTrackRefitter() override;

  // Operations

  void produce(edm::Event& event, const edm::EventSetup& eventSetup) override;

protected:
private:
  edm::ESHandle<MagneticField> theMGField;
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;

  edm::EDGetToken tracksSAToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> theMGFieldToken_;
  edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> theTrackingGeometryToken_;
};
#endif
