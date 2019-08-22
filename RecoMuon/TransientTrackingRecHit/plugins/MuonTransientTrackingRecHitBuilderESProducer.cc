/** \class MuonTransientTrackingRecHitBuilderESProducer
 *  ESProducer for the Muon Transient TrackingRecHit Builder. The Builder can be taken from the 
 *  EventSetup, decoupling the code in which it is used w.r.t. the RecoMuon/TransientTrackingRecHit
 *  lib.
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include <memory>

class MuonTransientTrackingRecHitBuilderESProducer : public edm::ESProducer {
public:
  /// Constructor
  MuonTransientTrackingRecHitBuilderESProducer(const edm::ParameterSet&);

  // Operations
  std::unique_ptr<TransientTrackingRecHitBuilder> produce(const TransientRecHitRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> const trackingGeometryToken_;
};

using namespace edm;
using namespace std;

MuonTransientTrackingRecHitBuilderESProducer::MuonTransientTrackingRecHitBuilderESProducer(
    const ParameterSet& parameterSet)
    : trackingGeometryToken_(setWhatProduced(this, parameterSet.getParameter<string>("ComponentName"))
                                 .consumesFrom<GlobalTrackingGeometry, GlobalTrackingGeometryRecord>()) {}

std::unique_ptr<TransientTrackingRecHitBuilder> MuonTransientTrackingRecHitBuilderESProducer::produce(
    const TransientRecHitRecord& iRecord) {
  return std::make_unique<MuonTransientTrackingRecHitBuilder>(iRecord.getHandle(trackingGeometryToken_));
}

void MuonTransientTrackingRecHitBuilderESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<string>("ComponentName");
  descriptions.addDefault(desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(MuonTransientTrackingRecHitBuilderESProducer);
