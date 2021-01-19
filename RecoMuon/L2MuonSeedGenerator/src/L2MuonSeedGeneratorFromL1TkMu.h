#ifndef RecoMuon_L2MuonSeedGenerator_L2MuonSeedGeneratorFromL1TkMu_H
#define RecoMuon_L2MuonSeedGenerator_L2MuonSeedGeneratorFromL1TkMu_H

/*  \class L2MuonSeedGeneratorFromL1TkMu
 *
 *   L2 muon seed generator:
 *   Transform the L1TkMuon informations in seeds
 *   for the L2 muon reconstruction
 *   (mimicking L2MuonSeedGeneratorFromL1T)
 *
 *    Author: H. Kwon
 *    Modified by M. Oh
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

// Data Formats
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

#include "DataFormats/L1TCorrelator/interface/TkMuon.h"
#include "DataFormats/L1TCorrelator/interface/TkMuonFwd.h"

#include "CLHEP/Vector/ThreeVector.h"

#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"

class MuonServiceProxy;
class MeasurementEstimator;
class TrajectorySeed;
class TrajectoryStateOnSurface;

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class L2MuonSeedGeneratorFromL1TkMu : public edm::stream::EDProducer<> {
public:
  /// Constructor
  explicit L2MuonSeedGeneratorFromL1TkMu(const edm::ParameterSet &);

  /// Destructor
  ~L2MuonSeedGeneratorFromL1TkMu() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  edm::InputTag theSource;
  edm::InputTag theOfflineSeedLabel;
  std::string thePropagatorName;

  edm::EDGetTokenT<l1t::TkMuonCollection> muCollToken_;
  edm::EDGetTokenT<edm::View<TrajectorySeed> > offlineSeedToken_;

  const double theL1MinPt;
  const double theL1MaxEta;
  const double theMinPtBarrel;
  const double theMinPtEndcap;
  const double theMinPL1Tk;
  const double theMinPtL1TkBarrel;
  const bool useOfflineSeed;
  const bool useUnassociatedL1;
  std::vector<double> matchingDR;
  std::vector<double> etaBins;

  /// the event setup proxy, it takes care the services update
  MuonServiceProxy *theService;

  MeasurementEstimator *theEstimator;

  const TrajectorySeed *associateOfflineSeedToL1(edm::Handle<edm::View<TrajectorySeed> > &,
                                                 std::vector<int> &,
                                                 TrajectoryStateOnSurface &,
                                                 double);

};

#endif
