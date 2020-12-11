#ifndef SHALLOW_TRACKCLUSTERS_PRODUCER
#define SHALLOW_TRACKCLUSTERS_PRODUCER

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"

class ShallowTrackClustersProducer : public edm::stream::EDProducer<> {
public:
  explicit ShallowTrackClustersProducer(const edm::ParameterSet &);

private:
  const edm::EDGetTokenT<edm::View<reco::Track> > tracks_token_;
  const edm::EDGetTokenT<TrajTrackAssociationCollection> association_token_;
  const edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clusters_token_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  const edm::ESGetToken<SiStripLorentzAngle, SiStripLorentzAngleDepRcd> laToken_;
  std::string Suffix;
  std::string Prefix;

  void produce(edm::Event &, const edm::EventSetup &) override;
};
#endif
