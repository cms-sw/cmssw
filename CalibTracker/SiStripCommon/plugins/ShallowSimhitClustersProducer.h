#ifndef SHALLOW_SIMHITCLUSTERS_PRODUCER
#define SHALLOW_SIMHITCLUSTERS_PRODUCER

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CalibTracker/SiStripCommon/interface/ShallowTools.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

class ShallowSimhitClustersProducer : public edm::stream::EDProducer<> {
public:
  explicit ShallowSimhitClustersProducer(const edm::ParameterSet&);

private:
  std::vector<edm::EDGetTokenT<std::vector<PSimHit> > > simhits_tokens_;
  const edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clusters_token_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  const edm::ESGetToken<SiStripLorentzAngle, SiStripLorentzAngleRcd> laToken_;
  std::string Prefix;
  std::string runningmode_;

  void produce(edm::Event&, const edm::EventSetup&) override;
  shallow::CLUSTERMAP::const_iterator match_cluster(const unsigned&,
                                                    const float&,
                                                    const shallow::CLUSTERMAP&,
                                                    const edmNew::DetSetVector<SiStripCluster>&) const;
};
#endif
