#ifndef SHALLOW_CLUSTERS_PRODUCER
#define SHALLOW_CLUSTERS_PRODUCER

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h"

class SiStripProcessedRawDigi;
class TrackerTopology;

class ShallowClustersProducer : public edm::EDProducer {
public:
  explicit ShallowClustersProducer(const edm::ParameterSet &);

private:
  edm::InputTag theClustersLabel;
  std::string Prefix;
  void produce(edm::Event &, const edm::EventSetup &) override;

  struct moduleVars {
    moduleVars(uint32_t, const TrackerTopology *);
    int subdetid, side, layerwheel, stringringrod, petal, stereo;
    uint32_t module;
  };

  struct NearDigis {
    NearDigis(const SiStripClusterInfo &);
    NearDigis(const SiStripClusterInfo &, const edm::DetSetVector<SiStripProcessedRawDigi> &);
    float max, left, right, first, last, Lleft, Rright;
    float etaX() const { return ((left + right) / max) / 2.; }
    float eta() const { return right > left ? max / (max + right) : left / (left + max); }
    float etaasymm() const { return right > left ? (right - max) / (right + max) : (max - left) / (max + left); }
    float outsideasymm() const { return (last - first) / (last + first); }
  };

  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > theClustersToken_;
  edm::EDGetTokenT<edm::DetSetVector<SiStripProcessedRawDigi> > theDigisToken_;
  SiStripClusterInfo siStripClusterInfo_;
};

#endif
