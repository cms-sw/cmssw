#ifndef RecoLocalTracker_SiStripCompressionAlgorithm_h
#define RecoLocalTracker_SiStripCompressionAlgorithm_h

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDetSetCompressedCluster/interface/SiStripDetSetCompressedCluster.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

typedef edm::DetSetVector<SiStripDigi> vdigis_t;
typedef edm::DetSet<SiStripDigi> digis_t;

typedef edmNew::DetSetVector<SiStripCluster> vclusters_t;
typedef edmNew::DetSet<SiStripCluster> clusters_t;

typedef edmNew::DetSetVector<SiStripDetSetCompressedCluster> vcomp_clusters_t;
typedef edmNew::DetSet<SiStripDetSetCompressedCluster> comp_clusters_t;

class SiStripCompressionAlgorithm {
public:
  explicit SiStripCompressionAlgorithm();
  virtual ~SiStripCompressionAlgorithm() {}

  void compress(vclusters_t const&, vcomp_clusters_t&);

private:
  void LoadRealModelDataFromFile();
  void commpressDetModule(const clusters_t&, vcomp_clusters_t::TSFastFiller&);
};
#endif