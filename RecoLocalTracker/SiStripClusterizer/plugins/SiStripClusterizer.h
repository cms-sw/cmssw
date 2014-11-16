#ifndef RecoLocalTracker_SiStripClusterizer_h
#define RecoLocalTracker_SiStripClusterizer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include <vector>
#include <memory>

class SiStripClusterizer : public edm::stream::EDProducer<>  {

public:

  explicit SiStripClusterizer(const edm::ParameterSet& conf);
  void beginRun(const edm::Run& run, const edm::EventSetup& es) override;
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:

  edm::ParameterSet confClusterizer_;
  bool doRefineCluster_;
  float occupancyThreshold_;
  unsigned widthThreshold_;
  template<class T> bool findInput(const edm::EDGetTokenT<T>&, edm::Handle<T>&, const edm::Event&);
  const std::vector<edm::InputTag> inputTags;
  std::auto_ptr<StripClusterizerAlgorithm> algorithm;
  void refineCluster(const edm::Handle< edm::DetSetVector<SiStripDigi> >& input,
		     std::auto_ptr< edmNew::DetSetVector<SiStripCluster> >& output);
  typedef edm::EDGetTokenT< edm::DetSetVector<SiStripDigi> > token_t;
  typedef std::vector<token_t> token_v;
  token_v inputTokens;

  edm::ESHandle<SiStripDetCabling> SiStripDetCabling_;
  edm::ESHandle<SiStripQuality> quality_;
};

#endif
