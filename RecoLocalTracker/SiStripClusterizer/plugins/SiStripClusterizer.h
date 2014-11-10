#ifndef RecoLocalTracker_SiStripClusterizer_h
#define RecoLocalTracker_SiStripClusterizer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include <vector>
#include <memory>

class SiStripClusterizer : public edm::stream::EDProducer<>  {

public:

  explicit SiStripClusterizer(const edm::ParameterSet& conf);
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:

  edm::ParameterSet confClusterizer_;
  bool doRefineCluster_;
  float occupancyThreshold_;
  unsigned widthThreshold_;
  bool useAssociator_;
  template<class T> bool findInput(const edm::EDGetTokenT<T>&, edm::Handle<T>&, const edm::Event&);
  const std::vector<edm::InputTag> inputTags;
  std::auto_ptr<StripClusterizerAlgorithm> algorithm;
  void refineCluster(const edm::Handle< edm::DetSetVector<SiStripDigi> >& input,
		     std::auto_ptr< edmNew::DetSetVector<SiStripCluster> >& output,
		     SiStripDetInfoFileReader* reader,
		     edm::ESHandle<SiStripQuality> quality,
		     std::shared_ptr<TrackerHitAssociator> associator);
  typedef edm::EDGetTokenT< edm::DetSetVector<SiStripDigi> > token_t;
  typedef std::vector<token_t> token_v;
  token_v inputTokens;
  typedef std::vector<std::string> vstring;

};

#endif
