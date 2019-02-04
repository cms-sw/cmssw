#ifndef CSCTrackFinder_CSCTFTrackProducer_h
#define CSCTrackFinder_CSCTFTrackProducer_h

#include <string>

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "L1Trigger/CSCTrackFinder/src/CSCTFDTReceiver.h"
#include "L1Trigger/CSCTrackFinder/src/CSCTFTrackBuilder.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

class CSCTFTrackBuilder;
class L1MuDTChambPhContainer;
template<typename T> class CSCTriggerContainer;
namespace csctf {
  class TrackStub;
}

class CSCTFTrackProducer : public edm::one::EDProducer<edm::one::SharedResources>
{
 public:
  CSCTFTrackProducer(const edm::ParameterSet&);
  void produce(edm::Event & e, const edm::EventSetup& c) override;

 private:
  CSCTFDTReceiver my_dtrc;
  const bool useDT;
  const bool TMB07;
  const bool readDtDirect;
  const edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> input_module;
  const edm::EDGetTokenT<L1MuDTChambPhContainer> dt_producer;
  const edm::EDGetTokenT<CSCTriggerContainer<csctf::TrackStub> >  directProd;
  const edm::ParameterSet sp_pset ;
  unsigned long long m_scalesCacheID ;
  unsigned long long m_ptScaleCacheID ;
  std::unique_ptr<CSCTFTrackBuilder> my_builder;
};

#endif
